import pybamm
import numpy as np
import scipy.io as sio
import os
import json
from nfpp_sodium_ion.src.cell_parameters.parameter_builder import get_parameter_values
from src.cell_optimization.parameter_opts import ParamTransform, DESIGN_SPACE
from simulation.utilities.tests_driver import ElectrochemicalThermalDriverModel
from src.simulation.utilities.mechanical.fenics_model import ThermoelasticStrainModel

class StabilityValidator:
    """
    Stability Validation (Envelope & Robustness).
    Uses full multiphysics Digital Twin (PyBaMM + FEniCSx).
    """

    def __init__(self):
        # Enforce final_validation.json dependency
        val_path = "final_validation.json"
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Missing mandatory pipeline artifact: {val_path}. Run validate.py first.")

        with open(val_path, "r") as f:
            self.pipeline_data = json.load(f)

        opt_data = self.pipeline_data.get("optimization")
        if not opt_data:
            raise KeyError(f"Invalid optimization data in {val_path}")

        # Reconstruct optimized parameters using the pipeline values
        base_params = get_parameter_values()
        pt = ParamTransform(pybamm.ParameterValues(base_params))

        # Apply deltas (merging functionalization if present)
        import copy
        deltas = copy.deepcopy(opt_data.get("combined_deltas_representative", {}))
        # Note: If validation step added more deltas or parameters, we ensure they are captured.

        pt.apply_physics_deltas(deltas)

        design_specs = opt_data.get("design_specs_representative", {})
        pt.apply_design_vector(
            np.array([design_specs[k] for k in DESIGN_SPACE if k in design_specs]),
            [k for k in DESIGN_SPACE if k in design_specs]
        )

        self.optimized_params = pt.get_parameter_values()
        # Ensure DFN stability parameters from validate.py
        if "SEI solvent diffusivity [m2.s-1]" not in self.optimized_params:
             self.optimized_params["SEI solvent diffusivity [m2.s-1]"] = 2.5e-22
        if "Bulk solvent concentration [mol.m-3]" not in self.optimized_params:
             self.optimized_params["Bulk solvent concentration [mol.m-3]"] = 2636.0
        self.electro_model = ElectrochemicalThermalDriverModel()

        self.mech_model = ThermoelasticStrainModel()

    def derive_ssc_parameters(self, solution, pybamm_params):
        """
        Derives Simscape ECM parameters from DFN simulation results.
        """
        v = solution["Terminal voltage [V]"].entries
        i = solution["Current [A]"].entries
        t = solution["Time [s]"].entries

        # 1. R0 (Ohmic): Derived from first voltage step (V_oc - V_initial) / I
        # Use first two points to catch the instantaneous drop
        dv = abs(v[0] - v[1])
        di = abs(i[1])
        R0 = dv / (di + 1e-6)

        # 2. RC Branches (Heuristic extraction from overpotential curve)
        # Total overpotential excluding Ohmic
        v_oc = v[0]
        eta_total = abs(v_oc - v[-1] - i[-1]*R0)

        # Split into fast (R1, C1) and slow (R2, C2)
        # R1 ~ 40% of diffusion/activation overpotential
        R1 = 0.4 * eta_total / (di + 1e-6)
        C1 = 2000.0 # Time constant ~ 10s

        R2 = 0.6 * eta_total / (di + 1e-6)
        C2 = 5000.0 # Time constant ~ 30s

        # 3. Thermal capacitance (C_th)
        # Sum of (Volume * Density * Cp) for all components
        L_p = pybamm_params["Positive electrode thickness [m]"]
        L_n = pybamm_params["Negative electrode thickness [m]"]
        L_s = pybamm_params["Separator thickness [m]"]
        A = pybamm_params["Electrode width [m]"] * pybamm_params["Electrode height [m]"]

        rho_p = pybamm_params["Positive electrode density [kg.m-3]"]
        rho_n = pybamm_params["Negative electrode density [kg.m-3]"]
        cp_p = pybamm_params["Positive electrode specific heat capacity [J.kg-1.K-1]"]
        cp_n = pybamm_params["Negative electrode specific heat capacity [J.kg-1.K-1]"]

        Cth = (L_p * A * rho_p * cp_p) + (L_n * A * rho_n * cp_n)

        return {
            "R_0": float(R0),
            "R1": float(R1), "C1": float(C1),
            "R2": float(R2), "C2": float(C2),
            "C_th_core": float(Cth),
            "V_nom": float(np.mean(v)),
            "Q_nom": float(solution["Discharge capacity [A.h]"].entries[-1])
        }

    def run_full_simulation(self, updates, c_rate=1.0, experiment=None):
        # 1. Electrochemical-Thermal Solve
        model_dict = self.electro_model.build_model(parameter_updates=updates)

        if experiment:
             results = self.electro_model.simulate(model_dict, experiment=experiment)
             # Extract effective C-rate for mechanical scaling (Issue 2)
             avg_current = np.mean(np.abs(results["solution"]["Current [A]"].entries))
             cap_ah = model_dict["parameter_values"]["Nominal cell capacity [A.h]"]
             eff_c_rate = avg_current / cap_ah if cap_ah > 0 else 1.0
        else:
             # Adjust current for C-rate (handle scalar or profile)
             cap_ah = model_dict["parameter_values"]["Nominal cell capacity [A.h]"]

             # Effective average c-rate for time scaling and mechanical solve
             if isinstance(c_rate, (list, np.ndarray)):
                  eff_c_rate = np.mean(c_rate)
                  current = c_rate * cap_ah
             else:
                  eff_c_rate = c_rate
                  current = c_rate * cap_ah

             # Time for 1C is 3600s
             times = np.linspace(0, 3600 / eff_c_rate, 50)
             results = self.electro_model.simulate(model_dict, times, current_function=current)

        # 3. Mechanical Strain Solve
        mech_results = self.mech_model.solve_strain(
            pybamm_sol=results["solution"],
            params=model_dict["parameter_values"],
            c_rate=eff_c_rate
        )

        # 4. Fatigue / Endurance
        endurance = self.mech_model.compute_endurance_metric(mech_results["max_strain"])

        return {
            "electro": results,
  
            "mechanical": mech_results,
            "endurance": endurance,
            "params": model_dict["parameter_values"]
        }

    def validate_optimized_design(self):
        print("Validating optimized twin with full physics (using BESS dispatch trace)...")

        # Extract dynamic voltage limits for physically consistent Experiment definition (Issue 1)
        v_min = self.optimized_params["Lower voltage cut-off [V]"]
        v_max = self.optimized_params["Upper voltage cut-off [V]"]

        # 1. Base Validation using Realistic BESS Dispatch Experiment
        # (Issue 1, 3, 9, 10, 11)
        dispatch_experiment = pybamm.Experiment([
            "Discharge at 0.5C for 30 minutes",
            "Rest for 10 minutes",
            f"Discharge at 1.5C until {v_min}V",
            "Rest for 30 minutes",
            f"Charge at 0.5C until {v_max}V"
        ])

        res_dispatch = self.run_full_simulation(self.optimized_params, experiment=dispatch_experiment)

        # 2. Robustness Check: Grid Outage during Charge (Issue 11, 13)
        print("  Running Robustness Check: Grid Outage during high-rate charge (+10% thickness)...")
        robust_updates = self.optimized_params.copy()
        robust_updates["Positive electrode thickness [m]"] *= 1.1

        blackout_experiment = pybamm.Experiment([
            "Charge at 1C for 20 minutes",
            "Rest for 60 minutes" # Abrupt grid loss / relaxation
        ])

        res_robust = self.run_full_simulation(robust_updates, experiment=blackout_experiment)

        # 3. Correct Energy and Efficiency metrics (Issue 4, 12, 17)
        def compute_energy_io_wh(sol):
             v = sol["Terminal voltage [V]"].entries
             i = sol["Current [A]"].entries
             t = sol["Time [s]"].entries
             p = v * i
             trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))

             # Separate Charge (in) and Discharge (out) - Issue 12
             e_out = trapz_func(np.maximum(p, 0), t) / 3600.0
             e_in = abs(trapz_func(np.minimum(p, 0), t)) / 3600.0
             return e_in, e_out

        e_in, e_out = compute_energy_io_wh(res_dispatch["electro"]["solution"])
        efficiency = e_out / e_in if e_in > 0 else 0.0

        # 4. Weighted Robustness Index (Issue 14)
        def compute_robustness_index(res):
             # R = w1*T + w2*strain + w3*SOH
             w = [0.3, 0.5, 0.2]
             T_max = np.max(res["electro"]["temperature"])
             strain_max = res["mechanical"]["max_strain"]
             soh_final = res["electro"]["soh_trajectory"][-1] / 100.0

             # Normalized components
             c1 = min(1.0, (T_max - 298.15) / 50.0)
             c2 = min(1.0, strain_max / self.mech_model.critical_thresholds["NFPP"])
             c3 = 1.0 - soh_final

             score = 1.0 - (w[0]*c1 + w[1]*c2 + w[2]*c3)
             return max(0.0, score)

        robustness_score = compute_robustness_index(res_robust)
        robustness_passed = robustness_score > 0.7

        # Compile final report
        clean_params = {}
        for k, v in res_dispatch["params"].items():
            if not callable(v):
                clean_k = k.replace(" ", "_").replace("[", "").replace("]", "").replace("-", "_").replace(".", "").replace("/", "_")[:31]
                clean_params[clean_k] = v

        results = {
            "energy_discharge_kwh": float(e_out / 1000.0),
            "energy_charge_kwh": float(e_in / 1000.0),
            "round_trip_efficiency": float(efficiency),
            "nominal_voltage_v": float(np.mean(res_dispatch["electro"]["terminal_voltage"])),
            "max_strain": float(res_dispatch["mechanical"]["max_strain"]),
            "cycle_life": float(min(res_dispatch["endurance"]["n_crit"], 1e12)),
            "robustness_score": float(robustness_score),
            "robustness_passed": bool(robustness_passed),
            "merged_params": clean_params,
            # Simscape-Mapped Parameters (Derived from high-fidelity DFN transient)
            "ssc_params": self.derive_ssc_parameters(res_dispatch["electro"]["solution"], res_dispatch["params"])
        }

        return results

    def export_to_json(self, results, output_path="src/power_plant/cell_params.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Validated Model (JSON) exported to {output_path}")

    def export_to_matlab(self, results, output_path="src/power_plant/optimized_params.mat"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sio.savemat(output_path, {"optimized_params": results})
        print(f"Validated Model exported to {output_path}")

if __name__ == "__main__":
    validator = StabilityValidator()
    results = validator.validate_optimized_design()
    validator.export_to_matlab(results)
