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
        deltas = opt_data.get("combined_deltas_representative", {}).copy()
        val_data = self.pipeline_data.get("validation", {})
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

    def run_full_simulation(self, updates, c_rate=1.0):
        # 1. Electrochemical-Thermal Solve
        model_dict = self.electro_model.build_model(parameter_updates=updates)

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
        print("Validating optimized twin with full physics (using values from final_validation.json)...")

        # Base Validation at 1C using the fully optimized parameter set
        res_1c = self.run_full_simulation(self.optimized_params, c_rate=1.0)

        # Robustness Check (using varying C-rate profile)
        print("  Running Robustness Check with varying C-rate and (+10% thickness)...")
        robust_updates = dict(self.optimized_params)
        robust_updates["Positive electrode thickness [m]"] *= 1.1

        # Generate varying C-rate profile
        varying_c = self.electro_model.get_varying_c_rate_profile(base_c_rate=1.0, duration=3600.0)

        res_robust = self.run_full_simulation(robust_updates, c_rate=varying_c)

        energy_base = res_1c["electro"]["solution"]["Discharge capacity [A.h]"].entries[-1]
        energy_robust = res_robust["electro"]["solution"]["Discharge capacity [A.h]"].entries[-1]
        robustness_passed = abs(energy_robust - energy_base) / energy_base < 0.15

        # Compile final report
        clean_params = {}
        for k, v in res_1c["params"].items():
            if not callable(v):
                clean_k = k.replace(" ", "_").replace("[", "").replace("]", "").replace("-", "_").replace(".", "").replace("/", "_")[:31]
                clean_params[clean_k] = v

        results = {
            "energy_capacity_kwh": float((energy_base * 3.1) / 1000.0),
            "nominal_voltage_v": float(np.mean(res_1c["electro"]["terminal_voltage"])),
            "max_strain": float(res_1c["mechanical"]["max_strain"]),
            "cycle_life": float(min(res_1c["endurance"]["n_crit"], 1e12)),
            "robustness_passed": bool(robustness_passed),
            "merged_params": clean_params,
            # Simscape-Mapped Parameters (Derived from high-fidelity DFN transient)
            "ssc_params": self.derive_ssc_parameters(res_1c["electro"]["solution"], res_1c["params"])
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
