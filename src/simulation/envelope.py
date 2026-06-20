import pybamm
import numpy as np
import scipy.io as sio
import os
import json
from src.simulation.utilities.parameters.parameter_builder import get_parameter_values
from src.simulation.utilities.electrochemical.pybamm_driver import ElectrochemicalThermalDriverModel
from src.simulation.utilities.thermal.pybamm_thermal import ThermalFieldModel
from src.simulation.utilities.mechanical.fenics_model import ThermoelasticStrainModel

class StabilityValidator:
    """
    Stability Validation (Envelope & Robustness).
    Uses full multiphysics Digital Twin (PyBaMM + FEniCSx).
    """

    def __init__(self, base_params_updates=None):
        self.base_updates = base_params_updates or {}
        self.electro_model = ElectrochemicalThermalDriverModel()
        self.thermal_model = ThermalFieldModel()
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

        # Adjust current for C-rate
        cap_ah = model_dict["parameter_values"]["Nominal cell capacity [A.h]"]
        current = c_rate * cap_ah

        # Time for 1C is 3600s
        times = np.linspace(0, 3600 / c_rate, 50)

        results = self.electro_model.simulate(model_dict, times, current_function=current)

        # 2. Thermal Field Extraction
        thermal_data = self.thermal_model.extract_thermal_data(results["solution"])

        # 3. Mechanical Strain Solve
        mech_results = self.mech_model.solve_strain(
            pybamm_solution=results["solution"],
            params=model_dict["parameter_values"]
        )

        # 4. Fatigue / Endurance
        endurance = self.mech_model.compute_endurance_metric(mech_results["max_strain"])

        return {
            "electro": results,
            "thermal": thermal_data,
            "mechanical": mech_results,
            "endurance": endurance,
            "params": model_dict["parameter_values"]
        }

    def validate_optimized_design(self, optimized_subset=None):
        print("Validating optimized twin with full physics...")

        # Enforce final_validation.json dependency
        val_path = "final_validation.json"
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Missing mandatory pipeline artifact: {val_path}")

        with open(val_path, "r") as f:
            pipeline_data = json.load(f)

        val_metrics = pipeline_data.get("validation")
        if not val_metrics:
            raise KeyError(f"Invalid validation data in {val_path}")

        design_updates = self.base_updates.copy()
        if optimized_subset:
            design = optimized_subset["design"]
            design_updates.update({
                "Positive electrode thickness [m]": design[0],
                "Negative electrode thickness [m]": design[1],
                "Positive electrode porosity": design[2],
                "Negative electrode porosity": design[3],
                "Positive particle radius [m]": design[4]
            })

        # Base Validation at 1C
        res_1c = self.run_full_simulation(design_updates, c_rate=1.0)

        # Envelope Sweep
        print("  Running Operating Envelope Sweep...")
        envelope = {}
        for cr in [0.5, 2.0]:
            sol = self.run_full_simulation(design_updates, c_rate=cr)
            key = f"C_{str(cr).replace('.', '_')}"
            envelope[key] = {
                "energy_wh": float(sol["electro"]["solution"]["Discharge capacity [A.h]"].entries[-1]) * 3.1,
                "max_temp": float(np.max(sol["electro"]["temperature"]))
            }

        # Robustness Check
        print("  Running Robustness Check (+10% thickness)...")
        robust_updates = design_updates.copy()
        robust_updates["Positive electrode thickness [m]"] *= 1.1
        res_robust = self.run_full_simulation(robust_updates, c_rate=1.0)

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
            "envelope_sweep": envelope,
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
    mock_design = [1.2e-4, 1.2e-4, 0.3, 0.3, 1e-6]
    results = validator.validate_optimized_design({"design": mock_design})
    validator.export_to_matlab(results)
