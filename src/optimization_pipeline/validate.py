import pybamm
import numpy as np
import scipy.io as sio
from scipy.integrate import trapezoid
import os
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

class StabilityValidator:
    """
    Stability Validation (Envelope & Robustness).
    Uses DFN models with envelope sweeps and parameter perturbations.
    """

    def __init__(self, base_params_dict):
        self.base_params = base_params_dict

    def run_single_simulation(self, params_dict, c_rate=1.0):
        param_values = pybamm.ParameterValues(params_dict)
        # Capacity ah approx 10Ah
        cap_ah = 10.0
        param_values["Current function [A]"] = c_rate * cap_ah

        # Ensure we use Sodium-Ion DFN (No silent fallback to Lithium-ion)
        try:
            model = pybamm.sodium_ion.DFN()
        except AttributeError:
            raise ImportError("PyBaMM version does not support sodium_ion model. Deployment aborted.")

        solver = pybamm.CasadiSolver(mode="safe")
        sim = pybamm.Simulation(model, parameter_values=param_values, solver=solver)
        return sim.solve([0, 3600 / c_rate])

    def validate_electrochemical_performance(self, optimized_subset=None):
        print("Validating optimized twin (Envelope & Robustness)...")

        # 1. Base Validation
        params_to_test = self.base_params.copy()
        if optimized_subset:
            design = optimized_subset["design"]
            params_to_test.update({
                "Positive electrode thickness [m]": design[0],
                "Negative electrode thickness [m]": design[1],
                "Positive electrode porosity": design[2],
                "Negative electrode porosity": design[3],
                "Positive particle radius [m]": design[4]
            })

        sol_base = self.run_single_simulation(params_to_test, c_rate=1.0)

        # 2. Operating Envelope Sweep (0.5C to 2.0C)
        print("  Running Operating Envelope Sweep (0.5C - 2.0C)...")
        envelope_results = {}
        for cr in [0.5, 2.0]:
            sol = self.run_single_simulation(params_to_test, c_rate=cr)
            envelope_results[f"{cr}C"] = {
                "energy_wh": sol["Discharge energy [W.h]"].data[-1],
                "max_temp": np.max(sol["Volume-averaged cell temperature [K]"].data)
            }

        # 3. Robustness Check (Parameter Perturbation)
        print("  Running Robustness Check (+10% thickness perturbation)...")
        params_pert = params_to_test.copy()
        params_pert["Positive electrode thickness [m]"] *= 1.1
        sol_pert = self.run_single_simulation(params_pert, c_rate=1.0)
        delta_energy = abs(sol_pert["Discharge energy [W.h]"].data[-1] - sol_base["Discharge energy [W.h]"].data[-1])
        robustness_passed = delta_energy / sol_base["Discharge energy [W.h]"].data[-1] < 0.15

        # Refined Degradation & Resistance Profile
        temp_k = sol_base["Volume-averaged cell temperature [K]"].data
        capacity_ah = sol_base["Discharge capacity [A.h]"].data[-1]
        soc = 1.0 - (sol_base["Discharge capacity [A.h]"].data / (capacity_ah + 1e-6))

        sei_rate = 2e-10 * (10.0**0.5) * np.exp(-35000 / (8.314 * temp_k)) * (1.0 + soc)
        sei_total = trapezoid(sei_rate, sol_base["Time [s]"].data)
        cycle_life = int(0.2 / (sei_total + 1e-15))

        results = {
            "energy_capacity_kwh": sol_base["Discharge energy [W.h]"].data[-1] / 1000.0,
            "nominal_voltage_v": np.mean(sol_base["Terminal voltage [V]"].data),
            "continuous_current_a": capacity_ah,
            "peak_current_a": 3.0 * capacity_ah,
            "charge_time_h": 1.0,
            "power_capability_kw": (np.mean(sol_base["Terminal voltage [V]"].data) * 3.0 * capacity_ah) / 1000.0,
            "cycle_life": min(cycle_life, 10000),
            "energy_density_wh_kg": sol_base["Discharge energy [W.h]"].data[-1] / 0.07,
            "envelope_sweep": envelope_results,
            "robustness_passed": robustness_passed,
            "resistance_profile": {
                "temperature": temp_k.tolist(),
                "resistance": (0.01 * np.exp(5000 / (8.314 * temp_k)) + 0.02 * soc).tolist()
            },
            "merged_params": params_to_test
        }
        return results

    def export_to_matlab(self, results, output_path="src/control_systems/optimized_params.mat"):
        merged_params = results["merged_params"]
        mat_ready_dict = {}
        for k, v in merged_params.items():
            clean_k = k.replace(" ", "_").replace("[", "").replace("]", "").replace("-", "_").replace(".", "").replace("/", "_")[:31]
            if not callable(v): mat_ready_dict[clean_k] = v

        mat_ready_dict["validation_metrics"] = {k: v for k, v in results.items() if k not in ["merged_params", "resistance_profile"]}
        mat_ready_dict["resistance_data"] = results["resistance_profile"]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sio.savemat(output_path, {"optimized_params": mat_ready_dict})
        print(f"Validated Model exported to {output_path}")

if __name__ == "__main__":
    base_p = get_parameter_values()
    validator = StabilityValidator(base_p)
    results = validator.validate_electrochemical_performance({"design": [0.0001, 0.00012, 0.3, 0.3, 1e-6]})
    validator.export_to_matlab(results)
