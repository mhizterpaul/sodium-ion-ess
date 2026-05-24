import pybamm
import numpy as np
import scipy.io as sio
import os
from src.simulation.utilities.parameters.parameter_builder import get_parameter_values
from src.simulation.utilities.electrochemical.pybamm_driver import ElectrochemicalThermalDriverModel
from src.simulation.utilities.thermal.pybamm_thermal import ThermalFieldModel
from src.simulation.utilities.mechanical.fenics_model import ThermoelasticStrainModel

class OptimizationValidator:
    """
    Optimization Validator.
    Computes final cell-level performance metrics using full multiphysics Digital Twin.
    """

    def __init__(self, optimized_params=None):
        self.params_updates = optimized_params or {}
        self.electro_model = ElectrochemicalThermalDriverModel()
        self.thermal_model = ThermalFieldModel()
        self.mech_model = ThermoelasticStrainModel()

    def run_discharge_experiment(self, c_rate=1.0):
        """Runs a discharge experiment to extract performance metrics."""
        model_dict = self.electro_model.build_model(parameter_updates=self.params_updates)
        cap_ah = model_dict["parameter_values"]["Nominal cell capacity [A.h]"]
        current = c_rate * cap_ah

        # Simulation times
        times = np.linspace(0, 3600 / c_rate, 100)
        results = self.electro_model.simulate(model_dict, times, current_function=current)
        return results, model_dict["parameter_values"]

    def compute_cell_attributes(self):
        print("Computing final cell-level performance metrics...")

        # 1. Nominal Performance (1C Discharge)
        res_1c, params = self.run_discharge_experiment(c_rate=1.0)

        v_terminal = res_1c["terminal_voltage"]
        t_final = res_1c["times"][-1]
        capacity_ah = float(res_1c["solution"]["Discharge capacity [A.h]"].entries[-1])

        # Use scipy.integrate.trapezoid as np.trapz is deprecated/removed in NumPy 2.0
        from scipy.integrate import trapezoid
        energy_kwh = float(trapezoid(v_terminal, res_1c["solution"]["Discharge capacity [A.h]"].entries) / 1000.0)
        v_nom = float(np.mean(v_terminal))

        # 2. Power Capability (Peak Current - 3C burst)
        # We estimate power at 50% SOC
        soc_idx = np.argmin(np.abs(res_1c["soc_trajectory"] - 0.5))
        v_50 = v_terminal[soc_idx]
        peak_current = 3.0 * params["Nominal cell capacity [A.h]"]
        power_kw = (v_50 * peak_current) / 1000.0

        # 3. Thermal Response (under peak 3C)
        res_3c, _ = self.run_discharge_experiment(c_rate=3.0)
        max_temp = float(np.max(res_3c["temperature"]))

        # 4. Mechanical Integrity
        mech_res = self.mech_model.solve_strain(
            pybamm_solution=res_1c["solution"],
            params=params
        )
        max_strain = mech_res["max_strain"]

        # 5. Cycle Life (Fatigue + Degradation)
        # Combined SOH fade and strain-life
        endurance = self.mech_model.compute_endurance_metric(max_strain)

        # Approximate cycle life from LAM fade rate
        lam_fade = res_1c["soh_trajectory"][-1] - res_1c["soh_trajectory"][0]
        # Avoid division by zero
        n_soh = int(20.0 / (lam_fade + 1e-12)) # 20% fade limit
        cycle_life = min(n_soh, endurance["n_crit"], 10000)

        attributes = {
            "Energy_capacity_kWh": energy_kwh,
            "Nominal_voltage_V": v_nom,
            "Continuous_current_A": float(params["Nominal cell capacity [A.h]"]),
            "Peak_current_A": float(peak_current),
            "Charge_time_min": 60.0, # Target 1C charging
            "Power_capability_kW": float(power_kw),
            "Cycle_life": int(cycle_life),
            "Max_operating_temp_K": max_temp,
            "Max_structural_strain": float(max_strain)
        }

        print("Final Cell Attributes:")
        for k, v in attributes.items():
            print(f"  {k}: {v}")

        return attributes

    def export_results(self, attributes, output_path="src/bms_design/cell_attributes.mat"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sio.savemat(output_path, {"cell_attributes": attributes})
        print(f"Cell attributes exported to {output_path}")

if __name__ == "__main__":
    # Example optimized parameters from previous stages
    optimized_design = {
        "Positive electrode thickness [m]": 1.2e-4,
        "Negative electrode thickness [m]": 1.2e-4,
        "Positive electrode porosity": 0.25,
        "Negative electrode porosity": 0.25,
        "Positive particle radius [m]": 1e-6
    }

    validator = OptimizationValidator(optimized_design)
    attributes = validator.compute_cell_attributes()
    validator.export_results(attributes)
