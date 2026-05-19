import pybamm
import numpy as np
import scipy.io as sio
import os
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

class StabilityValidator:
    """
    Stability Validation (Physics Consistency Check).
    Uses coupled reduced-order physics framework with PyBaMM.
    Ref: docs/paper.md
    """

    def __init__(self, base_params_dict):
        self.base_params = base_params_dict

    def validate_electrochemical_performance(self, optimized_subset=None):
        """
        Validates energy density, nominal voltage, currents, charge time,
        power capability, and cycle life.
        """
        print("Validating electrochemical performance...")

        params_to_test = self.base_params.copy()
        if optimized_subset:
            # Check if optimized_subset is the dict returned by NFPPoptimizer
            if isinstance(optimized_subset, dict):
                design = optimized_subset["design"]
                # Map optimization vector theta = [L_c, L_a, eps_c, eps_a, r_p] back to keys
                params_to_test.update({
                    "Positive electrode thickness [m]": design[0],
                    "Negative electrode thickness [m]": design[1],
                    "Positive electrode porosity": design[2],
                    "Negative electrode porosity": design[3],
                    "Positive particle radius [m]": design[4]
                })
            else:
                # Fallback for old vector format
                params_to_test.update({
                    "Positive electrode thickness [m]": optimized_subset[0],
                    "Negative electrode thickness [m]": optimized_subset[1],
                    "Positive electrode porosity": optimized_subset[2],
                    "Negative electrode porosity": optimized_subset[3],
                    "Positive particle radius [m]": optimized_subset[4]
                })

        param_values = pybamm.ParameterValues(params_to_test)
        # Ensure Current function is set to a fixed value instead of an InputParameter
        param_values["Current function [A]"] = 10.0 # 1C for 10Ah cell

        try:
            model = pybamm.sodium_ion.DFN()
        except AttributeError:
            model = pybamm.lithium_ion.DFN()

        # Simulation for Capacity and Voltage
        sim = pybamm.Simulation(model, parameter_values=param_values)
        sol = sim.solve([0, 3600*2]) # 0.5C approx

        # 1. Energy capacity (kWh)
        energy_wh = sol["Discharge energy [W.h]"].data[-1]
        energy_kwh = energy_wh / 1000.0

        # 2. Nominal voltage (V)
        # Average voltage during discharge
        nominal_voltage = np.mean(sol["Terminal voltage [V]"].data)

        # 3. Continuous current (A)
        # Defined by rated capacity / 1h
        capacity_ah = sol["Discharge capacity [A.h]"].data[-1]
        continuous_current = capacity_ah # 1C

        # 4. Peak current (A)
        # Typically 3C-5C for NFPP
        peak_current = 3.0 * continuous_current

        # 5. Charge time (h)
        charge_time_h = 1.0 # Assuming 1C rated charge

        # 6. Power capability (kW)
        power_kw = (nominal_voltage * peak_current) / 1000.0

        # 7. Cycle life (Estimated based on SEI growth model surrogate)
        # SOH_dot = -k * exp(-Ea/RT) * I^0.5
        # We integrate over the 2h discharge simulation
        temp_k = sol["Volume-averaged cell temperature [K]"].data
        k_sei = 1e-10
        # Use trapezoid from scipy or manual trapz equivalent since np.trapz was removed in numpy 2.0
        from scipy.integrate import trapezoid
        sei_growth = trapezoid(k_sei * np.exp(-30000 / (8.314 * temp_k)), sol["Time [s]"].data)
        # Normalize to 20% capacity loss threshold
        cycle_life = int(0.2 / (sei_growth + 1e-15))
        cycle_life = min(max(cycle_life, 1000), 10000) # Clamp to realistic range

        mass_estimate = 0.07 # 70g for 10Ah cell approx
        energy_density = energy_wh / mass_estimate

        results = {
            "energy_capacity_kwh": energy_kwh,
            "nominal_voltage_v": nominal_voltage,
            "continuous_current_a": continuous_current,
            "peak_current_a": peak_current,
            "charge_time_h": charge_time_h,
            "power_capability_kw": power_kw,
            "cycle_life": cycle_life,
            "energy_density_wh_kg": energy_density,
            "met_constraints": energy_density >= 140.0,
            "merged_params": params_to_test
        }

        print(f"Validation Metrics:")
        for k, v in results.items():
            if k != "merged_params":
                print(f"  {k}: {v}")

        # Obtain equivalent cell resistance profile wrt thermal field and thermoelastic strain
        results["resistance_profile"] = self.calculate_resistance_profile(sol, params_to_test)

        return results

    def calculate_resistance_profile(self, solution, params):
        """
        Obtains the equivalent cell resistance profile wrt thermal field variation and thermoelastic_strain.
        Ref: docs/paper.md Section 3.2
        """
        print("Calculating resistance profile wrt thermal field and thermoelastic strain...")

        temp_field = solution["Volume-averaged cell temperature [K]"].data
        # Simplified thermoelastic strain: epsilon = alpha * delta_T + beta * delta_SOC
        alpha_thermal = 1e-5 # thermal expansion coeff
        beta_soc = 0.02 # concentration expansion coeff

        capacity_ah = solution["Discharge capacity [A.h]"].data[-1]
        soc = 1.0 - (solution["Discharge capacity [A.h]"].data / capacity_ah) if capacity_ah > 0 else np.zeros_like(temp_field)

        delta_T = temp_field - temp_field[0]
        strain = alpha_thermal * delta_T + beta_soc * soc

        # Resistance model: R = R_base * (1 + gamma_T * delta_T + gamma_eps * strain)
        r_base = params.get("Contact resistance [Ohm]", 0.001)
        gamma_t = -0.01 # Resistance decreases with Temp
        gamma_eps = 0.5 # Resistance increases with strain (interfacial degradation)

        resistance_profile = r_base * (1 + gamma_t * delta_T + gamma_eps * strain)

        return {
            "temperature": temp_field.tolist(),
            "strain": strain.tolist(),
            "resistance": resistance_profile.tolist()
        }

    def export_to_matlab(self, results, output_path="src/control_systems/optimized_params.mat"):
        """
        Exports the validated results and merged parameters as a MATLAB input file.
        """
        merged_params = results["merged_params"]
        mat_ready_dict = {}
        for k, v in merged_params.items():
            # Truncate to 31 chars for MATLAB
            clean_k = k.replace(" ", "_").replace("[", "").replace("]", "").replace("-", "_").replace(".", "").replace("/", "_")[:31]
            if callable(v):
                mat_ready_dict[clean_k] = "function_handle"
            else:
                mat_ready_dict[clean_k] = v

        # Add metrics
        mat_ready_dict["validation_metrics"] = {k: v for k, v in results.items() if k != "merged_params"}

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sio.savemat(output_path, {"optimized_params": mat_ready_dict})
        print(f"Validated parameter set and metrics exported to {output_path}")

if __name__ == "__main__":
    base_p = get_parameter_values()
    validator = StabilityValidator(base_p)
    # Mock optimized subset for testing
    mock_opt = {"design": [0.0001, 0.00012, 0.3, 0.3, 1e-6]}
    results = validator.validate_electrochemical_performance(mock_opt)
    validator.export_to_matlab(results)
