"""Electrochemical-Thermal Driver"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

try:
    import pybamm
except ImportError:  # pragma: no cover
    pybamm = None

from nfpp_sodium_ion.src.cell_parameters.parameter_builder import get_parameter_values

@dataclass
class ElectrochemicalThermalDriverModel:
    """DFN Electrochemical-Thermal Driver."""

    name: str = "DFN Electrochemical-Thermal Driver"
    model_type: str = "DFN"

    def build_model(self, parameter_updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if pybamm is None:
            raise ImportError("pybamm is required for the electrochemical-thermal driver model")

        options = {
            "thermal": "x-full",
            "SEI": "reaction limited",
            "SEI porosity change": "true",
            "loss of active material": "stress-driven"
        }

        try:
            model = pybamm.sodium_ion.DFN(options=options)
        except AttributeError:
            model = pybamm.lithium_ion.DFN(options=options)

        param = get_parameter_values(updates=parameter_updates)
        return {"model": model, "parameter_values": param}

    def get_varying_c_rate_profile(self, base_c_rate: float, duration: float, n_points: int = 100) -> np.ndarray:
        """Generates a varying C-rate profile (sine-wave oscillation around base)."""
        t = np.linspace(0, duration, n_points)
        # Sine oscillation between 0.5x and 1.5x of base_c_rate
        profile = base_c_rate * (1.0 + 0.5 * np.sin(2 * np.pi * t / (duration / 2)))
        return profile

    def simulate(self, model_dict: Dict[str, Any], times: list, current_function=None) -> Dict[str, Any]:
        if pybamm is None:
            raise ImportError("pybamm is required for simulation")

        model = model_dict["model"]
        param = model_dict["parameter_values"]

        if current_function is not None:
            if isinstance(current_function, (list, np.ndarray)):
                # Use interpolation for varying current profile
                # Ensure times and current_function have the same length
                t_eval = np.array(times)
                if len(current_function) != len(t_eval):
                     # If length mismatch, assume current_function is a schedule and interpolate
                     current_profile = np.interp(t_eval, np.linspace(t_eval[0], t_eval[-1], len(current_function)), current_function)
                else:
                     current_profile = current_function

                param["Current function [A]"] = pybamm.Interpolant(t_eval, current_profile, pybamm.t)
            else:
                param["Current function [A]"] = current_function

        solver = pybamm.CasadiSolver(mode="safe")
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
        solution = sim.solve(times)

        cap_ah = solution["Discharge capacity [A.h]"].entries
        nom_cap = param["Nominal cell capacity [A.h]"]
        soc = 1.0 - (cap_ah / nom_cap)

        return {
            "solution": solution,
            "times": solution["Time [s]"].entries,
            "soc_trajectory": soc,
            "soh_trajectory": solution["Loss of active material in negative electrode [%]"].entries,
            "temperature": solution["Cell temperature [K]"].entries,
            "heat_generation_rate": solution["Total heating [W.m-3]"].entries,
            "terminal_voltage": solution["Terminal voltage [V]"].entries,
        }
