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

    def simulate(self, model_dict: Dict[str, Any], times: list, current_function=None) -> Dict[str, Any]:
        if pybamm is None:
            raise ImportError("pybamm is required for simulation")

        model = model_dict["model"]
        param = model_dict["parameter_values"]

        if current_function is not None:
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
