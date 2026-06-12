import numpy as np
import pybamm
import logging
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

# 4.1 PyBaMM parameter vector space theta
THETA_SPACE = [
    "Positive electrode OCP [V]",
    "Positive particle diffusivity [m2.s-1]",
    "Positive electrode conductivity [S.m-1]",
    "Electrolyte conductivity [S.m-1]",
    "SEI resistivity [Ohm.m]",
    "Positive electrode exchange-current density [A.m-2]"
]

# Design space (geometry, porosity, etc.)
DESIGN_SPACE = [
    "Positive electrode thickness [m]",
    "Negative electrode thickness [m]",
    "Positive electrode porosity",
    "Negative electrode porosity",
    "Positive particle radius [m]",
    "carbon_fraction"
]

def carbon_percolation_conductivity(fraction: float, base_cond: float = 100.0) -> float:
    phi_c = 0.03
    if fraction <= phi_c:
        return 1e-6
    return base_cond * (fraction - phi_c)**1.8

class ParamTransform:
    def __init__(self, base_values: pybamm.ParameterValues):
        self.values_dict = dict(base_values)

    def _apply_scaling(self, key: str, factor: float):
        original = self.values_dict.get(key)
        if original is None: return
        if callable(original):
            def scaled_func(*args, f=factor, orig=original, **kwargs):
                return orig(*args, **kwargs) * f
            self.values_dict[key] = scaled_func
        else:
            self.values_dict[key] *= factor

    def apply_physics_deltas(self, deltas: Dict[str, Any]):
        """Maps physics deltas from chem_regularization to PyBaMM parameters."""
        if "thermodynamic" in deltas:
            d = deltas["thermodynamic"]
            if "voltage_boost" in d:
                ocp = self.values_dict.get("Positive electrode OCP [V]")
                if callable(ocp):
                    def shifted_ocp(sto, b=d["voltage_boost"], f=ocp):
                        return f(sto) + b
                    self.values_dict["Positive electrode OCP [V]"] = shifted_ocp
                else:
                    self.values_dict["Positive electrode OCP [V]"] += d["voltage_boost"]

        if "transport" in deltas:
            d = deltas["transport"]
            if "diffusivity_log_delta" in d:
                self._apply_scaling("Positive particle diffusivity [m2.s-1]", np.exp(d["diffusivity_log_delta"]))
            if "conductivity_log_delta" in d:
                self._apply_scaling("Positive electrode conductivity [S.m-1]", np.exp(d["conductivity_log_delta"]))
            if "electrolyte_conductivity_log_delta" in d:
                self._apply_scaling("Electrolyte conductivity [S.m-1]", np.exp(d["electrolyte_conductivity_log_delta"]))

        if "kinetic" in deltas:
            d = deltas["kinetic"]
            if "exchange_current_log_delta" in d:
                self._apply_scaling("Positive electrode exchange-current density [A.m-2]", np.exp(d["exchange_current_log_delta"]))

    def apply_design_vector(self, x: np.ndarray, names: List[str]):
        for val, name in zip(x, names):
            if name == "carbon_fraction":
                self.values_dict["Positive electrode conductivity [S.m-1]"] = carbon_percolation_conductivity(val)
            else:
                self.values_dict[name] = val

    def get_parameter_values(self) -> pybamm.ParameterValues:
        return pybamm.ParameterValues(self.values_dict)

# 4.3 Optimization metric
def pybamm_loss(metrics: Dict[str, float]) -> float:
    """Multi-objective loss: Maximize energy and capacity while satisfying constraints."""
    if not metrics.get("success", False):
        return 1e6
    # Weighted sum: 1.0 * Energy (Wh) + 2.0 * Avg Voltage (V)
    return -(1.0 * metrics["energy"] + 2.0 * metrics["avg_voltage"])

class OptimizerEngine:
    def __init__(self, base_params: pybamm.ParameterValues):
        self.base_params = base_params
        try:
            self.model = pybamm.sodium_ion.SPM()
        except AttributeError:
            self.model = pybamm.lithium_ion.SPM()

    def simulate(self, params: pybamm.ParameterValues) -> Dict[str, float]:
        try:
            inputs = {"Current [A]": params["Nominal cell capacity [A.h]"]}
            sim = pybamm.Simulation(self.model, parameter_values=params)
            sol = sim.solve([0, 3600], inputs=inputs)

            V = sol["Terminal voltage [V]"].data
            I = sol["Current [A]"].data
            t = sol["Time [s]"].data

            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy = trapz_func(V * I, t) / 3600
            capacity = trapz_func(I, t) / 3600
            avg_v = np.mean(V)

            # Internal Resistance Estimate
            v_ocp = V[0]
            v_init = V[1] if len(V) > 1 else V[0]
            i_val = abs(I[0]) if abs(I[0]) > 1e-3 else 1.0
            r_int = abs(v_ocp - v_init) / i_val

            return {
                "energy": float(energy),
                "capacity": float(capacity),
                "avg_voltage": float(avg_v),
                "internal_resistance": float(r_int),
                "success": True
            }
        except Exception:
            return {"energy": 0, "capacity": 0, "success": False}

    # 4.4 Optimizer
    def optimize(self, material_deltas: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        bounds = [
            (30e-6, 150e-6), (30e-6, 150e-6),
            (0.2, 0.5), (0.2, 0.5),
            (1e-7, 10e-6),
            (0.02, 0.15)
        ]

        def objective(x):
            pt = ParamTransform(self.base_params)
            pt.apply_physics_deltas(material_deltas)
            pt.apply_design_vector(x, DESIGN_SPACE)
            res = self.simulate(pt.get_parameter_values())
            return pybamm_loss(res)

        x0 = np.array([0.5 * (b[0] + b[1]) for b in bounds])
        res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 5})

        final_pt = ParamTransform(self.base_params)
        final_pt.apply_physics_deltas(material_deltas)
        final_pt.apply_design_vector(res.x, DESIGN_SPACE)
        metrics = self.simulate(final_pt.get_parameter_values())
        return res.x, metrics

def run_workflow():
    from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCategory
    from src.cell_optimization.chem_regularization import derive_coupled_deltas, regularize_salt_props, regularize_functionalization

    engine = MaterialMappingEngine()
    db, bases = engine.run()
    if not bases: return

    opt = OptimizerEngine(pybamm.ParameterValues(engine.base_params))

    cat = db[MaterialCategory.CATHODE_DOPANT][0] if db[MaterialCategory.CATHODE_DOPANT] else None
    salt = db[MaterialCategory.SALT][0] if db[MaterialCategory.SALT] else None
    func = db[MaterialCategory.FUNCTIONALIZATION][0] if db[MaterialCategory.FUNCTIONALIZATION] else None

    deltas = {}
    if cat:
        d = derive_coupled_deltas(bases["cathode"]["properties"], cat.properties,
                                 bases["cathode"]["formula"], cat.composition)
        for k, v in d.items(): deltas.setdefault(k, {}).update(v)
    if salt:
        d = regularize_salt_props(bases["salt"]["solution"], salt.properties)
        for k, v in d.items(): deltas.setdefault(k, {}).update(v)
    if func:
        d = regularize_functionalization(func.properties)
        for k, v in d.items(): deltas.setdefault(k, {}).update(v)

    x_opt, metrics = opt.optimize(deltas)

    output = {
        "materials": {
            "cathode": {
                "name": cat.name if cat else "Base",
                "formula": cat.composition if cat else bases["cathode"]["formula"]
            },
            "electrolyte": {
                "salt": salt.name if salt else "Base",
                "functionalization": func.name if func else "None"
            }
        },
        "cell_parameters": {
            "voltage": round(metrics.get("avg_voltage", 0), 3),
            "energy_density": round(metrics.get("energy", 0) * 15, 2),
            "internal_resistance": round(metrics.get("internal_resistance", 0), 4)
        },
        "design_specs": dict(zip(DESIGN_SPACE, x_opt.tolist())),
        "combined_deltas": deltas
    }

    print("\nFINAL SYSTEM OUTPUT:")
    print(json.dumps(output, indent=2))
    return output

if __name__ == "__main__":
    run_workflow()
