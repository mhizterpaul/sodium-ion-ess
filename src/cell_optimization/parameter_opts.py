import numpy as np
import pybamm
import logging
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

# --- DESIGN SPACE (θ) ---
DESIGN_SPACE = [
    "Positive electrode thickness [m]",
    "Negative electrode thickness [m]",
    "Positive electrode porosity",
    "Negative electrode porosity",
    "Positive particle radius [m]",
    "Negative particle radius [m]",
    "Separator porosity",
    "carbon_fraction"
]

# --- PHYSICS MODELS ---

def carbon_percolation_conductivity(fraction: float, base_cond: float = 100.0) -> float:
    phi_c = 0.03
    if fraction <= phi_c:
        return 1e-6
    return base_cond * np.power(max((fraction - phi_c) / (1 - phi_c), 0.01), 1.8)

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
            if "initial_sodium_loss_delta" in d:
                self.values_dict["Initial concentration in negative electrode [mol.m-3]"] *= (1.0 + d["initial_sodium_loss_delta"])

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
            if "sei_growth_log_delta" in d:
                self._apply_scaling("SEI reaction exchange current density [A.m-2]", np.exp(d["sei_growth_log_delta"]))

    def apply_design_vector(self, x: np.ndarray, names: List[str]):
        for val, name in zip(x, names):
            if name == "carbon_fraction":
                self.values_dict["Positive electrode conductivity [S.m-1]"] = carbon_percolation_conductivity(val)
            else:
                self.values_dict[name] = val

    def get_parameter_values(self) -> pybamm.ParameterValues:
        if "Cell volume [m3]" not in self.values_dict:
            self.values_dict["Cell volume [m3]"] = 0.13 * 0.07 * 0.01
        if "Cell cooling surface area [m2]" not in self.values_dict:
            self.values_dict["Cell cooling surface area [m2]"] = 0.02
        if "Total heat transfer coefficient [W.m-2.K-1]" not in self.values_dict:
            self.values_dict["Total heat transfer coefficient [W.m-2.K-1]"] = 10.0
        if "SEI solvent diffusivity [m2.s-1]" not in self.values_dict:
            self.values_dict["SEI solvent diffusivity [m2.s-1]"] = 2.5e-22
        if "Bulk solvent concentration [mol.m-3]" not in self.values_dict:
            self.values_dict["Bulk solvent concentration [mol.m-3]"] = 2636.0
        return pybamm.ParameterValues(self.values_dict)

# --- PARETO OPTIMIZATION ENGINE (Layer 3) ---

def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """Checks if vector A dominates vector B (MAXIMIZATION)."""
    return np.all(obj_a >= obj_b) and np.any(obj_a > obj_b)

class ParetoOptimizer:
    def __init__(self, base_params: pybamm.ParameterValues):
        self.base_params = base_params
        self.model = pybamm.lithium_ion.SPM({
            "SEI": "solvent-diffusion limited",
            "loss of active material": "stress-driven",
            "thermal": "lumped"
        })

    def simulate(self, params: pybamm.ParameterValues) -> Dict[str, float]:
        try:
            inputs = {"Current [A]": params["Nominal cell capacity [A.h]"]}
            sim = pybamm.Simulation(self.model, parameter_values=params)
            sol = sim.solve([0, 3600], inputs=inputs)

            v = sol["Terminal voltage [V]"].data
            curr = sol["Current [A]"].data
            p = v * curr
            t = sol["Time [s]"].data

            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy = trapz_func(p, t) / 3600
            power = np.max(p)

            m_stability = 0.0
            try:
                 sei_growth = sol["X-averaged negative SEI thickness [m]"].data[-1] - sol["X-averaged negative SEI thickness [m]"].data[0]
                 m_stability = -sei_growth * 1e8
            except:
                 m_stability = -float(params["Positive particle radius [m]"]) * 1e6

            # Internal Resistance proxy
            r_int = abs(v[0] - v[1]) / (abs(curr[0]) + 1e-6) if len(v) > 1 else 0.0

            return {
                "energy": float(energy),
                "power": float(power),
                "mechanical_stability": float(m_stability),
                "avg_voltage": float(np.mean(v)),
                "internal_resistance": float(r_int),
                "success": True
            }
        except Exception:
            return {"success": False}

    def compute_jacobian(self, params: pybamm.ParameterValues) -> np.ndarray:
        """Computes G_ij = dJ_i / dp_j."""
        eps = 1e-4
        base_res = self.simulate(params)
        if not base_res["success"]: return np.zeros((3, len(DESIGN_SPACE)))

        j_base = np.array([base_res["energy"], base_res["power"], base_res["mechanical_stability"]])
        G = np.zeros((3, len(DESIGN_SPACE)))

        for j, name in enumerate(DESIGN_SPACE):
            x_base = []
            for n in DESIGN_SPACE:
                if n == "carbon_fraction": x_base.append(0.05)
                else: x_base.append(params[n])
            x_base = np.array(x_base)

            x_perturbed = x_base.copy()
            x_perturbed[j] *= (1 + eps)

            pt = ParamTransform(self.base_params)
            pt.values_dict = dict(params)
            pt.apply_design_vector(x_perturbed, DESIGN_SPACE)

            res = self.simulate(pt.get_parameter_values())
            if res["success"]:
                j_pert = np.array([res["energy"], res["power"], res["mechanical_stability"]])
                G[:, j] = (j_pert - j_base) / (np.abs(j_base) * eps + 1e-12)
        return G

    def find_pareto_set(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pareto = []
        for i, c1 in enumerate(candidates):
            dominated = False
            obj_a = np.array([c1["metrics"]["energy"], c1["metrics"]["power"], c1["metrics"]["mechanical_stability"]])
            for j, c2 in enumerate(candidates):
                if i == j: continue
                obj_b = np.array([c2["metrics"]["energy"], c2["metrics"]["power"], c2["metrics"]["mechanical_stability"]])
                if dominates(obj_b, obj_a):
                    dominated = True
                    break
            if not dominated:
                pareto.append(c1)
        return pareto

    def optimize_design(self, material_deltas: Dict[str, Any], initial_x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        bounds = [(30e-6, 150e-6), (30e-6, 150e-6), (0.2, 0.5), (0.2, 0.5), (1e-7, 10e-6), (1e-7, 10e-6), (0.3, 0.7), (0.02, 0.15)]
        def objective(x):
            pt = ParamTransform(self.base_params)
            pt.apply_physics_deltas(material_deltas)
            pt.apply_design_vector(x, DESIGN_SPACE)
            res = self.simulate(pt.get_parameter_values())
            if not res["success"]: return 1e6
            return -(1.0 * res["energy"] + 0.1 * res["power"] + 0.01 * res["mechanical_stability"])
        res = minimize(objective, initial_x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 10})
        final_pt = ParamTransform(self.base_params)
        final_pt.apply_physics_deltas(material_deltas)
        final_pt.apply_design_vector(res.x, DESIGN_SPACE)
        return res.x, self.simulate(final_pt.get_parameter_values())

def run_workflow():
    from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCategory
    from src.cell_optimization.chem_regularization import derive_coupled_deltas, regularize_salt_props, regularize_functionalization

    engine = MaterialMappingEngine()
    db, bases = engine.run()
    if not bases: return
    opt = ParetoOptimizer(pybamm.ParameterValues(engine.base_params))

    all_runs = []
    cathodes = db[MaterialCategory.CATHODE_DOPANT] or [None]
    salts = db[MaterialCategory.SALT] or [None]

    print("Executing Multi-Objective Pareto Optimization (Layer 3)...")
    for cat in cathodes[:1]:
        for salt in salts[:1]:
            deltas = {}
            if cat:
                d = derive_coupled_deltas(bases["cathode"]["properties"], cat.properties, bases["cathode"]["formula"], cat.composition)
                for k, v in d.items(): deltas.setdefault(k, {}).update(v)
            if salt:
                d = regularize_salt_props(bases["salt"]["solution"], salt.properties)
                for k, v in d.items(): deltas.setdefault(k, {}).update(v)

            x_opt, metrics = opt.optimize_design(deltas, np.array([100e-6, 100e-6, 0.3, 0.3, 1e-6, 5e-6, 0.5, 0.05]))
            if metrics.get("success"):
                all_runs.append({"cat": cat, "salt": salt, "x": x_opt, "metrics": metrics, "deltas": deltas})

    if not all_runs: return
    pareto_set = opt.find_pareto_set(all_runs)
    best = pareto_set[0]

    final_pt = ParamTransform(opt.base_params)
    final_pt.apply_physics_deltas(best["deltas"])
    final_pt.apply_design_vector(best["x"], DESIGN_SPACE)
    G = opt.compute_jacobian(final_pt.get_parameter_values())

    groups = {"Energy": [], "Power": [], "Stability": []}
    obj_names = ["Energy", "Power", "Stability"]
    for j, name in enumerate(DESIGN_SPACE):
        dominant_obj = np.argmax(np.abs(G[:, j]))
        groups[obj_names[dominant_obj]].append(name)

    # REQUIRED OUTPUT FORMAT
    output = {
        "materials": {
            "cathode": {"name": best["cat"].name if best["cat"] else "Base", "formula": best["cat"].composition if best["cat"] else "Base"},
            "electrolyte": {"salt": best["salt"].name if best["salt"] else "Base"}
        },
        "cell_parameters": {
            "voltage": round(best["metrics"]["avg_voltage"], 3),
            "energy_density_Wh": round(best["metrics"]["energy"], 3), # Proxy
            "internal_resistance": round(best["metrics"]["internal_resistance"], 4),
            "power_W": round(best["metrics"]["power"], 3),
            "mechanical_stability_proxy": round(best["metrics"]["mechanical_stability"], 3)
        },
        "parameter_grouping": groups,
        "sensitivity_matrix": G.tolist(),
        "design_specs": dict(zip(DESIGN_SPACE, best["x"].tolist())),
        "combined_deltas": best["deltas"]
    }

    print("\nFINAL OPTIMIZED OUTPUT:")
    print(json.dumps(output, indent=2))
    return output

if __name__ == "__main__":
    run_workflow()
