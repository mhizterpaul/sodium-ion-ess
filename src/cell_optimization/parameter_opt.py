import numpy as np
import pybamm
import logging
import math
import json
from typing import Dict, List, Any, Optional, Callable
from scipy.optimize import minimize
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCandidate

logging.basicConfig(level=logging.INFO)

class ParamTransform:
    """Pure parameter wrapper to prevent dictionary mutation leakage."""
    def __init__(self, base_values):
        self.base = base_values.copy()
        self.multiplier_map = {}
        self.additive_map = {}

    def add_multiplier(self, name, val):
        self.multiplier_map[name] = self.multiplier_map.get(name, 1.0) * val

    def add_additive(self, name, val):
        self.additive_map[name] = self.additive_map.get(name, 0.0) + val

    def evaluate(self):
        params = pybamm.ParameterValues(self.base)
        for name, m in self.multiplier_map.items():
            if name in params:
                base = params[name]
                if callable(base):
                    params[name] = (lambda *args, b=base, mult=m, **kwargs: b(*args, **kwargs) * mult)
                else:
                    params[name] = base * m
        for name, a in self.additive_map.items():
            if name in params:
                base = params[name]
                if callable(base):
                    params[name] = (lambda *args, b=base, add=a, **kwargs: b(*args, **kwargs) + add)
                else:
                    params[name] = base + a
        return params

def get_y(theta: np.ndarray, materials: List[MaterialCandidate], design_keys: List[str]) -> np.ndarray:
    """Runs PyBaMM simulation and extracts performance metrics."""
    base_params = get_parameter_values()
    transform = ParamTransform(base_params)

    # 1. Apply Design Parameters (θ)
    for i, key in enumerate(design_keys):
        if key in transform.base:
            transform.base[key] = theta[i]

    # 2. Conductive Carbon Percolation Model
    if "Positive electrode conductive carbon fraction" in design_keys:
        carbon_idx = design_keys.index("Positive electrode conductive carbon fraction")
        phi = theta[carbon_idx]
        phi_c = 0.03 # percolation threshold
        cond_mult = max(((phi - phi_c) / (1 - phi_c + 1e-9)), 0.01)**1.8
        transform.add_multiplier("Positive electrode conductivity [S.m-1]", cond_mult / (0.08 / (1-phi_c))**1.8) # Normalized to 0.08 baseline

    # 3. Apply Material Deltas (m)
    for m in materials:
        deltas = m.to_pybamm_delta()
        for name, (mode, val) in deltas.items():
            if mode == "multiplier": transform.add_multiplier(name, val)
            else: transform.add_additive(name, val)

    params = transform.evaluate()
    model = pybamm.lithium_ion.SPM({"thermal": "isothermal"})
    solver = pybamm.CasadiSolver(mode="safe")

    if "Cell volume [m3]" not in params:
        params["Cell volume [m3]"] = 0.130 * 0.070 * 0.0003

    sim = pybamm.Simulation(model, parameter_values=params, solver=solver)

    try:
        sl = sim.solve([0, 3600], inputs={"Current [A]": 1.0})
        v_entries = sl["Terminal voltage [V]"].entries
        i_entries = sl["Current [A]"].entries
        t_entries = sl.t
        trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
        energy_wh = np.abs(trapezoid(v_entries * i_entries, t_entries)) / 3600.0
        capacity_ah = np.abs(trapezoid(i_entries, t_entries)) / 3600.0
        power_w = np.mean(v_entries * i_entries)
        eff = float(np.mean(sl["X-averaged negative electrode transport efficiency"].entries))
        j_max = float(np.max(np.abs(sl["X-averaged negative electrode interfacial current density [A.m-2]"].entries)))

        return np.array([energy_wh, power_w, capacity_ah, eff, -j_max])
    except Exception:
        return np.array([0.0, 0.0, 0.0, 0.0, -1e6])

# --- OBJECTIVE WRAPPERS ---
def energy_obj(theta, materials, keys): return -get_y(theta, materials, keys)[0]
def power_obj(theta, materials, keys): return -get_y(theta, materials, keys)[1]
def capacity_obj(theta, materials, keys): return -get_y(theta, materials, keys)[2]
def efficiency_obj(theta, materials, keys): return -get_y(theta, materials, keys)[3]
def stability_obj(theta, materials, keys): return -get_y(theta, materials, keys)[4]

def optimize_objective(theta_init, materials, keys, obj_fn, bounds):
    """Inner loop: Optimize continuous design θ for a single objective function."""
    res = minimize(obj_fn, theta_init, args=(materials, keys), method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})
    return res.x, -res.fun

def compute_envelope(materials, design_keys, theta_init, bounds):
    envelope = {}
    envelope["energy"] = optimize_objective(theta_init, materials, design_keys, energy_obj, bounds)[1]
    envelope["power"] = optimize_objective(theta_init, materials, design_keys, power_obj, bounds)[1]
    envelope["capacity"] = optimize_objective(theta_init, materials, design_keys, capacity_obj, bounds)[1]
    envelope["efficiency"] = optimize_objective(theta_init, materials, design_keys, efficiency_obj, bounds)[1]
    envelope["stability"] = optimize_objective(theta_init, materials, design_keys, stability_obj, bounds)[1]
    return envelope

def optimize(materials_db: Dict[str, List[MaterialCandidate]]):
    """
    Two-level optimization: Ranks material envelopes, then solves design θ.
    MTMS is excluded from ranking as it is a fixed functionalization step.
    """
    design_keys = [
        "Positive electrode thickness [m]", "Negative electrode thickness [m]",
        "Positive electrode porosity", "Negative electrode porosity", "Separator porosity",
        "Positive electrode Bruggeman coefficient (electrolyte)", "Negative electrode Bruggeman coefficient (electrolyte)",
        "Positive electrode active material volume fraction", "Positive particle radius [m]", "Negative particle radius [m]",
        "Typical electrolyte concentration [mol.m-3]", "Positive electrode conductive carbon fraction"
    ]
    theta_init = np.array([1e-4, 1.2e-4, 0.3, 0.3, 0.5, 1.5, 1.5, 0.65, 1e-6, 5e-6, 1000.0, 0.08])
    bounds = [(5e-5, 3e-4)]*2 + [(0.2, 0.7)]*3 + [(1.0, 4.0)]*2 + [(0.4, 0.9)] + [(1e-7, 1e-5)]*2 + [(500.0, 2000.0), (0.01, 0.2)]

    best_score = -float('inf')
    best_config = {}

    cathodes = materials_db.get("Cathode_Dopant", []) or [None]
    salts = materials_db.get("Salt", []) or [None]
    # Functionalization (MTMS) removed from search space in optimizer

    refs = np.array([1.3, 2.5, 0.5, 0.9, 20.0])
    weights = np.array([1.0, 0.5, 1.0, 0.5, 0.2])

    for cathode in cathodes:
        for salt in salts:
            m_set = [m for m in [cathode, salt] if m is not None]
            print(f"Evaluating Material System: {[m.name for m in m_set]}")
            env = compute_envelope(m_set, design_keys, theta_init, bounds)

            # Ranking score (penalized by proxy uncertainty)
            perf_vector = np.array([env["energy"], env["power"], env["capacity"], env["efficiency"], env["stability"]])
            norm_perf = perf_vector / refs
            u_proxy = sum(m.proxy_uncertainty for m in m_set)

            score = np.dot(norm_perf, weights) - 2.0 * u_proxy # Uncertainty penalty

            if score > best_score:
                best_score = score
                def composite_obj(th, ms, ks): return -np.dot(get_y(th, ms, ks), np.array([1.0, 0.5, 1.0, 0.5, 0.001]))
                theta_opt, _ = optimize_objective(theta_init, m_set, design_keys, composite_obj, bounds)

                best_config = {
                    "materials": {
                        "cathode": cathode.name if cathode else "Base",
                        "electrolyte": f"{salt.name if salt else 'Base'}"
                    },
                    "cell_parameters": dict(zip(design_keys, theta_opt.tolist()))
                }

    return best_config

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    materials = engine.run()
    result = optimize(materials)
    print(json.dumps(result, indent=2))
