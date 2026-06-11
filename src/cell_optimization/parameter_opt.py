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
    """Runs PyBaMM simulation and extracts performance metrics for a specific theta/m configuration."""
    base_params = get_parameter_values()
    transform = ParamTransform(base_params)

    # 1. Apply Continuous Design Parameters (theta)
    for i, key in enumerate(design_keys):
        if key in transform.base:
            transform.base[key] = theta[i]

    # 2. Apply Compositional Constraints (e.g. Conductive Carbon effect)
    if "Positive electrode conductive carbon fraction" in design_keys:
        carbon_idx = design_keys.index("Positive electrode conductive carbon fraction")
        transform.add_multiplier("Positive electrode conductivity [S.m-1]", theta[carbon_idx] / 0.08)

    # 3. Apply Discrete Material Deltas (m)
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

        v_final = float(v_entries[-1])
        trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
        energy_wh = np.abs(trapezoid(v_entries * i_entries, t_entries)) / 3600.0
        capacity_ah = np.abs(trapezoid(i_entries, t_entries)) / 3600.0
        power_w = np.mean(v_entries * i_entries)
        eff = float(np.mean(sl["X-averaged negative electrode transport efficiency"].entries))
        # Degradation proxy: max interfacial current density
        j_max = float(np.max(np.abs(sl["X-averaged negative electrode interfacial current density [A.m-2]"].entries)))

        # [Energy, Power, Capacity, Efficiency, -Degradation]
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
    """
    For a given material set m, find the Pareto optimal performance values
    for each independent objective.
    """
    envelope = {}

    # 1. Energy Optimum
    _, val_e = optimize_objective(theta_init, materials, design_keys, energy_obj, bounds)
    envelope["energy"] = val_e

    # 2. Power Optimum
    _, val_p = optimize_objective(theta_init, materials, design_keys, power_obj, bounds)
    envelope["power"] = val_p

    # 3. Capacity Optimum
    _, val_c = optimize_objective(theta_init, materials, design_keys, capacity_obj, bounds)
    envelope["capacity"] = val_c

    # 4. Efficiency Optimum
    _, val_eff = optimize_objective(theta_init, materials, design_keys, efficiency_obj, bounds)
    envelope["efficiency"] = val_eff

    # 5. Stability Optimum
    _, val_s = optimize_objective(theta_init, materials, design_keys, stability_obj, bounds)
    envelope["stability"] = val_s

    return envelope

def optimize(materials_db: Dict[str, List[MaterialCandidate]]):
    """
    Outer loop: Material candidate evaluation and ranking.
    Compares the achievable performance envelopes of different material systems.
    """
    design_keys = [
        "Positive electrode thickness [m]",
        "Negative electrode thickness [m]",
        "Positive electrode porosity",
        "Negative electrode porosity",
        "Separator porosity",
        "Positive electrode Bruggeman coefficient (electrolyte)",
        "Negative electrode Bruggeman coefficient (electrolyte)",
        "Positive electrode active material volume fraction",
        "Positive particle radius [m]",
        "Negative particle radius [m]",
        "Typical electrolyte concentration [mol.m-3]",
        "Positive electrode conductive carbon fraction"
    ]
    theta_init = np.array([1e-4, 1.2e-4, 0.3, 0.3, 0.5, 1.5, 1.5, 0.65, 1e-6, 5e-6, 1000.0, 0.08])
    bounds = [
        (5e-5, 3e-4), (5e-5, 3e-4), # thickness
        (0.2, 0.7), (0.2, 0.7), (0.2, 0.7), # porosity
        (1.0, 4.0), (1.0, 4.0), # tortuosity
        (0.4, 0.9), # loading
        (1e-7, 1e-5), (1e-7, 1e-5), # radius
        (500.0, 2000.0), # concentration
        (0.01, 0.2) # carbon
    ]

    best_score = -float('inf')
    best_config = {}

    cathodes = materials_db.get("Cathode_Dopant", []) or [None]
    salts = materials_db.get("Salt", []) or [None]
    funcs = materials_db.get("Functionalization", []) or [None]

    # Reference values for normalization (rough estimates)
    # Energy: 1.3 Wh, Power: 2.5 W, Capacity: 0.5 Ah, Efficiency: 0.9, Stability: -j_max (-20 A/m2)
    refs = np.array([1.3, 2.5, 0.5, 0.9, 20.0])
    weights = np.array([1.0, 0.5, 1.0, 0.5, 0.2]) # Compose weights

    for cathode in cathodes:
        for salt in salts:
            for func in funcs:
                m_set = [m for m in [cathode, salt, func] if m is not None]

                print(f"Evaluating Material System: {[m.name for m in m_set if m]}")
                env = compute_envelope(m_set, design_keys, theta_init, bounds)

                # Ranking score based on normalized envelope vector
                # Higher is better for all: [Energy, Power, Capacity, Efficiency, -j_max]
                perf_vector = np.array([
                    env["energy"], env["power"], env["capacity"],
                    env["efficiency"], env["stability"]
                ])

                # Normalize metrics by reference values
                # env["stability"] is -j_max (e.g. -20).
                # Dividing -20 / 20 = -1.
                # Weights: [1.0, 0.5, 1.0, 0.5, 0.2]
                norm_perf = perf_vector / refs
                score = np.dot(norm_perf, weights)

                if score > best_score:
                    best_score = score
                    # For the winning material, find a balanced design point
                    # by optimizing the weighted composite once.
                    def composite_obj(th, ms, ks):
                        y = get_y(th, ms, ks)
                        # [energy, power, capacity, eff, -j_max]
                        w = np.array([1.0, 0.5, 1.0, 0.5, 0.001]) # small weight for stability
                        return -np.dot(y, w)

                    theta_opt, _ = optimize_objective(theta_init, m_set, design_keys, composite_obj, bounds)

                    best_config = {
                        "materials": {
                            "cathode": cathode.name if cathode else "Base",
                            "electrolyte": f"{salt.name if salt else 'Base'} + {func.name if func else 'None'}"
                        },
                        "cell_parameters": dict(zip(design_keys, theta_opt.tolist()))
                    }

    return best_config

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    materials = engine.run()
    result = optimize(materials)
    print(json.dumps(result, indent=2))
