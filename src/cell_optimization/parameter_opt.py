import numpy as np
import pybamm
import logging
import math
from typing import Dict, List, Any, Optional
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCandidate

logging.basicConfig(level=logging.INFO)

class ParamTransform:
    """Pure parameter wrapper to prevent dictionary mutation leakage."""
    def __init__(self, base_values):
        self.base = base_values
        self.multiplier_map = {}
        self.additive_map = {}

    def add_multiplier(self, name, val):
        self.multiplier_map[name] = self.multiplier_map.get(name, 1.0) * val

    def add_additive(self, name, val):
        self.additive_map[name] = self.additive_map.get(name, 0.0) + val

    def evaluate(self):
        params = pybamm.ParameterValues(self.base)
        for name, m in self.multiplier_map.items():
            base = params[name]
            if callable(base):
                params[name] = (lambda *args, b=base, mult=m, **kwargs: b(*args, **kwargs) * mult)
            else:
                params[name] = base * m
        for name, a in self.additive_map.items():
            base = params[name]
            if callable(base):
                params[name] = (lambda *args, b=base, add=a, **kwargs: b(*args, **kwargs) + add)
            else:
                params[name] = base + a
        return params

def get_y(params: pybamm.ParameterValues, horizon=1800) -> np.ndarray:
    """Runs PyBaMM simulation and extracts performance metrics."""
    # Ensure physical consistency for SPM
    c_max_n = params["Maximum concentration in negative electrode [mol.m-3]"]
    c_max_p = params["Maximum concentration in positive electrode [mol.m-3]"]
    params["Initial concentration in negative electrode [mol.m-3]"] = 0.1 * c_max_n
    params["Initial concentration in positive electrode [mol.m-3]"] = 0.9 * c_max_p

    model = pybamm.lithium_ion.SPM()
    solver = pybamm.CasadiSolver(mode="safe")
    sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
    try:
        sl = sim.solve([0, horizon], inputs={"Current [A]": 1.0}) # Lower current for stability
        v_final = float(sl["Terminal voltage [V]"].entries[-1])
        t_max = float(np.max(sl["Cell temperature [K]"].entries))

        # Energy = int(V * I dt)
        v_entries = sl["Terminal voltage [V]"].entries
        i_entries = sl["Current [A]"].entries
        t_entries = sl.t

        # Use trapezoidal integration (fallback for NumPy < 1.25.0)
        trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
        energy = np.abs(trapezoid(v_entries * i_entries, t_entries)) / 3600.0 # Wh

        # Capacity (Ah)
        capacity = np.abs(trapezoid(i_entries, t_entries)) / 3600.0

        return np.array([v_final, energy, capacity, t_max])
    except Exception as e:
        logging.warning(f"Simulation failed: {e}")
        return np.array([0.0, 0.0, 0.0, 400.0])

def compute_sensitivity(theta: np.ndarray, materials: List[MaterialCandidate], structural_keys: List[str]) -> np.ndarray:
    """
    Computes the parameter Jacobian S_{ij} = dy_i / dtheta_j
    theta: structural parameter vector
    materials: selected materials (used to set the baseline)
    """
    base_params_vals = get_parameter_values()

    def get_params(th):
        transform = ParamTransform(base_params_vals)
        # Apply structural params
        for i, key in enumerate(structural_keys):
            transform.base[key] = th[i]
        # Apply material deltas
        for m in materials:
            deltas = m.to_pybamm_delta()
            for name, (mode, val) in deltas.items():
                if mode == "multiplier":
                    transform.add_multiplier(name, val)
                else:
                    transform.add_additive(name, val)
        return transform.evaluate()

    y_base = get_y(get_params(theta))
    n_y = len(y_base)
    n_theta = len(theta)
    S = np.zeros((n_y, n_theta))
    eps = 1e-4

    for j in range(n_theta):
        th_plus = theta.copy()
        th_plus[j] += eps
        y_plus = get_y(get_params(th_plus))
        S[:, j] = (y_plus - y_base) / eps

    return S

def pybamm_loss(y: np.ndarray, target_y: np.ndarray) -> float:
    """
    Optimization metric.
    y = [voltage, energy, capacity, t_max]
    """
    # Maximize energy, maximize capacity, minimize t_max
    # target_y used for normalization or reference
    weights = np.array([-1.0, -10.0, -10.0, 0.1]) # Negative for maximization
    return float(np.dot(y, weights))

def optimize(materials_db: Dict[str, List[MaterialCandidate]]):
    """
    Main optimization loop.
    Iterates over material combinations and optimizes structural parameters.
    """
    structural_keys = [
        "Positive electrode thickness [m]",
        "Negative electrode thickness [m]",
        "Positive electrode porosity",
        "Negative electrode porosity"
    ]
    # Initial guess
    theta = np.array([1.2e-4, 1.2e-4, 0.3, 0.3])

    best_overall_loss = float('inf')
    best_config = {}

    # Simple greedy search over material combinations
    cathodes = materials_db.get("Cathode_Dopant", [])
    if not cathodes: cathodes = [None]
    salts = materials_db.get("Salt", [])
    if not salts: salts = [None]
    funcs = materials_db.get("Functionalization", [])
    if not funcs: funcs = [None]

    for cathode in cathodes:
        for salt in salts:
            for func in funcs:
                selected_materials = [m for m in [cathode, salt, func] if m is not None]

                # Optimize structural parameters for this material set
                curr_theta = theta.copy()
                for i in range(2): # Reduced iterations for speed
                    S = compute_sensitivity(curr_theta, selected_materials, structural_keys)

                    def get_params_local(th):
                        transform = ParamTransform(get_parameter_values())
                        for idx, key in enumerate(structural_keys):
                            transform.base[key] = th[idx]
                        for m in selected_materials:
                            deltas = m.to_pybamm_delta()
                            for name, (mode, val) in deltas.items():
                                if mode == "multiplier": transform.add_multiplier(name, val)
                                else: transform.add_additive(name, val)
                        return transform.evaluate()

                    y = get_y(get_params_local(curr_theta))

                    # Gradient of loss wrt theta: dL/dtheta = dL/dy * dy/dtheta = weights * S
                    weights = np.array([-1.0, -10.0, -10.0, 0.1])
                    grad = weights @ S
                    curr_theta -= 0.05 * grad * curr_theta # Relative step

                    # Physical constraints
                    curr_theta[0:2] = np.clip(curr_theta[0:2], 5e-5, 3e-4)
                    curr_theta[2:4] = np.clip(curr_theta[2:4], 0.2, 0.7)

                final_params = get_params_local(curr_theta)
                final_y = get_y(final_params)
                loss = pybamm_loss(final_y, None)

                if loss < best_overall_loss:
                    best_overall_loss = loss
                    best_config = {
                        "materials": {
                            "cathode": cathode.name if cathode else "Base",
                            "electrolyte": f"{salt.name if salt else 'Base'} + {func.name if func else 'None'}"
                        },
                        "cell_parameters": {
                            "voltage": float(final_y[0]),
                            "energy_density": float(final_y[1]), # Wh (approx)
                            "internal_resistance": 0.1 # Placeholder
                        },
                        "structural": dict(zip(structural_keys, curr_theta.tolist()))
                    }

    return best_config

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    materials = engine.run()
    result = optimize(materials)
    import json
    print(json.dumps(result, indent=2))
