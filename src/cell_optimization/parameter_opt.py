import numpy as np
import pybamm
import logging
import math
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from scipy.optimize import minimize
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCandidate
from src.cell_optimization.chem_regularization import regularize_material, KT

logging.basicConfig(level=logging.INFO)

class ParamTransform:
    """DSMO logic: Handles mapping of physics deltas and design parameters to PyBaMM."""
    def __init__(self, base_values):
        self.base = base_values.copy()
        self.multiplier_map = {}
        self.additive_map = {}

    def add_multiplier(self, name, val):
        self.multiplier_map[name] = self.multiplier_map.get(name, 1.0) * val

    def add_additive(self, name, val):
        self.additive_map[name] = self.additive_map.get(name, 0.0) + val

    def apply_design(self, theta: np.ndarray, design_keys: List[str]):
        """DSMO transformation: Maps continuous design vector θ to parameters."""
        for i, key in enumerate(design_keys):
            if key in self.base:
                self.base[key] = theta[i]

        # DSMO logic: Conductive Carbon Percolation
        if "Positive electrode conductive carbon fraction" in design_keys:
            idx = design_keys.index("Positive electrode conductive carbon fraction")
            phi = theta[idx]
            phi_c = 0.03
            cond_mult = max(((phi - phi_c) / (1 - phi_c + 1e-9)), 0.01)**1.8
            norm = (0.08 / (1-phi_c))**1.8
            self.add_multiplier("Positive electrode conductivity [S.m-1]", cond_mult / norm)

    def apply_physics(self, reg_data: Dict[str, Any], category: str):
        """
        DSMO transformation: Performs all delta derivation and mappings.
        Takes attenuated residuals and applies physics coupling rules.
        """
        res = reg_data["residuals"]
        dE, dG, dV, dS = res["dE"], res["dG"], res["dV"], res["dS"]
        is_network = reg_data.get("is_network", False)

        # 1. Physics Coupling Rules (Delta Derivation)
        voltage_boost = -0.01 * dE

        activation_delta = 0.2 * dV + 0.1 * dG
        network_attenuation = 0.5 if is_network else 1.0
        diffusivity_log_delta = -activation_delta * network_attenuation / (KT + 1e-9)

        reaction_rate_log_delta = 0.1 * dE - 0.3 * dG
        initial_loss_mult = math.exp(np.clip(0.2 * dS, -5, 5))
        sei_growth_mult = math.exp(np.clip(0.5 * dE - 0.2 * dS, -5, 5))
        negative_exchange_log_delta = 0.4 * dS - 0.1 * dG

        transport_log_delta = -0.5 * dE + 0.2 * dV
        interfacial_log_delta = -0.8 * dS + 0.3 * dG

        def clip_log(x): return float(np.clip(x, -5, 5))

        # 2. Mapping to PyBaMM Parameter Names
        if category == "Cathode_Dopant":
            self.add_additive("Positive electrode OCP [V]", voltage_boost)
            self.add_multiplier("Positive particle diffusivity [m2.s-1]", math.exp(clip_log(diffusivity_log_delta)))
            self.add_multiplier("Positive electrode exchange-current density [A.m-2]", math.exp(clip_log(reaction_rate_log_delta)))
        elif category == "Salt":
            self.add_multiplier("Electrolyte conductivity [S.m-1]", math.exp(clip_log(transport_log_delta)))
            self.add_multiplier("Cation transference number", 1.0 + 0.1 * float(np.tanh(transport_log_delta)))
        elif category in ["Functionalization", "Functionalization_Proxy"]:
            self.add_multiplier("SEI reaction exchange current density [A.m-2]", sei_growth_mult)
            self.add_multiplier("Initial concentration in negative electrode [mol.m-3]", initial_loss_mult)
            self.add_multiplier("SEI resistivity [Ohm.m]", math.exp(clip_log(interfacial_log_delta)))
            self.add_multiplier("Negative electrode exchange-current density [A.m-2]", math.exp(clip_log(negative_exchange_log_delta)))

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

def get_y(theta: np.ndarray, materials_reg: List[Tuple[MaterialCandidate, Dict[str, Any]]], design_keys: List[str]) -> np.ndarray:
    """Runs PyBaMM simulation for a specific design (θ) and regularized material set."""
    base_params = get_parameter_values()
    transform = ParamTransform(base_params)
    transform.apply_design(theta, design_keys)
    for cand, reg in materials_reg:
        transform.apply_physics(reg, cand.category)

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
def energy_obj(theta, materials_reg, keys): return -get_y(theta, materials_reg, keys)[0]
def power_obj(theta, materials_reg, keys): return -get_y(theta, materials_reg, keys)[1]
def capacity_obj(theta, materials_reg, keys): return -get_y(theta, materials_reg, keys)[2]
def efficiency_obj(theta, materials_reg, keys): return -get_y(theta, materials_reg, keys)[3]
def stability_obj(theta, materials_reg, keys): return -get_y(theta, materials_reg, keys)[4]

def optimize_objective(theta_init, materials_reg, keys, obj_fn, bounds):
    """Inner loop: Optimize continuous design θ for a single objective function."""
    res = minimize(obj_fn, theta_init, args=(materials_reg, keys), method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})
    return res.x, -res.fun

def compute_envelope(materials_reg, design_keys, theta_init, bounds):
    envelope = {}
    envelope["energy"] = optimize_objective(theta_init, materials_reg, design_keys, energy_obj, bounds)[1]
    envelope["power"] = optimize_objective(theta_init, materials_reg, design_keys, power_obj, bounds)[1]
    envelope["capacity"] = optimize_objective(theta_init, materials_reg, design_keys, capacity_obj, bounds)[1]
    envelope["efficiency"] = optimize_objective(theta_init, materials_reg, design_keys, efficiency_obj, bounds)[1]
    envelope["stability"] = optimize_objective(theta_init, materials_reg, design_keys, stability_obj, bounds)[1]
    return envelope

def optimize(db: Dict[str, List[MaterialCandidate]], bases: Dict[str, Any]):
    """Two-level optimization: Ranks regularized material envelopes, then solves design θ."""
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

    cathodes = db.get("Cathode_Dopant", []) or [None]
    salts = db.get("Salt", []) or [None]

    refs = np.array([1.3, 2.5, 0.5, 0.9, 20.0])
    weights = np.array([1.0, 0.5, 1.0, 0.5, 0.2])

    for cathode in cathodes:
        for salt in salts:
            candidates = [m for m in [cathode, salt] if m is not None]
            materials_reg = []
            u_total = 0.0
            for c in candidates:
                base = bases["cathode"] if c.category == "Cathode_Dopant" else bases["salt"]
                reg = regularize_material(c, base)
                materials_reg.append((c, reg))
                u_total += reg["proxy_uncertainty"]

            print(f"Evaluating Regularized Material System: {[c.name for c in candidates]}")
            env = compute_envelope(materials_reg, design_keys, theta_init, bounds)
            perf_vector = np.array([env["energy"], env["power"], env["capacity"], env["efficiency"], env["stability"]])
            norm_perf = perf_vector / refs
            score = np.dot(norm_perf, weights) - 2.0 * u_total

            if score > best_score:
                best_score = score
                def composite_obj(th, ms_reg, ks): return -np.dot(get_y(th, ms_reg, ks), np.array([1.0, 0.5, 1.0, 0.5, 0.001]))
                theta_opt, _ = optimize_objective(theta_init, materials_reg, design_keys, composite_obj, bounds)
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
    db, bases = engine.run()
    result = optimize(db, bases)
    print(json.dumps(result, indent=2))
