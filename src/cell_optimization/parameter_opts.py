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

DESIGN_BOUNDS = [
    (30e-6, 150e-6), (30e-6, 150e-6),
    (0.2, 0.5), (0.2, 0.5),
    (1e-7, 10e-6), (1e-7, 10e-6),
    (0.3, 0.7),
    (0.02, 0.15)
]

# --- PHYSICS MODELS ---

def carbon_percolation_conductivity(fraction: float, base_cond: float = 100.0) -> float:
    # Percolation theory: sigma_eff = sigma_0 * max((phi - 0.03)/(1 - 0.03), 0.01)^1.8
    phi_c = 0.03
    if fraction <= phi_c: return 1e-6
    return base_cond * np.power(max((fraction - phi_c) / (1.0 - phi_c), 0.01), 1.8)

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
                    def shifted_ocp(sto, b=d["voltage_boost"], f=ocp): return f(sto) + b
                    self.values_dict["Positive electrode OCP [V]"] = shifted_ocp
                else:
                    self.values_dict["Positive electrode OCP [V]"] += d["voltage_boost"]
            if "initial_sodium_loss_delta" in d:
                self.values_dict["Initial concentration in negative electrode [mol.m-3]"] *= (1.0 + d["initial_sodium_loss_delta"])
            if "stability_shift" in d:
                 # Stability shift reduces degradation rates via exp(-dS) scaling
                 self._apply_scaling("SEI reaction exchange current density [A.m-2]", np.exp(-d["stability_shift"]))
                 self._apply_scaling("Positive electrode LAM constant proportional term [s-1]", np.exp(-d["stability_shift"]))

        if "transport" in deltas:
            d = deltas["transport"]
            if "diffusivity_log_delta" in d:
                self._apply_scaling("Positive particle diffusivity [m2.s-1]", np.exp(d["diffusivity_log_delta"]))
            if "conductivity_log_delta" in d:
                self._apply_scaling("Positive electrode conductivity [S.m-1]", np.exp(d["conductivity_log_delta"]))
            if "electrolyte_conductivity_log_delta" in d:
                self._apply_scaling("Electrolyte conductivity [S.m-1]", np.exp(d["electrolyte_conductivity_log_delta"]))
            if "electrolyte_diffusivity_log_delta" in d:
                self._apply_scaling("Electrolyte diffusivity [m2.s-1]", np.exp(d["electrolyte_diffusivity_log_delta"]))

        if "kinetic" in deltas:
            d = deltas["kinetic"]
            if "exchange_current_log_delta" in d:
                self._apply_scaling("Positive electrode exchange-current density [A.m-2]", np.exp(d["exchange_current_log_delta"]))
            if "sei_growth_log_delta" in d:
                self._apply_scaling("SEI reaction exchange current density [A.m-2]", np.exp(d["sei_growth_log_delta"]))
            if "sei_resistivity_log_delta" in d:
                self._apply_scaling("SEI resistivity [Ohm.m]", np.exp(d["sei_resistivity_log_delta"]))

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

# --- INDIVIDUAL OBJECTIVE OPTIMIZER ---

class HierarchicalOptimizer:
    def __init__(self, engine: Optional[Any] = None, base_params: Optional[pybamm.ParameterValues] = None):
        if engine is None:
            from src.cell_optimization.material_opt import MaterialMappingEngine
            engine = MaterialMappingEngine()
        self.engine = engine
        self.base_params = base_params or pybamm.ParameterValues(engine.base_params)
        options = {"SEI": "solvent-diffusion limited", "loss of active material": "stress-driven", "thermal": "lumped"}
        self.model = pybamm.lithium_ion.SPM(options)
        self.solver = pybamm.IDAKLUSolver()

    def simulate(self, params: pybamm.ParameterValues) -> Dict[str, Any]:
        try:
            if "Internal resistance [Ohm]" not in params:
                params["Internal resistance [Ohm]"] = 0.001
            self.solver.tol = 1e-3
            cap = float(params["Nominal cell capacity [A.h]"])
            sim = pybamm.Simulation(self.model, parameter_values=params, solver=self.solver)
            sol = sim.solve([0, 3600], inputs={"Current [A]": cap})

            v = sol["Terminal voltage [V]"].data
            curr = sol["Current [A]"].data
            p = v * curr
            t = sol["Time [s]"].data

            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy = trapz_func(p, t) / 3600
            power = np.max(p)

            stress_vars = ["Positive particle surface tangential stress [Pa]", "Negative particle surface tangential stress [Pa]"]
            max_stress = 1e-6
            for sv in stress_vars:
                 try:
                    var_data = sol[sv].data
                    max_stress = max(max_stress, np.max(np.abs(var_data)))
                 except Exception:
                    continue

            if t[-1] < 600:
                return {"success": False, "reason": "Short discharge"}

            return {
                "energy": float(energy), "power": float(power), "mechanical_stability": float(-max_stress),
                "success": True
            }
        except Exception as e:
            return {"success": False, "reason": f"{e}"}

    def compute_jacobian(self, x: np.ndarray, deltas: Dict[str, Any]) -> np.ndarray:
        eps = 1e-4
        pt = ParamTransform(self.base_params)
        pt.apply_physics_deltas(deltas)
        pt.apply_design_vector(x, DESIGN_SPACE)
        base_res = self.simulate(pt.get_parameter_values())
        if not base_res["success"]: return np.zeros((3, len(DESIGN_SPACE)))

        j_base = np.array([base_res["energy"], base_res["power"], base_res["mechanical_stability"]])
        G = np.zeros((3, len(DESIGN_SPACE)))

        for j in range(len(DESIGN_SPACE)):
            x_pert = x.copy()
            x_pert[j] *= (1 + eps)
            pt_p = ParamTransform(self.base_params)
            pt_p.apply_physics_deltas(deltas)
            pt_p.apply_design_vector(x_pert, DESIGN_SPACE)
            res = self.simulate(pt_p.get_parameter_values())
            if res["success"]:
                j_pert = np.array([res["energy"], res["power"], res["mechanical_stability"]])
                G[:, j] = (j_pert - j_base) / (np.abs(j_base) * eps + 1e-12)
        return G

    def _objective(self, x_active, x_full, active_indices, deltas, mode):
        x = x_full.copy()
        x[active_indices] = x_active
        pt = ParamTransform(self.base_params)
        pt.apply_physics_deltas(deltas)
        pt.apply_design_vector(x, DESIGN_SPACE)
        res = self.simulate(pt.get_parameter_values())
        if not res["success"]: return 1e9
        if mode == "energy": return -res["energy"]
        if mode == "power": return -res["power"]
        if mode == "stability": return -res["mechanical_stability"]
        return 1e9

    def run(self):
        return run_workflow(engine=self.engine)

def run_workflow(engine: Optional[Any] = None):
    from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCategory
    from src.cell_optimization.chem_regularization import derive_coupled_deltas, regularize_salt_props

    if engine is None:
        engine = MaterialMappingEngine()

    db, bases = engine.run()
    if not bases: return
    optimizer = HierarchicalOptimizer(engine=engine, base_params=pybamm.ParameterValues(engine.base_params))

    print("Executing Sensitivity-Driven Hierarchical Optimization (Layer 3)...")
    print("Each objective function is individually optimized using its primary drivers.")

    cathodes = db[MaterialCategory.CATHODE_DOPANT] if db[MaterialCategory.CATHODE_DOPANT] else [None]
    salts = db[MaterialCategory.SALT] if db[MaterialCategory.SALT] else [None]

    material_results = []
    for cat, salt in [(c, s) for c in cathodes[:2] for s in salts[:2]]:
        deltas = {}
        if cat:
            d = derive_coupled_deltas(bases["cathode"]["properties"], cat.properties, bases["cathode"]["formula"], cat.composition)
            for k, v in d.items(): deltas.setdefault(k, {}).update(v)
        if salt:
            d = regularize_salt_props(bases["salt"]["properties"], salt.properties)
            for k, v in d.items(): deltas.setdefault(k, {}).update(v)

        print(f"\nEvaluating system: {cat.name if cat else 'Base'} + {salt.name if salt else 'Base'}")

        x_base = np.array([np.mean(b) for b in DESIGN_BOUNDS])

        # 1. Sensitivity Analysis (Existing Technique)
        G = optimizer.compute_jacobian(x_base, deltas)
        G_abs = np.abs(G)

        opt_designs = {}
        modes = ["energy", "power", "stability"]
        obj_names = ["Energy", "Power", "Stability"]

        threshold = 0.5
        for i, mode in enumerate(modes):
            # Identify primary drivers for this objective
            max_s = np.max(G_abs[i, :]) + 1e-12
            active_indices = [j for j in range(len(DESIGN_SPACE)) if G_abs[i, j] / max_s > threshold]

            print(f"  Primary Drivers for {obj_names[i]}: {[DESIGN_SPACE[j] for j in active_indices]}")

            # 2. Optimization using identified parameters
            x0_active = x_base[active_indices]
            bounds_active = [DESIGN_BOUNDS[j] for j in active_indices]

            res = minimize(
                optimizer._objective, x0_active,
                args=(x_base, active_indices, deltas, mode),
                bounds=bounds_active, method='L-BFGS-B', options={'maxiter': 10}
            )

            x_opt = x_base.copy()
            x_opt[active_indices] = res.x
            opt_designs[mode] = x_opt

        # 3. Composition: weighted average of optimal design vectors
        final_x = (0.4 * opt_designs["energy"] + 0.3 * opt_designs["power"] + 0.3 * opt_designs["stability"])

        pt = ParamTransform(optimizer.base_params)
        pt.apply_physics_deltas(deltas)
        pt.apply_design_vector(final_x, DESIGN_SPACE)
        final_metrics = optimizer.simulate(pt.get_parameter_values())

        if final_metrics["success"]:
            material_results.append({
                "cat": cat, "salt": salt,
                "x": final_x, "metrics": final_metrics, "deltas": deltas,
                "score": final_metrics["energy"],
                "jacobian": G
            })

    if not material_results:
        print("No valid designs found.")
        return

    best = max(material_results, key=lambda r: r["score"])

    # Compute Metadata for identified drivers
    G_avg = best["jacobian"]
    G_row_max = np.max(np.abs(G_avg), axis=1).reshape(-1, 1) + 1e-12
    S = np.abs(G_avg) / G_row_max

    groups = {"Energy": [], "Power": [], "Stability": [], "Coupled": []}
    obj_names = ["Energy", "Power", "Stability"]
    for j, name in enumerate(DESIGN_SPACE):
        member_of = []
        for i, obj in enumerate(obj_names):
            if S[i, j] > 0.5:
                groups[obj].append(name)
                member_of.append(obj)
        if len(member_of) > 1: groups["Coupled"].append(name)

    output = {
        "materials": {
            "cathode": {"name": best["cat"].name if best["cat"] else "Base", "formula": best["cat"].composition if best["cat"] else "Base"},
            "electrolyte": {"salt": best["salt"].name if best["salt"] else "Base"}
        },
        "performance_envelope": {
             # Range metadata from individual objectives if needed, here providing composing result metrics
            "energy_Wh": round(best["metrics"]["energy"], 4),
            "power_W": round(best["metrics"]["power"], 4),
            "stability_Pa": round(best["metrics"]["mechanical_stability"], 4)
        },
        "knee_point_design": {
            "metrics": {k: round(float(v), 4) for k, v in best["metrics"].items() if k != "success"},
            "metadata": {"rank": 1}
        },
        "parameter_grouping": groups,
        "sensitivity_matrix": G_avg.tolist(),
        "design_specs_representative": dict(zip(DESIGN_SPACE, best["x"].tolist())),
        "combined_deltas_representative": best["deltas"]
    }

    print("\nFINAL HIERARCHICAL OPTIMIZATION OUTPUT:")
    print(json.dumps(output, indent=2))

    with open("result.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to result.json")

    return output

if __name__ == "__main__":
    optimizer = HierarchicalOptimizer()
    optimizer.run()
