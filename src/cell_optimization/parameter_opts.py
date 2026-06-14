import numpy as np
import pybamm
import logging
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
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

DESIGN_BOUNDS = np.array([
    [30e-6, 150e-6], [30e-6, 150e-6],
    [0.2, 0.5], [0.2, 0.5],
    [1e-7, 10e-6], [1e-7, 10e-6],
    [0.3, 0.7],
    [0.02, 0.15]
])

# --- PHYSICS MODELS ---

def carbon_percolation_conductivity(fraction: float, base_cond: float = 100.0) -> float:
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
                 # Positive stability_shift means MORE stable, so we should REDUCE side reactions
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

# --- PARETO ANALYZER ---

class BatteryOptimizationProblem(Problem):
    def __init__(self, analyzer, material_deltas: Dict[str, Any]):
        super().__init__(n_var=len(DESIGN_SPACE), n_obj=3, n_constr=0, xl=DESIGN_BOUNDS[:, 0], xu=DESIGN_BOUNDS[:, 1])
        self.analyzer = analyzer
        self.material_deltas = material_deltas

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        for xi in x:
            pt = ParamTransform(self.analyzer.base_params)
            pt.apply_physics_deltas(self.material_deltas)
            pt.apply_design_vector(xi, DESIGN_SPACE)
            res = self.analyzer.simulate(pt.get_parameter_values())
            if res["success"]:
                F.append([-res["energy"], -res["power"], -res["mechanical_stability"]])
            else:
                print(f"Sim failed: {res.get('reason')}")
                F.append([1e9, 1e6, 1e9])
        out["F"] = np.array(F)

class ParetoAnalyzer:
    def __init__(self, base_params: pybamm.ParameterValues):
        self.base_params = base_params
        options = {"SEI": "solvent-diffusion limited", "loss of active material": "stress-driven", "thermal": "lumped"}
        self.model = pybamm.lithium_ion.SPM(options)
        # Use IDAKLUSolver for efficient sensitivity calculation as requested
        self.solver = pybamm.IDAKLUSolver()

    def simulate(self, params: pybamm.ParameterValues) -> Dict[str, Any]:
        try:
            # Add small resistance to avoid singularity if current is 0
            if "Internal resistance [Ohm]" not in params:
                params["Internal resistance [Ohm]"] = 0.001

            # Lower solver tolerance for speed and robustness
            self.solver.tol = 1e-3

            cap = float(params["Nominal cell capacity [A.h]"])
            sim = pybamm.Simulation(self.model, parameter_values=params, solver=self.solver)
            sol = sim.solve([0, 3600], inputs={"Current [A]": cap})

            v = sol["Terminal voltage [V]"].data
            curr = sol["Current [A]"].data
            p = v * curr
            t = sol["Time [s]"].data

            # Use fallback for trapz (numpy 2.0 rename)
            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy = trapz_func(p, t) / 3600
            power = np.max(p)

            stress_vars = ["Positive particle surface tangential stress [Pa]", "Negative particle surface tangential stress [Pa]"]
            max_stress = 1e-6
            for sv in stress_vars:
                 try:
                    # Avoid 'in' operator if it causes issues in this PyBaMM version
                    var_data = sol[sv].data
                    max_stress = max(max_stress, np.max(np.abs(var_data)))
                 except Exception:
                    continue

            m_stability = -max_stress

            # Constraint: Must discharge for at least 10 minutes to be valid
            if t[-1] < 600:
                return {"success": False, "reason": "Short discharge"}

            return {
                "energy": float(energy), "power": float(power), "mechanical_stability": float(m_stability),
                "avg_voltage": float(np.mean(v)), "success": True
            }
        except Exception as e:
            import traceback
            # logging.debug(f"Simulation failed: {e}")
            return {"success": False, "reason": f"{e}\n{traceback.format_exc()}"}

    def compute_jacobian(self, params: pybamm.ParameterValues, x: np.ndarray) -> np.ndarray:
        eps = 1e-4
        base_res = self.simulate(params)
        if not base_res["success"]: return np.zeros((3, len(DESIGN_SPACE)))
        j_base = np.array([base_res["energy"], base_res["power"], base_res["mechanical_stability"]])
        G = np.zeros((3, len(DESIGN_SPACE)))
        for j in range(len(DESIGN_SPACE)):
            x_pert = x.copy()
            x_pert[j] *= (1 + eps)
            pt = ParamTransform(self.base_params)
            pt.values_dict = dict(params)
            pt.apply_design_vector(x_pert, DESIGN_SPACE)
            res = self.simulate(pt.get_parameter_values())
            if res["success"]:
                j_pert = np.array([res["energy"], res["power"], res["mechanical_stability"]])
                G[:, j] = (j_pert - j_base) / (np.abs(j_base) * eps + 1e-12)
        return G

class DSMOptimizer(ParetoAnalyzer):
    def __init__(self):
        from src.cell_optimization.material_opt import MaterialMappingEngine
        engine = MaterialMappingEngine()
        super().__init__(pybamm.ParameterValues(engine.base_params))
        self.engine = engine
        self.selected_dopant_idx = 0
        self.selected_salt_idx = 0
        self.mtms_enabled = True

    def run(self):
        return run_workflow()

def find_knee_point(pareto_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    objs = np.array([[p["metrics"]["energy"], p["metrics"]["power"], p["metrics"]["mechanical_stability"]] for p in pareto_set])
    mins, maxs = objs.min(axis=0), objs.max(axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    norm_objs = (objs - mins) / denom
    distances = np.linalg.norm(1.0 - norm_objs, axis=1)
    best_idx = np.argmin(distances)

    # Calculate tradeoff metadata
    res = pareto_set[best_idx].copy()
    res["pareto_metadata"] = {
        "knee_distance": float(distances[best_idx]),
        "total_points": len(pareto_set),
        "rank": 1
    }
    return res

def run_workflow():
    from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCategory
    from src.cell_optimization.chem_regularization import derive_coupled_deltas, regularize_salt_props
    engine = MaterialMappingEngine()
    db, bases = engine.run()
    if not bases: return
    analyzer = ParetoAnalyzer(pybamm.ParameterValues(engine.base_params))

    print("Executing Two-Level Pareto Design Space Search (Layer 3)...")

    candidate_combinations = []
    # Outer Loop: Material Candidates
    cathodes = db[MaterialCategory.CATHODE_DOPANT] if db[MaterialCategory.CATHODE_DOPANT] else [None]
    salts = db[MaterialCategory.SALT] if db[MaterialCategory.SALT] else [None]

    for cat in cathodes[:2]:
        for salt in salts[:2]:
            candidate_combinations.append((cat, salt))

    all_pareto_points = []
    for cat, salt in candidate_combinations:
        deltas = {}
        if cat:
            d = derive_coupled_deltas(bases["cathode"]["properties"], cat.properties, bases["cathode"]["formula"], cat.composition)
            for k, v in d.items(): deltas.setdefault(k, {}).update(v)
        if salt:
            d = regularize_salt_props(bases["salt"]["solution"], salt.properties)
            for k, v in d.items(): deltas.setdefault(k, {}).update(v)

        # NSGA-II Search
        print(f"Running NSGA-II for {cat.name if cat else 'Base'} + {salt.name if salt else 'Base'}...")
        problem = BatteryOptimizationProblem(analyzer, deltas)
        algorithm = NSGA2(pop_size=12)
        res_opt = pymoo_minimize(problem, algorithm, ('n_gen', 5), verbose=False)

        if res_opt.X is not None:
            X_p = np.atleast_2d(res_opt.X)
            F_p = np.atleast_2d(res_opt.F)
            for i in range(len(X_p)):
                metrics = {"energy": -F_p[i, 0], "power": -F_p[i, 1], "mechanical_stability": -F_p[i, 2]}
                if metrics["energy"] > 0:
                     all_pareto_points.append({"cat": cat, "salt": salt, "x": X_p[i], "metrics": metrics, "deltas": deltas})

    if not all_pareto_points:
        print("No Pareto points found.")
        return

    # Pareto Filter
    pareto_final = []
    for i, c1 in enumerate(all_pareto_points):
        dominated = False
        obj_a = np.array([c1["metrics"]["energy"], c1["metrics"]["power"], c1["metrics"]["mechanical_stability"]])
        for j, c2 in enumerate(all_pareto_points):
            if i == j: continue
            obj_b = np.array([c2["metrics"]["energy"], c2["metrics"]["power"], c2["metrics"]["mechanical_stability"]])
            if np.all(obj_b >= obj_a) and np.any(obj_b > obj_a): dominated = True; break
        if not dominated: pareto_final.append(c1)

    best = find_knee_point(pareto_final)

    print(f"Aggregating sensitivities over {len(pareto_final)} Pareto points...")
    G_all = []
    for p in pareto_final:
        pt = ParamTransform(analyzer.base_params)
        pt.apply_physics_deltas(p["deltas"])
        pt.apply_design_vector(p["x"], DESIGN_SPACE)
        G_k = analyzer.compute_jacobian(pt.get_parameter_values(), p["x"])
        G_all.append(np.abs(G_k))

    G_avg = np.mean(G_all, axis=0)
    G_row_max = np.max(G_avg, axis=1).reshape(-1, 1) + 1e-12
    S = G_avg / G_row_max

    threshold = 0.5
    groups = {"Energy": [], "Power": [], "Stability": [], "Coupled": []}
    obj_names = ["Energy", "Power", "Stability"]
    for j, name in enumerate(DESIGN_SPACE):
        member_of = []
        for i, obj in enumerate(obj_names):
            if S[i, j] > threshold:
                groups[obj].append(name)
                member_of.append(obj)
        if len(member_of) > 1: groups["Coupled"].append(name)

    output = {
        "materials": {
            "cathode": {"name": best["cat"].name if best["cat"] else "Base", "formula": best["cat"].composition if best["cat"] else "Base"},
            "electrolyte": {"salt": best["salt"].name if best["salt"] else "Base"}
        },
        "performance_envelope": {
            "energy_Wh_range": [round(float(min(p["metrics"]["energy"] for p in pareto_final)), 3), round(float(max(p["metrics"]["energy"] for p in pareto_final)), 3)],
            "power_W_range": [round(float(min(p["metrics"]["power"] for p in pareto_final)), 3), round(float(max(p["metrics"]["power"] for p in pareto_final)), 3)],
            "stability_Pa_range": [round(float(min(p["metrics"]["mechanical_stability"] for p in pareto_final)), 3), round(float(max(p["metrics"]["mechanical_stability"] for p in pareto_final)), 3)]
        },
        "knee_point_design": {
            "metrics": {k: round(float(v), 4) for k, v in best["metrics"].items() if k != "success"},
            "metadata": best["pareto_metadata"]
        },
        "parameter_grouping": groups,
        "sensitivity_matrix": G_avg.tolist(),
        "design_specs_representative": dict(zip(DESIGN_SPACE, best["x"].tolist())),
        "combined_deltas_representative": best["deltas"]
    }
    print("\nFINAL ROBUST KNEE-POINT OUTPUT:")
    print(json.dumps(output, indent=2))

    with open("result.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to result.json")

    return output

if __name__ == "__main__":
    optimizer = DSMOptimizer()
    optimizer.run()
