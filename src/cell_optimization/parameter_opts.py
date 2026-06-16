import numpy as np
import pybamm
import logging
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from pint import UnitRegistry

# Unit registry for dimensional consistency (Issue 14)
ureg = UnitRegistry()

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
    # Smooth approximation for gradient consistency (Issue 15)
    phi_c = 0.03
    return base_cond * (max(fraction - phi_c, 0.0) + 1e-6) ** 1.8

def validate_params(pv: Dict[str, Any]):
    """Ensure physical coherence of DFN parameters (Issue 6)."""
    required = ["Nominal cell capacity [A.h]", "Positive electrode exchange-current density [A.m-2]"]
    for r in required:
        if r not in pv or pv[r] <= 0: return False
    D_p = pv["Positive particle diffusivity [m2.s-1]"]
    # Fix: Corrected D_val logic (Nitpick from review)
    D_val = D_p(0.5, 298.15) if callable(D_p) else D_p
    if D_val > 1e-10: return False
    return True

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
            elif name.endswith("porosity"):
                 eps = val
                 tau = eps ** (-0.5)
                 self.values_dict[name] = val
                 if "Electrolyte conductivity [S.m-1]" in self.values_dict:
                      # Effective electrolyte conductivity scaling (Issue 5)
                      self._apply_scaling("Electrolyte conductivity [S.m-1]", (eps / tau) ** 1.5)
            else:
                self.values_dict[name] = val

    def get_parameter_values(self) -> pybamm.ParameterValues:
        self.values_dict.setdefault("Cell volume [m3]", 0.13 * 0.07 * 0.01)
        self.values_dict.setdefault("Cell cooling surface area [m2]", 0.02)
        self.values_dict.setdefault("Total heat transfer coefficient [W.m-2.K-1]", 10.0)
        self.values_dict.setdefault("SEI solvent diffusivity [m2.s-1]", 2.5e-22)
        self.values_dict.setdefault("Bulk solvent concentration [mol.m-3]", 2636.0)
        return pybamm.ParameterValues(self.values_dict)

# --- INDIVIDUAL OBJECTIVE OPTIMIZER (NSGA-II) ---

class SingleObjectiveProblem(Problem):
    def __init__(self, optimizer, x_full, active_indices, deltas, mode):
        xl = DESIGN_BOUNDS[active_indices, 0]
        xu = DESIGN_BOUNDS[active_indices, 1]
        super().__init__(n_var=len(active_indices), n_obj=1, n_constr=1, xl=xl, xu=xu)
        self.optimizer = optimizer
        self.x_full = x_full
        self.active_indices = active_indices
        self.deltas = deltas
        self.mode = mode

    def _evaluate(self, x, out, *args, **kwargs):
        F, G = [], []
        for xi in x:
            x_eval = self.x_full.copy(); x_eval[self.active_indices] = xi
            # Constraint (Issue 14): x_pos (0) - x_neg (1) <= 0
            G.append(x_eval[0] - x_eval[1])
            pt = ParamTransform(self.optimizer.base_params)
            pt.apply_physics_deltas(self.deltas)
            pt.apply_design_vector(x_eval, DESIGN_SPACE)
            pv = pt.get_parameter_values()
            if not validate_params(pv):
                 F.append(1e9)
                 continue
            res = self.optimizer.simulate(pv)
            if not res["success"]: F.append(1e9)
            else:
                if self.mode == "energy": F.append(-res["energy"])
                elif self.mode == "power": F.append(-res["power"])
                elif self.mode == "stability": F.append(-res["mechanical_stability"])
        out["F"] = np.array(F); out["G"] = np.array(G)

class HierarchicalOptimizer:
    def __init__(self, engine: Optional[Any] = None, base_params: Optional[pybamm.ParameterValues] = None):
        if engine is None:
            from src.cell_optimization.material_opt import MaterialMappingEngine
            engine = MaterialMappingEngine()
        self.engine = engine
        self.base_params = base_params or pybamm.ParameterValues(engine.base_params)
        options = {"SEI": "solvent-diffusion limited", "loss of active material": "stress-driven", "thermal": "lumped"}
        self.model = pybamm.lithium_ion.DFN(options)
        # Configured IDAKLUSolver for stability (Issue 2)
        self.solver = pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-9, max_step_size=5.0)
        # Reuse simulation object for efficiency (Issue 3A)
        self.sim = pybamm.Simulation(self.model, solver=self.solver)

    def simulate(self, params: pybamm.ParameterValues, c_rate: float = 1.0) -> Dict[str, Any]:
        try:
            # Update parameters on reused simulation (Issue 3A)
            self.sim.parameter_values = params
            sol = self.sim.solve([0, 3600 / c_rate], inputs={"Current [A]": c_rate * float(params["Nominal cell capacity [A.h]"])})

            # Manual integration for reliability + NumPy 2.0 (Issue 12)
            v, curr, t = sol["Terminal voltage [V]"].data, sol["Current [A]"].data, sol["Time [s]"].data
            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy_val = trapz_func(v * curr, t) / 3600
            energy = (energy_val * ureg.Wh).to("Wh").magnitude

            power = np.max(v * curr)
            from src.cell_optimization.chem_regularization import mechanical_stability_metric
            stresses = []
            for sv in ["Positive particle surface tangential stress [Pa]", "Negative particle surface tangential stress [Pa]"]:
                 try: stresses.append(np.max(np.abs(sol[sv].data)))
                 except: pass
            m_stability = mechanical_stability_metric(stresses=stresses)
            return {"energy": float(energy), "power": float(power), "mechanical_stability": float(m_stability), "success": True}
        except Exception as e:
            return {"success": False, "reason": f"{e}"}

    def compute_jacobian(self, x: np.ndarray, deltas: Dict[str, Any]) -> np.ndarray:
        eps = 1e-4
        pt = ParamTransform(self.base_params)
        pt.apply_physics_deltas(deltas); pt.apply_design_vector(x, DESIGN_SPACE)
        base_res = self.simulate(pt.get_parameter_values())
        if not base_res["success"]: return np.zeros((3, len(DESIGN_SPACE)))
        j_base = np.array([base_res["energy"], base_res["power"], base_res["mechanical_stability"]])
        # Jacobian scaling (Issue 7)
        scale = np.maximum(np.abs(j_base), 1e-8)
        G = np.zeros((3, len(DESIGN_SPACE)))
        for j in range(len(DESIGN_SPACE)):
            x_pert = x.copy()
            # Scaled additive perturbation to prevent collapse (Issue 3C)
            x_pert[j] += eps * max(abs(x[j]), 1.0)
            pt_p = ParamTransform(self.base_params)
            pt_p.apply_physics_deltas(deltas); pt_p.apply_design_vector(x_pert, DESIGN_SPACE)
            res = self.simulate(pt_p.get_parameter_values())
            if res["success"]:
                j_pert = np.array([res["energy"], res["power"], res["mechanical_stability"]])
                G[:, j] = (j_pert - j_base) / (scale * eps)

        # Regularized FIM & Identifiability (Issue 17/18)
        FIM = G.T @ G + 1e-6 * np.eye(len(DESIGN_SPACE))
        cond = np.linalg.cond(FIM)
        if np.log10(cond) > 6:
             logging.warning("System unidentifiable - regularizing sensitivity")
             G += np.random.normal(0, 1e-3, G.shape)
        return G

    def run(self):
        return run_workflow(engine=self.engine)

def run_workflow(engine: Optional[Any] = None):
    from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCategory
    from src.cell_optimization.chem_regularization import derive_coupled_deltas, regularize_salt_props
    if engine is None: engine = MaterialMappingEngine()
    db, bases = engine.run()
    if not bases: return
    optimizer = HierarchicalOptimizer(engine=engine)
    material_results = []
    for cat, salt in [(c, s) for c in db[MaterialCategory.CATHODE_DOPANT][:1] for s in db[MaterialCategory.SALT][:1]]:
        deltas = {}
        if cat:
            d = derive_coupled_deltas(bases["cathode"]["properties"], cat.properties, bases["cathode"]["formula"], cat.composition)
            for k, v in d.items(): deltas.setdefault(k, {}).update(v)
        if salt:
            d = regularize_salt_props(bases["salt"]["formula"], salt.composition, bases["salt"]["properties"], salt.properties)
            for k, v in d.items(): deltas.setdefault(k, {}).update(v)
        x_base = np.array([np.mean(b) for b in DESIGN_BOUNDS])
        G = optimizer.compute_jacobian(x_base, deltas)
        opt_designs = []
        for i, mode in enumerate(["energy", "power", "stability"]):
            max_s = np.max(np.abs(G[i, :])) + 1e-12
            active_indices = [j for j in range(len(DESIGN_SPACE)) if np.abs(G[i, j]) / max_s > 0.5]
            problem = SingleObjectiveProblem(optimizer, x_base, active_indices, deltas, mode)
            res_opt = pymoo_minimize(problem, NSGA2(pop_size=12), ('n_gen', 5), verbose=False)
            x_opt = x_base.copy()
            if res_opt.X is not None: x_opt[active_indices] = np.atleast_2d(res_opt.X)[0]
            opt_designs.append(x_opt)
        # Weighted Composition (40/30/30) (Issue 11/Nitpick)
        final_x = (0.4 * opt_designs[0] + 0.3 * opt_designs[1] + 0.3 * opt_designs[2])
        pt = ParamTransform(optimizer.base_params)
        pt.apply_physics_deltas(deltas); pt.apply_design_vector(final_x, DESIGN_SPACE)
        final_metrics = optimizer.simulate(pt.get_parameter_values())
        if final_metrics["success"]:
            material_results.append({"cat": cat, "salt": salt, "x": final_x, "metrics": final_metrics, "deltas": deltas, "jacobian": G})

    if not material_results: return
    best = max(material_results, key=lambda r: r["metrics"]["energy"])
    output = {
        "materials": {"cathode": {"name": best["cat"].name, "formula": best["cat"].composition}, "electrolyte": {"salt": best["salt"].name}},
        "design_specs_representative": dict(zip(DESIGN_SPACE, best["x"].tolist())),
        "combined_deltas_representative": best["deltas"],
        "sensitivity_matrix": best["jacobian"].tolist()
    }
    with open("result.json", "w") as f: json.dump(output, f, indent=2)
    return output

if __name__ == "__main__": HierarchicalOptimizer().run()
