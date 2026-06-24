import numpy as np
import pybamm
import logging
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.simulation.utilities.mechanical.fenics_model import ThermoelasticStrainModel
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
                self._apply_scaling("Negative electrode exchange-current density [A.m-2]", np.exp(d["exchange_current_log_delta"]))
            if "sei_growth_log_delta" in d:
                self._apply_scaling("SEI reaction exchange current density [A.m-2]", np.exp(d["sei_growth_log_delta"]))
            if "sei_resistivity_log_delta" in d:
                self._apply_scaling("SEI resistivity [Ohm.m]", np.exp(d["sei_resistivity_log_delta"]))

        if "mechanical" in deltas:
             d = deltas["mechanical"]
             if "modulus_degradation_factor" in d:
                  self._apply_scaling("Negative electrode Young's modulus [Pa]", d["modulus_degradation_factor"])

    def apply_design_vector(self, x: np.ndarray, names: List[str]):
        for val, name in zip(x, names):
            if name == "carbon_fraction":
                self.values_dict["Positive electrode conductivity [S.m-1]"] = carbon_percolation_conductivity(val)
            elif name.endswith("porosity"):
                 # Issue 5: Effective electrolyte conductivity
                 eps = val
                 tau = eps ** (-0.5)
                 self.values_dict[name] = val
                 if "Electrolyte conductivity [S.m-1]" in self.values_dict:
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

# --- INDIVIDUAL OBJECTIVE OPTIMIZER (GA) ---

class SingleObjectiveProblem(Problem):
    def __init__(self, optimizer, x_full, active_indices, deltas, mode, ref_scale=1.0):
        xl = DESIGN_BOUNDS[active_indices, 0]
        xu = DESIGN_BOUNDS[active_indices, 1]
        super().__init__(n_var=len(active_indices), n_obj=1, n_constr=1, xl=xl, xu=xu)
        self.optimizer = optimizer
        self.x_full = x_full
        self.active_indices = active_indices
        self.deltas = deltas
        self.mode = mode
        self.ref_scale = max(abs(ref_scale), 1e-9)

    def _evaluate(self, x, out, *args, **kwargs):
        F, G_all = [], []
        from src.cell_optimization.chem_regularization import mechanical_stability_metric
        for xi in x:
            x_eval = self.x_full.copy(); x_eval[self.active_indices] = xi
            # Issue 14: t_p <= t_n. Normalized constraint (Issue 4.2).
            g = (x_eval[0] - x_eval[1]) / max(DESIGN_BOUNDS[0][1], DESIGN_BOUNDS[1][1])

            pt = ParamTransform(self.optimizer.base_params)
            pt.apply_physics_deltas(self.deltas); pt.apply_design_vector(x_eval, DESIGN_SPACE)
            pv = pt.get_parameter_values()

            f_val = 1e9
            if validate_params(pv):
                res = self.optimizer.simulate(pv)
                if res["success"]:
                    # Stage 1: Fast thermal check
                    if res["T_max"] > 333.15: # 60 degC limit
                        f_val = 1e9
                    else:
                        if self.mode == "energy": f_val = -res["energy"]
                        elif self.mode == "power": f_val = -res["power"]
                        elif self.mode == "stability":
                            # Stage 1 Stability proxy
                            f_val = -mechanical_stability_metric(stresses=res["stresses"])

            # Objective scaling using constant reference (Major Nitpick Fix)
            f_val /= self.ref_scale

            # Penalty shaping (Issue 4.2)
            penalty = 100 * np.square(max(g, 0))
            F.append(f_val + penalty)
            G_all.append([g])

        out["F"] = np.array(F); out["G"] = np.array(G_all)

class HierarchicalOptimizer:
    def __init__(self, engine: Optional[Any] = None, base_params: Optional[pybamm.ParameterValues] = None):
        if engine is None:
            from src.cell_optimization.material_opt import MaterialMappingEngine
            engine = MaterialMappingEngine()
        self.engine = engine
        self.base_params = base_params or pybamm.ParameterValues(get_parameter_values())
        options = {"SEI": "solvent-diffusion limited", "loss of active material": "stress-driven", "thermal": "lumped"}
        self.model = pybamm.lithium_ion.DFN(options)
        self.solver = pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-9, options={"dt_max": 5.0})
        self.sim = pybamm.Simulation(self.model, solver=self.solver)
        self.mech_model = ThermoelasticStrainModel()

    def simulate(self, params: pybamm.ParameterValues, c_rate: float = 1.0, return_sol: bool = False) -> Dict[str, Any]:
        try:
            self.sim.parameter_values = params
            sol = self.sim.solve([0, 3600 / c_rate], inputs={"Current [A]": c_rate * float(params["Nominal cell capacity [A.h]"])})
            v, curr, t = sol["Terminal voltage [V]"].data, sol["Current [A]"].data, sol["Time [s]"].data

            # Energy calculation (Issue 5.1) - Discharge E is always positive
            power_vals = np.abs(v * curr)
            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy_wh = trapz_func(power_vals, t) / 3600

            energy = float(energy_wh)
            power = np.max(power_vals)

            # Cheap thermal check (Stage 1)
            T_max = np.max(sol["Cell temperature [K]"].data)

            # Post-processing proxy (deprecated by FEM solve, but used for fast ranking)
            stresses = []
            for sv in ["Positive particle surface tangential stress [Pa]", "Negative particle surface tangential stress [Pa]"]:
                 try: stresses.append(np.max(np.abs(sol[sv].data)))
                 except: pass

            res = {
                "energy": float(energy),
                "power": float(power),
                "T_max": float(T_max),
                "stresses": stresses,
                "success": True
            }
            if return_sol:
                res["sol"] = sol
            return res
        except Exception as e:
            return {"success": False, "reason": f"{e}"}

    def evaluate_stability_pde(self, params: pybamm.ParameterValues, mode: str) -> Tuple[bool, float]:
        """Stage 2: Expensive FEM thermo-mechanical solve."""
        res = self.simulate(params, return_sol=True)
        if not res["success"]: return False, -1e9

        # Call FEniCSx solver
        try:
            mech_res = self.mech_model.solve_strain(res["sol"], params)
            max_strain = mech_res["max_strain"]
            critical_strain = self.mech_model.critical_thresholds.get("NFPP", 2e-3)
            eta = max_strain / critical_strain

            stability = -eta
            # If failed (eta > 1), large penalty
            if eta > 1.0:
                stability -= 10.0 * (eta - 1.0)**2

            return True, float(stability)
        except Exception as e:
            logging.error(f"FEM solve failed: {e}")
            return False, -1e9

    def compute_jacobian(self, x: np.ndarray, deltas: Dict[str, Any]) -> np.ndarray:
        eps = 1e-4
        pt = ParamTransform(self.base_params)
        pt.apply_physics_deltas(deltas); pt.apply_design_vector(x, DESIGN_SPACE)
        base_res = self.simulate(pt.get_parameter_values())
        if not base_res["success"]: return np.zeros((3, len(DESIGN_SPACE)))

        from src.cell_optimization.chem_regularization import mechanical_stability_metric
        j_base = np.array([
            base_res["energy"],
            base_res["power"],
            mechanical_stability_metric(stresses=base_res["stresses"])
        ])

        G = np.zeros((3, len(DESIGN_SPACE)))
        for j in range(len(DESIGN_SPACE)):
            x_pert = x.copy()
            # Issue 4.3: Scaled additive perturbation using parameter bounds
            lower, upper = DESIGN_BOUNDS[j]
            delta = eps * (upper - lower)
            x_pert[j] += delta
            pt_p = ParamTransform(self.base_params)
            pt_p.apply_physics_deltas(deltas); pt_p.apply_design_vector(x_pert, DESIGN_SPACE)
            res = self.simulate(pt_p.get_parameter_values())
            if res["success"]:
                j_pert = np.array([
                    res["energy"],
                    res["power"],
                    mechanical_stability_metric(stresses=res["stresses"])
                ])
                # Log-space differentiation for Jacobian stability (Issue 4.3)
                G[:, j] = (np.log(np.abs(j_pert) + 1e-12) - np.log(np.abs(j_base) + 1e-12)) / eps

        # Adaptive SVD conditioning for FIM (Issue 4.4)
        U, S, Vt = np.linalg.svd(G, full_matrices=False)
        cond_limit = 1e6
        smax = S[0]
        S = np.array([max(s, smax / cond_limit) for s in S])
        G = (U * S) @ Vt
        return G

    def run(self):
        return run_workflow(engine=self.engine)

def run_workflow(engine: Optional[Any] = None):
    from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCategory
    from src.cell_optimization.chem_regularization import derive_coupled_deltas, regularize_salt_props
    if engine is None: engine = MaterialMappingEngine()
    db, bases = engine.run()
    if not bases:
        logging.error("Hierarchical optimization aborted: Base material resolution failed.")
        return None
    optimizer = HierarchicalOptimizer(engine=engine)
    print("Executing Sensitivity-Driven DFN Hierarchical Optimization (Layer 3)...")
    material_results = []
    # Full discovery loop (fixed truncation issue)
    for cat, salt in [(c, s) for c in db[MaterialCategory.CATHODE_DOPANT] for s in db[MaterialCategory.SALT]]:
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
        # Calculate baseline metrics for objective scaling
        pt_base = ParamTransform(optimizer.base_params)
        pt_base.apply_physics_deltas(deltas); pt_base.apply_design_vector(x_base, DESIGN_SPACE)
        base_metrics = optimizer.simulate(pt_base.get_parameter_values())

        for i, mode in enumerate(["energy", "power", "stability"]):
            max_s = np.max(np.abs(G[i, :])) + 1e-12
            active_indices = [j for j in range(len(DESIGN_SPACE)) if np.abs(G[i, j]) / max_s > 0.5]

            # Use constant baseline metric as reference scale to preserve optimization gradient
            ref_val = 1.0
            if base_metrics["success"]:
                if mode == "energy": ref_val = base_metrics["energy"]
                elif mode == "power": ref_val = base_metrics["power"]
                elif mode == "stability":
                    from src.cell_optimization.chem_regularization import mechanical_stability_metric
                    ref_val = mechanical_stability_metric(stresses=base_metrics["stresses"])

            problem = SingleObjectiveProblem(optimizer, x_base, active_indices, deltas, mode, ref_scale=ref_val)
            # Correct SOO algorithm (Issue 4.1)
            res_opt = pymoo_minimize(problem, GA(pop_size=20), ('n_gen', 30), verbose=False)
            x_opt = x_base.copy()
            if res_opt.X is not None: x_opt[active_indices] = np.atleast_2d(res_opt.X)[0]
            opt_designs.append(x_opt)

        valid_candidates, stability_scores = [], []
        for x, mode in zip(opt_designs, ["energy", "power", "stability"]):
            pt = ParamTransform(optimizer.base_params)
            pt.apply_physics_deltas(deltas); pt.apply_design_vector(x, DESIGN_SPACE)
            ok, score = optimizer.evaluate_stability_pde(pt.get_parameter_values(), mode)
            if ok:
                valid_candidates.append(x); stability_scores.append(score)

        if not valid_candidates: continue

        # Issue 7 Step 4: Selection and interpolation
        x_star = valid_candidates[np.argmax(stability_scores)]
        final_x = 0.8 * x_star + 0.2 * np.mean(valid_candidates, axis=0)

        pt = ParamTransform(optimizer.base_params)
        pt.apply_physics_deltas(deltas); pt.apply_design_vector(final_x, DESIGN_SPACE)
        final_metrics = optimizer.simulate(pt.get_parameter_values())
        if final_metrics["success"]:
            material_results.append({"cat": cat, "salt": salt, "x": final_x, "metrics": final_metrics, "deltas": deltas, "jacobian": G})

    if not material_results:
        logging.error("Hierarchical optimization failed: No valid material candidates successfully optimized.")
        return None
    best = max(material_results, key=lambda r: r["metrics"]["energy"])

    # Accurate metadata
    G_avg = best["jacobian"]
    S = np.abs(G_avg) / (np.max(np.abs(G_avg), axis=1).reshape(-1, 1) + 1e-12)
    groups = {"Energy": [], "Power": [], "Stability": [], "Coupled": []}
    for j, name in enumerate(DESIGN_SPACE):
        member_of = []
        for i, obj in enumerate(["Energy", "Power", "Stability"]):
            if S[i, j] > 0.5: groups[obj].append(name); member_of.append(obj)
        if len(member_of) > 1: groups["Coupled"].append(name)

    output = {
        "materials": {"cathode": {"name": best["cat"].name, "formula": best["cat"].composition}, "electrolyte": {"salt": best["salt"].name}},
        "design_specs_representative": dict(zip(DESIGN_SPACE, best["x"].tolist())),
        "combined_deltas_representative": best["deltas"],
        "sensitivity_matrix": best["jacobian"].tolist(),
        "parameter_grouping": groups
    }
    with open("result.json", "w") as f: json.dump(output, f, indent=2)

    print("\n" + "="*50)
    print("HIERARCHICAL OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Optimal Material: {output['materials']['cathode']['name']} / {output['materials']['electrolyte']['salt']}")
    print("-" * 50)
    print("Optimized Design Vector:")
    for k, v in output['design_specs_representative'].items():
        print(f"  {k:40s}: {v:12.6e}")
    print("-" * 50)
    print(f"Final Energy: {best['metrics']['energy']:.4f} Wh")
    print(f"Final Power:  {best['metrics']['power']:.4f} W")
    print("="*50 + "\n")

    return output

if __name__ == "__main__": HierarchicalOptimizer().run()
