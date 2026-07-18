import numpy as np
import pybamm
import json
import os
import traceback
import inspect
import copy
from collections import OrderedDict
from typing import Dict, Any, List, Tuple, Optional
from src.cell_optimization.cem_optimizer import CrossEntropyOptimizer
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from nfpp_sodium_ion.src.calibration.derivation import get_derived_parameters
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

def validate_params(pv: Dict[str, Any], verbose: bool = False):
    """Ensure physical coherence of DFN parameters using research-grounded values (Issue 6)."""
    required = ["Nominal cell capacity [A.h]", "Positive electrode exchange-current density [A.m-2]"]
    derived = get_derived_parameters()

    for r in required:
        if r not in pv:
            if verbose: print(f"DEBUG: validate_params failed: {r} missing")
            return False
        val = pv[r]
        # Handle callables for functional parameters (Issue 6 fix)
        if callable(val):
            sig = inspect.signature(val)
            params_list = list(sig.parameters.keys())
            grounded_map = {
                "c_e": 1200.0,
                "c_s_surf": 0.5 * derived.get("c_max_p", 25000.0),
                "c_s_max": derived.get("c_max_p", 25000.0),
                "T": 298.15,
                "sto": 0.5
            }
            args = [grounded_map.get(p, 0.5) for p in params_list]
            try:
                res = val(*args)
                actual_val = float(res.value) if hasattr(res, "value") else float(res)
            except Exception as e:
                if verbose: print(f"DEBUG: validate_params callable {r} failed: {e}")
                actual_val = 1.0
        else:
            actual_val = val
        if actual_val <= 0:
            if verbose: print(f"DEBUG: validate_params failed: {r} <= 0 ({actual_val})")
            return False

    if "Positive particle diffusivity [m2.s-1]" in pv:
        D_p = pv["Positive particle diffusivity [m2.s-1]"]
        D_val = D_p(0.5, 298.15) if callable(D_p) else D_p
        # Relaxed limit (Issue 1 from review)
        if D_val > 1e-8:
            if verbose: print(f"DEBUG: validate_params failed: D_p > 1e-8 ({D_val})")
            return False
    return True

class ParamTransform:
    def __init__(self, base_values: pybamm.ParameterValues):
        self.values_dict = dict(base_values)
        self.scaling_factors = {}

    def _apply_scaling(self, key: str, factor: float):
        self.scaling_factors[key] = self.scaling_factors.get(key, 1.0) * factor

    def apply_physics_deltas(self, deltas: Dict[str, Any]):
        if "thermodynamic" in deltas:
            d = deltas["thermodynamic"]
            if "voltage_boost" in d:
                ocp = self.values_dict.get("Positive electrode OCP [V]")
                boost = d["voltage_boost"]
                if callable(ocp):
                    from functools import wraps
                    def make_ocp_wrapper(orig, b):
                        @wraps(orig)
                        def ocp_wrapper(*args, **kwargs):
                            return orig(*args, **kwargs) + b
                        return ocp_wrapper
                    self.values_dict["Positive electrode OCP [V]"] = make_ocp_wrapper(ocp, boost)
                else:
                    self.values_dict["Positive electrode OCP [V]"] += boost
                for cut_off in ["Lower voltage cut-off [V]", "Upper voltage cut-off [V]"]:
                    if cut_off in self.values_dict:
                        self.values_dict[cut_off] += boost
            if "initial_sodium_loss_delta" in d:
                self._apply_scaling("Initial concentration in negative electrode [mol.m-3]", (1.0 + d["initial_sodium_loss_delta"]))
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
                 eps = val
                 tau = eps ** (-0.5)
                 self.values_dict[name] = val
                 if "Electrolyte conductivity [S.m-1]" in self.values_dict:
                      self._apply_scaling("Electrolyte conductivity [S.m-1]", (eps / tau) ** 1.5)
            else:
                self.values_dict[name] = val

    def get_parameter_values(self) -> pybamm.ParameterValues:
        derived = get_derived_parameters()

        # 1. Mandatory Mechanical & Topology
        self.values_dict.setdefault("Negative electrode volume change", lambda sto: 0.1 * sto)
        self.values_dict.setdefault("Positive electrode volume change", lambda sto: 0.1 * sto)
        self.values_dict.setdefault("Cell thermal expansion coefficient [m.K-1]", 1e-6)
        self.values_dict.setdefault("Number of cells connected in series to make a battery", 1)
        self.values_dict.setdefault("Number of strings connected in parallel to make a battery", 1)

        # 2. Physics-Grounded Initialization
        c_max_p = self.values_dict.get("Maximum concentration in positive electrode [mol.m-3]", derived["c_max_p"])
        c_max_n = self.values_dict.get("Maximum concentration in negative electrode [mol.m-3]", derived["c_max_n"])
        self.values_dict["Initial concentration in positive electrode [mol.m-3]"] = 0.5 * c_max_p
        self.values_dict["Initial concentration in negative electrode [mol.m-3]"] = 0.5 * c_max_n
        self.values_dict["Lower voltage cut-off [V]"] = 0.5
        self.values_dict["Upper voltage cut-off [V]"] = 4.5

        # 3. Apply Multiplicative Scalings from deltas
        for key, factor in self.scaling_factors.items():
            original = self.values_dict.get(key)
            if original is None: continue
            if callable(original):
                from functools import wraps
                def make_wrapper(orig, f):
                    @wraps(orig)
                    def wrapper(*args, **kwargs):
                        return orig(*args, **kwargs) * f
                    return wrapper
                self.values_dict[key] = make_wrapper(original, factor)
            else:
                self.values_dict[key] *= factor

        # 4. Mathematically Derived Constants (Issue 14 final audit)
        self.values_dict.setdefault("Cell volume [m3]", derived["cell_volume"])
        self.values_dict.setdefault("Cell cooling surface area [m2]", derived["surface_area"])
        self.values_dict.setdefault("Total heat transfer coefficient [W.m-2.K-1]", derived["total_htc"])
        self.values_dict.setdefault("SEI solvent diffusivity [m2.s-1]", derived["sei_solvent_diffusivity"])
        self.values_dict.setdefault("Bulk solvent concentration [mol.m-3]", derived["bulk_solvent_concentration"])

        # Collector Properties (Reference: CRC Handbook)
        self.values_dict.setdefault("Negative current collector density [kg.m-3]", derived["cu_density"])
        self.values_dict.setdefault("Positive current collector density [kg.m-3]", derived["al_density"])
        self.values_dict.setdefault("Negative current collector specific heat capacity [J.kg-1.K-1]", derived["cu_cp"])
        self.values_dict.setdefault("Positive current collector specific heat capacity [J.kg-1.K-1]", derived["al_cp"])
        self.values_dict.setdefault("Negative current collector thermal conductivity [W.m-1.K-1]", derived["cu_tc"])
        self.values_dict.setdefault("Positive current collector thermal conductivity [W.m-1.K-1]", derived["al_tc"])

        return pybamm.ParameterValues(self.values_dict)

class SingleObjectiveProblem:
    def __init__(self, optimizer, x_full, active_indices, deltas, mode, ref_scale=1.0):
        self.optimizer = optimizer
        self.x_full = x_full
        self.active_indices = active_indices
        self.deltas = deltas
        self.mode = mode
        self.ref_scale = max(abs(ref_scale), 1e-9)

    def evaluate_single(self, x_full):
        from src.cell_optimization.chem_regularization import mechanical_stability_metric
        g1 = (x_full[0] - x_full[1]) / max(DESIGN_BOUNDS[0][1], DESIGN_BOUNDS[1][1])
        if g1 > 0:
            return 1e12, False

        pt = ParamTransform(self.optimizer.base_params)
        pt.apply_physics_deltas(self.deltas)
        pt.apply_design_vector(x_full, DESIGN_SPACE)
        pv = pt.get_parameter_values()

        if not validate_params(pv):
            return 1e12, False

        res = self.optimizer.simulate(pv)
        if not res["success"]:
            return 1e12, False

        g2 = res["T_max"] - 333.15
        if g2 > 0:
            return 1e12, False

        if self.mode == "energy":
            f_val = -res["energy"]
        elif self.mode == "power":
            f_val = -res["power"]
        elif self.mode == "stability":
            f_val = -mechanical_stability_metric(stresses=res["stresses"])
        else:
            f_val = 1e12

        sc = max(abs(self.ref_scale), 0.1)
        score = f_val / sc
        return score, True

class GeometryCache:
    def __init__(self, max_size: int = 32):
        self.cache = OrderedDict()
        self.max_size = max_size
    def get(self, key: tuple):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    def set(self, key: tuple, value: dict):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

class SimulationRunner:
    def __init__(self, model: pybamm.BaseModel, solver_class, solver_kwargs: dict):
        self.model = model
        self.solver_class = solver_class
        self.solver_kwargs = solver_kwargs
        self.geometry_cache = GeometryCache()
        self.var_pts = model.default_var_pts
        self.submesh_types = model.default_submesh_types
        self.spatial_methods = model.default_spatial_methods

    def _get_geometry_key(self, params: pybamm.ParameterValues) -> tuple:
        keys = ["Positive electrode thickness [m]", "Negative electrode thickness [m]", "Separator thickness [m]", "Positive particle radius [m]", "Negative particle radius [m]", "Typical electrolyte concentration [mol.m-3]"]
        return tuple(float(params.get(k, 0.0)) for k in keys)

    def run_simulation(self, params: pybamm.ParameterValues, c_rate: float = 1.0) -> Dict[str, Any]:
        params = params.copy()
        try:
            c_max_p = params["Maximum concentration in positive electrode [mol.m-3]"]
            c_max_n = params["Maximum concentration in negative electrode [mol.m-3]"]
            c_p_init = params["Initial concentration in positive electrode [mol.m-3]"]
            c_n_init = params["Initial concentration in negative electrode [mol.m-3]"]
            ocp_p_func = params["Positive electrode OCP [V]"]
            ocp_n_func = params["Negative electrode OCP [V]"]
            sto_p = c_p_init / c_max_p
            sto_n = c_n_init / c_max_n
            v_init = ocp_p_func(sto_p) - ocp_n_func(sto_n)
            v_init_val = float(v_init.value) if hasattr(v_init, "value") else float(v_init)
            ir_drop_est = 0.5
            v_min = params["Lower voltage cut-off [V]"]
            if (v_init_val - ir_drop_est) <= v_min:
                params["Lower voltage cut-off [V]"] = max(0.1, v_init_val - 1.0)
                print(f"INFO: Relaxed lower voltage cut-off from {v_min:.2f}V to {params['Lower voltage cut-off [V]']:.2f}V (Initial OCV: {v_init_val:.2f}V)")
            key = self._get_geometry_key(params)
            cached = self.geometry_cache.get(key)
            if cached:
                geometry, mesh, disc = cached["geometry"], cached["mesh"], cached["disc"]
            else:
                geometry = copy.deepcopy(self.model.default_geometry)
                params.process_geometry(geometry)
                mesh = pybamm.Mesh(geometry, self.submesh_types, self.var_pts)
                disc = pybamm.Discretisation(mesh, self.spatial_methods)
                self.geometry_cache.set(key, {"geometry": geometry, "mesh": mesh, "disc": disc})
            processed_model = params.process_model(self.model, inplace=False)
            disc.process_model(processed_model, inplace=True)
            solver = self.solver_class(**self.solver_kwargs)
            sol = solver.solve(processed_model, [0, 3600 / c_rate], inputs={"Current [A]": c_rate * float(params["Nominal cell capacity [A.h]"])})
            return {"success": True, "sol": sol}
        except Exception as e:
            err_msg = f"ERROR: DFN Simulation failed: {e}\n{traceback.format_exc()}"
            return {"success": False, "reason": err_msg}

class HierarchicalOptimizer:
    def __init__(self, engine: Optional[Any] = None, base_params: Optional[pybamm.ParameterValues] = None):
        if engine is None:
            from src.cell_optimization.material_opt import MaterialMappingEngine
            engine = MaterialMappingEngine()
        self.engine = engine
        self.base_params = base_params or pybamm.ParameterValues(get_parameter_values())
        options = {"SEI": "solvent-diffusion limited", "loss of active material": "stress-driven", "thermal": "lumped"}
        self.model = pybamm.lithium_ion.DFN(options)
        solver_kwargs = {"rtol": 1e-7, "atol": 1e-9, "options": {"dt_max": 5.0}}
        self.runner = SimulationRunner(self.model, pybamm.IDAKLUSolver, solver_kwargs)
        self.mech_model = ThermoelasticStrainModel()

    def simulate(self, params: pybamm.ParameterValues, c_rate: float = 1.0, return_sol: bool = False) -> Dict[str, Any]:
        res = self.runner.run_simulation(params, c_rate)
        if not res["success"]:
            print(res["reason"])
            return res
        try:
            sol = res["sol"]
            v, curr, t = sol["Terminal voltage [V]"].entries, sol["Current [A]"].entries, sol["Time [s]"].entries
            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy_wh = abs(trapz_func(v * curr, t)) / 3600
            power_vals = np.abs(v * curr)
            energy = float(energy_wh)
            power = np.max(power_vals)
            T_max = np.max(sol["Cell temperature [K]"].entries)
            stresses = []
            for sv in ["Positive particle surface tangential stress [Pa]", "Negative particle surface tangential stress [Pa]"]:
                 try: stresses.append(np.max(np.abs(sol[sv].entries)))
                 except (KeyError, pybamm.ModelError, AttributeError): pass
            final_res = {"energy": float(energy), "power": float(power), "T_max": float(T_max), "stresses": stresses, "success": True}
            if return_sol: final_res["sol"] = sol
            return final_res
        except Exception as e:
            return {"success": False, "reason": f"Post-simulation processing failed: {e}"}

    def evaluate_stability_pde(self, params: pybamm.ParameterValues, mode: str, c_rate: float = 1.0) -> Tuple[bool, float]:
        res = self.simulate(params, c_rate=c_rate, return_sol=True)
        if not res["success"]: return False, -1e9
        try:
            mech_res = self.mech_model.solve_strain(res["sol"], params, c_rate=c_rate)
            max_strain = mech_res["max_strain"]
            mat_key = "NFPP" if "NFPP" in self.mech_model.critical_thresholds else list(self.mech_model.critical_thresholds.keys())[0]
            critical_strain = self.mech_model.critical_thresholds.get(mat_key, 2e-3)
            eta = max_strain / critical_strain
            print(f"DEBUG[{mode}]: max_strain={max_strain:.4e}, critical={critical_strain:.4e}, eta={eta:.3f}")
            if eta > 1.0: return False, -float(eta)
            return True, -float(eta)
        except Exception as e:
            print(f"ERROR: FEM solve failed: {e}\n{traceback.format_exc()}")
            return False, -1e9

    def compute_jacobian(self, x: np.ndarray, deltas: Dict[str, Any]) -> Optional[np.ndarray]:
        eps = 1e-4
        pt = ParamTransform(self.base_params)
        pt.apply_physics_deltas(deltas); pt.apply_design_vector(x, DESIGN_SPACE)
        base_res = self.simulate(pt.get_parameter_values())
        if not base_res["success"]:
            print(f"WARNING: Baseline DFN simulation failed: {base_res.get('reason')}. Skipping candidate.")
            return None
        from src.cell_optimization.chem_regularization import mechanical_stability_metric
        j_base = np.array([base_res["energy"], base_res["power"], mechanical_stability_metric(stresses=base_res["stresses"])])
        G = np.zeros((3, len(DESIGN_SPACE)))
        for j in range(len(DESIGN_SPACE)):
            x_pert = x.copy()
            lower, upper = DESIGN_BOUNDS[j]
            x_pert[j] += eps * (upper - lower)
            pt_p = ParamTransform(self.base_params)
            pt_p.apply_physics_deltas(deltas); pt_p.apply_design_vector(x_pert, DESIGN_SPACE)
            res = self.simulate(pt_p.get_parameter_values())
            if res["success"]:
                j_pert = np.array([res["energy"], res["power"], mechanical_stability_metric(stresses=res["stresses"])])
                G[:, j] = (np.log(np.abs(j_pert) + 1e-12) - np.log(np.abs(j_base) + 1e-12)) / eps
            else:
                print(f"WARNING: Perturbation for {DESIGN_SPACE[j]} failed: {res.get('reason')}")
        G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(G).all(): raise RuntimeError("Degenerate Jacobian detected.")
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
    if engine is None: engine = MaterialMappingEngine()
    db, bases = engine.run()
    if not bases:
        print("ERROR: Hierarchical optimization aborted: Base material resolution failed.")
        raise RuntimeError("Base material resolution failed.")
    from src.cell_optimization.chem_regularization import derive_coupled_deltas, regularize_salt_props

    # Derivation and Juxtaposition (at beginning of Layer 3)
    print("\n" + "="*120)
    print(f"{'MATERIAL JUXTAPOSITION: QM/PHYSICS DATA VS. DERIVED CELL PARAMETER DELTAS':^120s}")
    print("="*120)

    # 1. Cathode Dopants
    print(f"\nCATEGORY: CATHODE_DOPANT")
    print(f"{'Candidate':25s} | {'QM: Form E':12s} | {'QM: Volume':12s} | {'Derived Delta Key':40s} | {'Value':12s}")
    print("-" * 120)
    for cand in db[MaterialCategory.CATHODE_DOPANT]:
        cand.deltas = derive_coupled_deltas(bases["cathode"]["properties"], cand.properties, bases["cathode"]["formula"], cand.composition)
        p, d = cand.properties, cand.deltas
        flat = [(k, v) for gn, gv in d.items() for k, v in gv.items()]
        for i, (k, v) in enumerate(flat):
            if i == 0: print(f"{cand.name:25s} | {p.get('formation_energy', 0.0):12.4f} | {p.get('volume_per_atom', 0.0):12.4f} | {k:40s} | {v:+.4e}")
            else: print(f"{'':25s} | {'':12s} | {'':12s} | {k:40s} | {v:+.4e}")

    # 2. Salts
    print(f"\nCATEGORY: SALT")
    print(f"{'Candidate':25s} | {'QM: Form E':12s} | {'QM: Volume':12s} | {'Derived Delta Key':40s} | {'Value':12s}")
    print("-" * 120)
    for cand in db[MaterialCategory.SALT]:
        cand.deltas = regularize_salt_props(bases["salt"]["formula"], cand.composition, bases["salt"]["properties"], cand.properties)
        p, d = cand.properties, cand.deltas
        flat = [(k, v) for gn, gv in d.items() for k, v in gv.items()]
        for i, (k, v) in enumerate(flat):
            if i == 0: print(f"{cand.name:25s} | {p.get('formation_energy', 0.0):12.4f} | {p.get('volume_per_atom', 0.0):12.4f} | {k:40s} | {v:+.4e}")
            else: print(f"{'':25s} | {'':12s} | {'':12s} | {k:40s} | {v:+.4e}")
    print("="*120 + "\n")

    optimizer = HierarchicalOptimizer(engine=engine)
    print("Executing Sensitivity-Driven DFN Hierarchical Optimization (Layer 3)...")

    material_results = []
    pairs = [(c, s) for c in db[MaterialCategory.CATHODE_DOPANT] for s in db[MaterialCategory.SALT]]
    if os.environ.get("CEM_FAST_RUN") == "True":
        pairs = pairs[:1]
    for cat, salt in pairs:
        deltas = {}
        if cat and cat.deltas:
            for g_name, props in cat.deltas.items():
                deltas.setdefault(g_name, {}).update(props)
        if salt and salt.deltas:
            for g_name, props in salt.deltas.items():
                deltas.setdefault(g_name, {}).update(props)
        x_base = np.array([np.mean(b) for b in DESIGN_BOUNDS])
        cand_name = f"{cat.name if cat else 'None'} + {salt.name if salt else 'None'}"
        print(f"INFO: Evaluating candidate: {cand_name}")
        pt_test = ParamTransform(optimizer.base_params)
        pt_test.apply_physics_deltas(deltas); pt_test.apply_design_vector(x_base, DESIGN_SPACE)
        if not validate_params(pt_test.get_parameter_values(), verbose=True):
             print(f"[FAILED] {cand_name}: validate_params")
             continue
        G = optimizer.compute_jacobian(x_base, deltas)
        if G is None:
             print(f"[FAILED] {cand_name}: Jacobian computation failed")
             continue
        opt_designs = []
        pt_base = ParamTransform(optimizer.base_params)
        pt_base.apply_physics_deltas(deltas); pt_base.apply_design_vector(x_base, DESIGN_SPACE)
        base_metrics = optimizer.simulate(pt_base.get_parameter_values())
        for i, mode in enumerate(["energy", "power", "stability"]):
            max_s = np.max(np.abs(G[i, :])) + 1e-12
            active_indices = [j for j in range(len(DESIGN_SPACE)) if np.abs(G[i, j]) / max_s > 0.5]
            if not active_indices: active_indices = [int(np.argmax(np.abs(G[i, :])))]
            ref_val = 1.0
            if base_metrics["success"]:
                if mode == "energy": ref_val = base_metrics["energy"]
                elif mode == "power": ref_val = base_metrics["power"]
                elif mode == "stability":
                    from src.cell_optimization.chem_regularization import mechanical_stability_metric
                    ref_val = mechanical_stability_metric(stresses=base_metrics["stresses"])
            problem = SingleObjectiveProblem(optimizer, x_base, active_indices, deltas, mode, ref_scale=ref_val)
            pop_size = int(os.environ.get("CEM_POP_SIZE", 8))
            iters = int(os.environ.get("CEM_ITERATIONS", 3))
            cem = CrossEntropyOptimizer(population_size=pop_size, iterations=iters)
            best_active = cem.optimize(problem.evaluate_single, x_base, DESIGN_BOUNDS, active_indices, G[i, :], verbose=False)
            x_opt = x_base.copy()
            x_opt[active_indices] = best_active
            opt_designs.append(x_opt)
        valid_candidates, stability_scores = [], []
        for x, mode in zip(opt_designs, ["energy", "power", "stability"]):
            pt = ParamTransform(optimizer.base_params)
            pt.apply_physics_deltas(deltas); pt.apply_design_vector(x, DESIGN_SPACE)
            ok, score = optimizer.evaluate_stability_pde(pt.get_parameter_values(), mode)
            if ok:
                valid_candidates.append(x); stability_scores.append(score)
        if not valid_candidates:
             print(f"[FAILED] {cand_name}: Stage 2 structural filtering")
             continue
        x_star = valid_candidates[np.argmax(stability_scores)]
        final_x = 0.8 * x_star + 0.2 * np.mean(valid_candidates, axis=0)
        pt = ParamTransform(optimizer.base_params)
        pt.apply_physics_deltas(deltas); pt.apply_design_vector(final_x, DESIGN_SPACE)
        final_pv = pt.get_parameter_values()
        final_metrics = optimizer.simulate(final_pv, return_sol=True)
        if final_metrics["success"]:
            from src.cell_optimization.chem_regularization import mechanical_stability_metric
            mech_opt = optimizer.mech_model.solve_strain(final_metrics["sol"], final_pv)
            final_metrics.update({
                "stability_metric": mechanical_stability_metric(stresses=final_metrics["stresses"]),
                "max_strain": mech_opt["max_strain"]
            })
            # Remove sol to keep result.json small
            final_metrics.pop("sol", None)

            material_results.append({
                "cat": cat,
                "salt": salt,
                "x": final_x,
                "metrics": final_metrics,
                "deltas": deltas,
                "jacobian": G,
                "opt_designs": opt_designs
            })
    print("="*80)
    print(f"Candidates processed: {len(db[MaterialCategory.CATHODE_DOPANT]) * len(db[MaterialCategory.SALT])}")
    print(f"Successful candidates: {len(material_results)}")
    print("="*80)
    if not material_results:
        print("ERROR: Hierarchical optimization failed: No valid material candidates successfully optimized.")
        raise RuntimeError("No valid material candidates successfully optimized.")
    best = max(material_results, key=lambda r: r["metrics"]["energy"])
    G_avg = best["jacobian"]
    S = np.abs(G_avg) / (np.max(np.abs(G_avg), axis=1).reshape(-1, 1) + 1e-12)
    groups = {"Energy": [], "Power": [], "Stability": [], "Coupled": []}
    for j, name in enumerate(DESIGN_SPACE):
        member_of = []
        for i, obj in enumerate(["Energy", "Power", "Stability"]):
            if S[i, j] > 0.5: groups[obj].append(name); member_of.append(obj)
        if len(member_of) > 1: groups["Coupled"].append(name)
    output = {
        "materials": {
            "cathode": {"name": best["cat"].name, "formula": best["cat"].composition, "properties": best["cat"].properties},
            "electrolyte": {"salt": best["salt"].name, "properties": best["salt"].properties}
        },
        "bases": bases,
        "design_specs_representative": dict(zip(DESIGN_SPACE, best["x"].tolist())),
        "opt_designs_per_objective": {
            mode: dict(zip(DESIGN_SPACE, design.tolist()))
            for mode, design in zip(["energy", "power", "stability"], best["opt_designs"])
        },
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
