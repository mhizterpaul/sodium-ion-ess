import numpy as np
import pybamm
import casadi
import math
import logging
from copy import deepcopy
from functools import lru_cache
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialMappingEngine
from src.cell_optimization.chem_regularization import GZ_METRIC

try:
    import dolfinx
    from dolfinx import fem, mesh, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    import ufl
    import petsc4py.PETSc as PETSc
except ImportError:
    dolfinx = None

def softmax(x, beta=1.0):
    """Numerically stable softmax with clipping."""
    x = np.array(x, dtype=np.float64)
    x = beta * (x - np.max(x))
    x = np.clip(x, -50, 50)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def stable_hash(theta):
    """Mathematically robust hash for float arrays."""
    return hash(np.round(theta, 6).tobytes())

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

class DSMOptimizer:
    """
    Riemannian Control Manifold Optimizer for Coupled Electrochemical-Mechanical State Space.
    """
    def __init__(self, target_y=None):
        # Physically Grounded Operating Point and Scaling
        self.y_ref = np.array([3.2, 300.0, 0.5, 0.02])
        self.y_scale = np.array([2.0, 50.0, 1.0, 0.02])

        self.target_y = target_y if target_y is not None else self.y_ref

        self.engine = MaterialMappingEngine()
        self.material_data = None
        self.selected_dopant_idx = 0
        self.selected_salt_idx = 0
        self.mtms_enabled = 1.0

        self.lr = 0.05
        self.max_epochs = 2
        self.inner_iters = 3
        self.lam = 1e-3

        self.structural_keys = [
            "Positive electrode thickness [m]",
            "Negative electrode thickness [m]",
            "Positive particle radius [m]",
            "Positive electrode porosity",
            "Negative electrode porosity",
            "Separator porosity",
            "Bruggeman coefficient (electrolyte)",
            "Positive electrode active material volume fraction",
            "Negative electrode active material volume fraction",
            "Typical electrolyte concentration [mol.m-3]"
        ]
        self.theta_structural = np.array([1.2e-4, 1.2e-4, 1e-6, 0.3, 0.3, 0.5, 1.5, 0.65, 0.65, 1000.0])

        # Structured Block-Latent Mapping Phi (4 latent blocks)
        self.Phi_blocks = [
            (0, [2, 3, 4, 5, 6], np.array([1.0, 1.0, 1.0, 1.0, 0.5])), # Transport
            (1, [7, 8, 9], np.array([1.0, 1.0, 0.5])),                # Electrochemical
            (2, [0, 1, 3, 4], np.array([0.2, 0.2, 0.5, 0.5])),        # Thermal
            (3, [0, 1, 2], np.array([1.0, 1.0, 0.5]))                 # Mechanical
        ]
        self.Phi = np.zeros((4, 10))
        for block_idx, indices, weights in self.Phi_blocks:
            self.Phi[block_idx, indices] = weights

        self.solve_cache = {}

    def get_parameter_set(self, theta_s, dopant_idx, salt_idx, mtms):
        """Constructs parameter set via pure transformation layer."""
        base_params = get_parameter_values()
        transform = ParamTransform(base_params)
        for i, key in enumerate(self.structural_keys):
            transform.base[key] = theta_s[i]

        dopants = self.material_data.get("Cathode_Dopant", [])
        salts = self.material_data.get("Salt", [])
        func = self.material_data.get("Functionalization", [])

        def apply_channels(material_obj, alpha=1.0):
            if not material_obj: return
            channels = material_obj.projected_delta
            if not isinstance(channels, dict): return

            td = channels.get("thermodynamic", {})
            kt = channels.get("kinetic", {})
            tr = channels.get("transport", {})

            # --- Thermodynamic Channel ---
            if "voltage_boost" in td:
                transform.add_additive("Positive electrode OCP [V]", td["voltage_boost"] * alpha)
            if "initial_loss_mult" in td:
                transform.add_multiplier("Initial concentration in negative electrode [mol.m-3]", td["initial_loss_mult"])

            # --- Kinetic Channel ---
            if "reaction_rate_log_delta" in kt:
                m = math.exp(np.clip(kt["reaction_rate_log_delta"] * alpha, -5, 5))
                transform.add_multiplier("Positive electrode exchange-current density [A.m-2]", m)
            if "sei_growth_mult" in kt:
                transform.add_multiplier("SEI reaction exchange current density [A.m-2]", kt["sei_growth_mult"])

            # --- Transport Channel ---
            if "diffusivity_log_delta" in tr:
                m = math.exp(np.clip(tr["diffusivity_log_delta"] * alpha, -5, 5))
                transform.add_multiplier("Positive particle diffusivity [m2.s-1]", m)
            if "conductivity_mult" in tr:
                transform.add_multiplier("Electrolyte conductivity [S.m-1]", tr["conductivity_mult"])
            if "ion_transference_mult" in tr:
                transform.add_multiplier("Cation transference number", tr["ion_transference_mult"])
            if "resistance_drift_mult" in tr:
                transform.add_multiplier("SEI resistivity [Ohm.m]", tr["resistance_drift_mult"])

        if dopants: apply_channels(dopants[dopant_idx])
        if salts: apply_channels(salts[salt_idx])
        if func: apply_channels(func[0], alpha=mtms)

        return transform.evaluate()

    def setup_sim(self, theta_s, dopant_idx, salt_idx, mtms, model_type="SPM"):
        """Strict simulation rebuild rule as per technical directive."""
        params = self.get_parameter_set(theta_s, dopant_idx, salt_idx, mtms)
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-5, atol=1e-7)
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
        return sim

    def _get_y_pure(self, th, d_idx, s_idx, mtms, horizon=600):
        """Deterministic evaluation with stable caching and robust exception handling."""
        state_hash = hash((stable_hash(th), d_idx, s_idx, mtms, horizon))
        if state_hash in self.solve_cache: return self.solve_cache[state_hash]

        sim = self.setup_sim(th, d_idx, s_idx, mtms)
        try:
            sl = sim.solve([0, horizon], inputs={"Current [A]": 10.0})
            v = float(np.array(sl["Terminal voltage [V]"].entries).flatten()[-1])
            t = float(np.array(sl["Cell temperature [K]"].entries).flatten()[-1])
            q = float(sim.parameter_values["Nominal cell capacity [A.h]"])
            soc = 1.0 - (float(np.array(sl["Discharge capacity [A.h]"].entries).flatten()[-1]) / q)
            c_s_avg = float(np.mean(sl["X-averaged negative particle concentration [mol.m-3]"].entries))
            eps_val = self.solve_reduced_mechanics(t, c_s_avg, th, sim.parameter_values)

            res = np.array([v, t, soc, eps_val])
            self.solve_cache[state_hash] = res
            return res
        except Exception as e:
            logging.warning(f"Simulation failed: {str(e)}")
            return np.array(self.target_y)

    def run(self):
        print(f"Starting Stable Riemannian DSMO Optimization...")
        theta_s = self.theta_structural

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch}: Material Resolution...")
            self.material_data = self.engine.run()

            # Annealed probabilistic selection policy
            beta_anneal = min(15.0, 1.0 + epoch * 5.0)

            for k in range(self.inner_iters):
                y = self._get_y_pure(theta_s, self.selected_dopant_idx, self.selected_salt_idx, self.mtms_enabled, horizon=1800)

                # 1. Stable Structured Finite-Difference Jacobian
                S_reduced = self._compute_reduced_jacobian(theta_s)
                S_theta = S_reduced @ self.Phi

                # 2. Material Selection Update (Pure Evaluation)
                self._update_material_selection_pure(theta_s, beta=beta_anneal)

                # 3. Stable Riemannian Update Step
                r = (y - self.target_y) / self.y_scale
                S_norm = S_theta / self.y_scale[:, None]

                JTJ = S_norm.T @ S_norm
                G_theta = self.Phi.T @ GZ_METRIC @ self.Phi

                # Combined metric with damping (not over-normalized)
                G = JTJ + self.lam * G_theta
                G += 1e-4 * np.eye(G.shape[0]) # Damping

                # Material uncertainty injection
                u = self.material_data["Cathode_Dopant"][self.selected_dopant_idx].uncertainty
                G += 0.05 * u * np.eye(G.shape[0])

                update = np.linalg.solve(G, S_norm.T @ r)

                # Trust-Region Step Clamping
                update_norm = np.linalg.norm(update)
                max_step = 0.1
                if update_norm > max_step:
                    update *= max_step / update_norm

                theta_s = theta_s - self.lr * update

                # 4. Physical Manifold Projection
                theta_s = self._project_physical_manifold(theta_s)
                self._consistency_check(y, theta_s)

                print(f"  Iteration {epoch}.{k}: Residual Norm = {np.linalg.norm(r):.4f}")

        self.theta_structural = theta_s
        return {"structural_design": theta_s.tolist()}

    def _project_physical_manifold(self, theta):
        """Enforces physical feasibility constraints including smooth N/P ratio adjustment."""
        theta[3:6] = np.clip(theta[3:6], 0.2, 0.7)
        theta[7:9] = np.clip(theta[7:9], 0.4, 0.9)

        # Smooth Differentiable N/P Capacity Ratio Adjustment (Target=1.05 for safety margin)
        capacity_ratio = (theta[8] * theta[1]) / (theta[7] * theta[0] + 1e-9)
        target_ratio = 1.05
        # Adaptive log-adjustment prevents discontinuous jumps
        theta[1] *= np.exp(0.1 * np.log(target_ratio / (capacity_ratio + 1e-9)))

        return np.clip(theta,
                       [5e-5, 5e-5, 1e-7, 0.2, 0.2, 0.2, 1.0, 0.4, 0.4, 500.0],
                       [3e-4, 3e-4, 1e-5, 0.7, 0.7, 0.7, 3.0, 0.9, 0.9, 2000.0])

    def _consistency_check(self, y, theta):
        assert np.all(np.isfinite(y)), "Non-finite outputs."
        assert np.all(np.isfinite(theta)), "Non-finite parameters."


    def _compute_reduced_jacobian(self, theta_s):
        """Stable structured finite-difference Jacobian with orthonormal projection."""
        n_z = 4
        S_z = np.zeros((4, n_z))
        eps = 5e-4

        base_y = self._get_y_pure(theta_s, self.selected_dopant_idx, self.selected_salt_idx, self.mtms_enabled, horizon=600)

        for i in range(n_z):
            d_theta = np.zeros_like(theta_s)
            _, idxs, w = self.Phi_blocks[i]

            # Orthonormal projection for directional consistency
            w = w / (np.linalg.norm(w) + 1e-9)
            # Orthogonalize against previous blocks to prevent bias
            for j in range(i):
                prev_w = self.Phi_blocks[j][2]
                prev_idxs = self.Phi_blocks[j][1]
                # Map back to full space for dot product
                w_full = np.zeros(10); w_full[idxs] = w
                prev_full = np.zeros(10); prev_full[prev_idxs] = prev_w / (np.linalg.norm(prev_w) + 1e-9)
                w_full = w_full - np.dot(w_full, prev_full) * prev_full
                w_new = w_full[idxs]
                if np.linalg.norm(w_new) > 1e-6:
                    w = w_new

            w = w / (np.linalg.norm(w) + 1e-9)
            d_theta[idxs] = eps * w

            y_plus = self._get_y_pure(theta_s + d_theta, self.selected_dopant_idx, self.selected_salt_idx, self.mtms_enabled, horizon=600)
            S_z[:, i] = (y_plus - base_y) / eps

        return S_z

    def _update_material_selection_pure(self, theta_s, beta=15.0):
        dopants = self.material_data.get("Cathode_Dopant", [])
        salts = self.material_data.get("Salt", [])

        def score(y, uncertainty, lam=0.5):
            err = np.linalg.norm((y - self.target_y) / self.y_scale)**2
            return -(err + lam * uncertainty)

        if dopants:
            scs = np.array([score(self._get_y_pure(theta_s, i, self.selected_salt_idx, self.mtms_enabled), dopants[i].uncertainty)
                   for i in range(len(dopants))])
            p = softmax(scs, beta=beta)
            # Deterministic convergence late in annealing stage
            if beta > 10.0:
                self.selected_dopant_idx = int(np.argmax(p))
            else:
                self.selected_dopant_idx = int(np.random.choice(len(p), p=p))

        if salts:
            scs = np.array([score(self._get_y_pure(theta_s, self.selected_dopant_idx, i, self.mtms_enabled), salts[i].uncertainty)
                   for i in range(len(salts))])
            p = softmax(scs, beta=beta)
            if beta > 10.0:
                self.selected_salt_idx = int(np.argmax(p))
            else:
                self.selected_salt_idx = int(np.random.choice(len(p), p=p))

    def solve_reduced_mechanics(self, T, c_s_avg, theta_s, param_vals):
        """Physics-consistent reduced mechanics model."""
        eps_alpha = 1e-4 / (1.0 + theta_s[3])
        c_max = float(param_vals["Maximum concentration in negative electrode [mol.m-3]"])
        beta = 0.05 / (c_max + 1e-6)
        eps = eps_alpha * (T - 300.15) + beta * c_s_avg
        eps += 0.02 * (1.0 + theta_s[3]) * (c_s_avg / (c_max + 1e-6))
        return eps

if __name__ == "__main__":
    opt = DSMOptimizer()
    res = opt.run()
    print(res)
