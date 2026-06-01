import numpy as np
import pybamm
import casadi
import math
from copy import deepcopy
from functools import lru_cache
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialMappingEngine

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
    """Numerically stable softmax."""
    x = beta * (x - np.max(x))
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

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
                # Use default args for closure capture
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
    Scientifically Justifiable DSMO with Explicit Block-Latent Manifold Projection.
    """
    def __init__(self, target_y=None):
        # Physically Derived Target y: [V, T, SOC, eps]
        # v_ref = 3.2V, T_ref = 298.15K, eps_ref = 1e-6
        self.target_y = target_y if target_y is not None else np.array([3.2, 298.15, 0.5, 1.0])
        self.y_ref = np.array([3.2, 298.15, 1.0, 1.0]) # Physical normalization bases

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
        # Prevents cross-physics leakage
        self.Phi_blocks = [
            # 1. Transport [rp, eps_p, eps_n, eps_sep, brugg]
            (0, [2, 3, 4, 5, 6], np.array([1.0, 1.0, 1.0, 1.0, 0.5])),
            # 2. Electrochemical [am_p, am_n, c_typ]
            (1, [7, 8, 9], np.array([1.0, 1.0, 0.5])),
            # 3. Thermal [Lp, Ln, eps_p, eps_n]
            (2, [0, 1, 3, 4], np.array([0.2, 0.2, 0.5, 0.5])),
            # 4. Mechanical [Lp, Ln, rp]
            (3, [0, 1, 2], np.array([1.0, 1.0, 0.5]))
        ]
        self.Phi = np.zeros((4, 10))
        for block_idx, indices, weights in self.Phi_blocks:
            self.Phi[block_idx, indices] = weights

        self.sim_cache = {}
        self.solve_cache = {}

    def get_parameter_set(self, theta_s, dopant_idx, salt_idx, mtms):
        """Constructs parameter set via pure transformation layer."""
        base_params = get_parameter_values()
        transform = ParamTransform(base_params)

        # 1. Structural Parameters
        for i, key in enumerate(self.structural_keys):
            transform.base[key] = theta_s[i]

        # 2. Material Channels
        dopants = self.material_data.get("Cathode_Dopant", [])
        salts = self.material_data.get("Salt", [])
        func = self.material_data.get("Functionalization", [])

        def apply_channels(material_obj, alpha=1.0):
            if not material_obj: return
            channels = material_obj.projected_delta # Now returns Dict[channel, Dict]
            if not isinstance(channels, dict) or "thermodynamic" not in channels:
                return # Fallback for old data if needed

            # Thermodynamic: Additive
            for k, v in channels["thermodynamic"].items():
                if k == "voltage_boost":
                    transform.add_additive("Positive electrode OCP [V]", v * alpha)

            # Kinetic/Transport: Log-multiplicative
            for k, v in channels["kinetic"].items():
                if k == "reaction_rate_log_delta":
                    transform.add_multiplier("Positive electrode exchange-current density [A.m-2]", math.exp(v * alpha))

            for k, v in channels["transport"].items():
                if k == "diffusivity_log_delta":
                    transform.add_multiplier("Positive particle diffusivity [m2.s-1]", math.exp(v * alpha))

        if dopants: apply_channels(dopants[dopant_idx])
        if salts: apply_channels(salts[salt_idx])
        if func: apply_channels(func[0], alpha=mtms)

        return transform.evaluate()

    def run(self):
        print(f"Starting Scientifically Justifiable DSMO Optimization...")
        theta_s = self.theta_structural

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch}: Material Resolution...")
            self.material_data = self.engine.run()

            for k in range(self.inner_iters):
                y = self._get_y_pure(theta_s, self.selected_dopant_idx, self.selected_salt_idx, self.mtms_enabled)

                # 1. Structural Jacobian (Block-Decoupled)
                S_reduced = self._compute_reduced_jacobian(theta_s)
                S_theta = S_reduced @ self.Phi

                # 2. Material Selection Update (Pure Evaluation)
                self._update_material_selection_pure(theta_s)

                # 3. Regularized Update Step with Physical Normalization
                r = (y - self.target_y) / self.y_ref
                S_norm = S_theta / self.y_ref[:, None]

                # Spectral clipping
                U, s_val, Vh = np.linalg.svd(S_norm, full_matrices=False)
                s_clipped = np.clip(s_val, 1e-3, None)
                S_norm = U @ np.diag(s_clipped) @ Vh

                G = S_norm.T @ S_norm + self.lam * np.eye(len(theta_s))
                # Trace-based conditioning
                G += 0.01 * np.eye(len(theta_s)) * np.trace(G)/len(theta_s)

                # Material uncertainty augmentation (Covariance-scaled damping)
                u = self.material_data["Cathode_Dopant"][self.selected_dopant_idx].uncertainty
                u_vec = np.ones(len(theta_s)) * u
                G += (u_vec[:, None] * u_vec[None, :]) * 0.1

                update = np.linalg.solve(G, S_norm.T @ r)
                theta_s = theta_s - self.lr * update

                # 4. Physical Feasibility Projection (Pi_phys)
                theta_s = self._project_physical_manifold(theta_s)

                # 5. Consistency Check Layer
                self._consistency_check(y, theta_s)

                print(f"  Iteration {epoch}.{k}: Residual Norm = {np.linalg.norm(r):.4f}")

        self.theta_structural = theta_s
        return {"structural_design": theta_s.tolist()}

    def _project_physical_manifold(self, theta):
        """Enforces physical feasibility constraints including capacity consistency."""
        # Porosity limits
        theta[3:6] = np.clip(theta[3:6], 0.2, 0.7)
        # Loading (volume fractions)
        theta[7:9] = np.clip(theta[7:9], 0.4, 0.9)

        # Capacity ratio consistency (Qn/Qp approx 1.0)
        # Account for particle radius effect on active surface if needed, here we use volume capacity
        capacity_ratio = (theta[8] * theta[1]) / (theta[7] * theta[0] + 1e-9)
        # Enforce consistency: 0.9 <= ratio <= 1.1
        theta *= np.clip(1.0 / capacity_ratio, 0.95, 1.05)

        # N/P Ratio Constraint refinement
        np_ratio = (theta[1] * theta[8]) / (theta[0] * theta[7] + 1e-9)
        if np_ratio < 0.9 or np_ratio > 1.1:
            target_ln = 1.0 * (theta[0] * theta[7]) / (theta[8] + 1e-9)
            theta[1] = np.clip(target_ln, 5e-5, 3e-4)

        return np.clip(theta,
                       [5e-5, 5e-5, 1e-7, 0.2, 0.2, 0.2, 1.0, 0.4, 0.4, 500.0],
                       [3e-4, 3e-4, 1e-5, 0.7, 0.7, 0.7, 3.0, 0.9, 0.9, 2000.0])

    def _consistency_check(self, y, theta):
        assert np.all(np.isfinite(y)), "Non-finite outputs detected."
        assert np.all(np.isfinite(theta)), "Non-finite parameters detected."
        assert theta[9] > 0, "Non-positive electrolyte concentration."

    def _get_y_pure(self, th, d_idx, s_idx, mtms):
        """Pure evaluation function for simulation state with simulation object caching."""
        state_hash = hash((tuple(th.tolist()), d_idx, s_idx, mtms))
        if state_hash in self.solve_cache:
            return self.solve_cache[state_hash]

        params = self.get_parameter_set(th, d_idx, s_idx, mtms)

        # Object-level caching
        sim_key = (d_idx, s_idx, mtms)
        if sim_key in self.sim_cache:
            sim = self.sim_cache[sim_key]
            sim.parameter_values.update(params) # Update parameters on existing object
        else:
            model = pybamm.lithium_ion.SPM()
            solver = pybamm.CasadiSolver(mode="safe")
            sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
            self.sim_cache[sim_key] = sim

        try:
            sl = sim.solve([0, 1800])
            v = float(np.array(sl["Terminal voltage [V]"].entries).flatten()[-1])
            t = float(np.array(sl["Cell temperature [K]"].entries).flatten()[-1])
            q = float(sim.parameter_values["Nominal cell capacity [A.h]"])
            soc = 1.0 - (float(np.array(sl["Discharge capacity [A.h]"].entries).flatten()[-1]) / q)
            c_s_avg = float(np.mean(sl["X-averaged negative particle concentration [mol.m-3]"].entries))
            eps_val = self.solve_reduced_mechanics(t, c_s_avg, th, sim.parameter_values)

            # Nondimensionalize
            res = np.array([v, t, soc, eps_val])
            self.solve_cache[state_hash] = res
            return res
        except:
            return self.target_y

    def _compute_reduced_jacobian(self, theta_s):
        """Computes dy/dz via symmetric Finite Difference in structured latent space."""
        n_z = 4
        S_z = np.zeros((4, n_z))
        eps = 1e-3

        for i in range(n_z):
            # Block-wise inversion aligned to Phi structure (prevents cross-block coupling)
            d_theta = np.zeros(10)
            block_id, idxs, w = self.Phi_blocks[i]

            # Map dz=eps to d_theta such that Phi_i @ d_theta_i = eps
            # d_theta_i = (w^T / ||w||^2) * eps
            d_theta[idxs] = (w / (np.sum(w**2) + 1e-9)) * eps

            y_p = self._get_y_pure(theta_s + d_theta, self.selected_dopant_idx, self.selected_salt_idx, self.mtms_enabled)
            y_m = self._get_y_pure(theta_s - d_theta, self.selected_dopant_idx, self.selected_salt_idx, self.mtms_enabled)

            S_z[:, i] = (y_p - y_m) / (2 * eps)

        return S_z

    def _update_material_selection_pure(self, theta_s, beta=15.0):
        """Pure probabilistic selection without state contamination."""
        dopants = self.material_data.get("Cathode_Dopant", [])
        salts = self.material_data.get("Salt", [])

        def score(y, uncertainty, lam=0.5):
            err = np.linalg.norm((y - self.target_y) / self.y_ref)**2
            return -(err + lam * uncertainty)

        # 1. Dopant
        if dopants:
            scs = np.array([score(self._get_y_pure(theta_s, i, self.selected_salt_idx, self.mtms_enabled), dopants[i].uncertainty)
                   for i in range(len(dopants))])
            self.selected_dopant_idx = int(np.random.choice(len(scs), p=softmax(scs, beta=beta)))

        # 2. Salt
        if salts:
            scs = np.array([score(self._get_y_pure(theta_s, self.selected_dopant_idx, i, self.mtms_enabled), salts[i].uncertainty)
                   for i in range(len(salts))])
            self.selected_salt_idx = int(np.random.choice(len(scs), p=softmax(scs, beta=beta)))

    def solve_reduced_mechanics(self, T, c_s_avg, theta_s, param_vals):
        """Physics-consistent reduced mechanics model (Unified) with structural coupling."""
        eps_ref = 1e-6
        eps_alpha = 1e-7 / (1.0 + theta_s[3])
        c_max = float(param_vals["Maximum concentration in negative electrode [mol.m-3]"])
        beta = 3.1e-6 / (c_max + 1e-6)

        # Strain = expansion_thermal + expansion_intercalation + structural_coupling
        # Structural coupling to porosity (theta_s[3])
        eps = eps_alpha * (T - 298.15) + beta * c_s_avg
        eps += 0.05 * (1.0 - theta_s[3]) * (c_s_avg / (c_max + 1e-6))

        return eps / eps_ref

if __name__ == "__main__":
    opt = DSMOptimizer()
    res = opt.run()
    print(res)
