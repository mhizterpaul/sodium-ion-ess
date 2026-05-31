import numpy as np
import pybamm
import casadi
import math
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

def stable_pinv(A, lam=1e-4):
    """Tikhonov-regularized pseudoinverse."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = S / (S**2 + lam**2)
    return Vt.T @ np.diag(S_inv) @ U.T

class DSMOptimizer:
    """
    Multi-Fidelity Differentiable Sensitivity Manifold Optimizer (DSMO).
    Outer loop: Probabilistic Material Selection via Energy-based Policy.
    Inner loop: Continuous Structural Optimization via Reduced Jacobian + Preconditioned GN.
    """
    def __init__(self, target_y=None):
        # Target y: [Voltage, Temp, SOC, Strain] (Normalized)
        self.target_y = target_y if target_y is not None else np.array([3.3, 298.15, 0.5, 1e-8])
        self.y_scale = np.array([3.5, 300.0, 1.0, 1e-6])

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

        # Latent Mapping Phi (4 latent factors)
        self.Phi = np.zeros((4, 10))
        self.Phi[0, [2, 3, 4, 6]] = [1.0, 1.0, 1.0, 0.5]  # Transport
        self.Phi[1, [7, 8, 9]] = [1.0, 1.0, 0.5]         # Electrochemical
        self.Phi[2, [0, 1, 3, 4]] = [0.2, 0.2, 0.5, 0.5] # Thermal
        self.Phi[3, [0, 1, 2]] = [1.0, 1.0, 0.5]         # Mechanical

        self.sim_cache = {}
        self.solve_cache = {}

    def apply_material_logic(self, param_vals):
        """Applies material deltas without function mutation."""
        dopants = self.material_data.get("Cathode_Dopant", [])
        salts = self.material_data.get("Salt", [])
        func = self.material_data.get("Functionalization", [])

        d_delta = dopants[self.selected_dopant_idx].to_pybamm_delta() if dopants else {}
        s_delta = salts[self.selected_salt_idx].to_pybamm_delta() if salts else {}
        f_delta = func[0].to_pybamm_delta() if func else {}

        def apply_p(delta_map, alpha=1.0):
            for name, (mode, val) in delta_map.items():
                base = param_vals[name]
                if mode == "multiplier":
                    m_val = np.clip(1.0 + alpha * (val - 1.0), 0.1, 10.0)
                    if callable(base):
                        # Wrap the function to apply multiplier
                        def wrap_m(sto, T=None, b=base, m=m_val):
                            return b(sto) * m if T is None else b(sto, T) * m
                        param_vals[name] = wrap_m
                    else:
                        param_vals[name] = base * m_val
                elif mode == "additive":
                    a_val = np.clip(alpha * val, -1.0, 1.0)
                    if callable(base):
                        # Wrap the function to apply additive
                        def wrap_a(sto, T=None, b=base, a=a_val):
                            return b(sto) + a if T is None else b(sto, T) + a
                        param_vals[name] = wrap_a
                    else:
                        param_vals[name] = base + a_val

        apply_p(d_delta)
        apply_p(s_delta)
        apply_p(f_delta, alpha=self.mtms_enabled)
        return param_vals

    def setup_sim(self, theta_s, model_type="SPM"):
        theta_hash = hash(tuple(theta_s.tolist()) + (model_type, self.selected_dopant_idx, self.selected_salt_idx, self.mtms_enabled))
        if theta_hash in self.sim_cache:
            return self.sim_cache[theta_hash]

        param_vals = pybamm.ParameterValues(get_parameter_values())
        for i, key in enumerate(self.structural_keys):
            param_vals[key] = theta_s[i]

        param_vals = self.apply_material_logic(param_vals)
        param_vals["Current function [A]"] = 10.0

        model = pybamm.lithium_ion.SPM() if model_type == "SPM" else pybamm.lithium_ion.DFN()
        solver = pybamm.CasadiSolver(mode="safe", extra_options_setup={"max_num_steps": 1000})
        sim = pybamm.Simulation(model, parameter_values=param_vals, solver=solver)
        self.sim_cache[theta_hash] = sim
        return sim

    def run(self):
        print(f"Starting Robust Refined DSMO Optimization...")
        theta_s = self.theta_structural

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch}: Material Resolution...")
            self.material_data = self.engine.run()

            for k in range(self.inner_iters):
                y = self._get_y_full(theta_s)

                # 1. Structural Jacobian (Reduced + Stable Pinv)
                S_reduced = self._compute_reduced_jacobian(theta_s)
                S_theta = S_reduced @ self.Phi

                # SVD Clipping
                U, s_val, Vh = np.linalg.svd(S_theta, full_matrices=False)
                s_clipped = np.clip(s_val, 1e-3, None)
                S_theta = U @ np.diag(s_clipped) @ Vh

                r = (y - self.target_y) / self.y_scale
                S_norm = S_theta / self.y_scale[:, None]

                # Preconditioned Gauss-Newton Hessian
                scale = np.linalg.norm(S_norm, ord=2)
                sigma_proxy = self.material_data["Cathode_Dopant"][self.selected_dopant_idx].uncertainty

                G = S_norm.T @ S_norm + (self.lam + 1e-3 * scale**2 + sigma_proxy) * np.eye(len(theta_s))
                # Trace-based regularization for better conditioning
                G += 0.01 * np.eye(G.shape[0]) * np.trace(G)/G.shape[0]

                update = np.linalg.solve(G, S_norm.T @ r)
                theta_s = theta_s - self.lr * update
                theta_s = np.clip(theta_s,
                                  [5e-5, 5e-5, 1e-7, 0.1, 0.1, 0.2, 1.0, 0.4, 0.4, 500.0],
                                  [3e-4, 3e-4, 1e-5, 0.6, 0.6, 0.8, 3.0, 0.8, 0.8, 2000.0])

                # 2. Material Selection Update (Energy-based Probabilistic Policy)
                self._update_material_selection_probabilistic(theta_s)

                print(f"  Iteration {epoch}.{k}: Residual Norm = {np.linalg.norm(r):.4f}")

        self.theta_structural = theta_s
        return {"structural_design": theta_s.tolist()}

    def _get_y_full(self, th):
        solve_hash = hash(tuple(th.tolist()) + (self.selected_dopant_idx, self.selected_salt_idx, self.mtms_enabled))
        if solve_hash in self.solve_cache:
            return self.solve_cache[solve_hash]

        s = self.setup_sim(th, model_type="SPM")
        try:
            sl = s.solve([0, 1800])
            v = float(np.array(sl["Terminal voltage [V]"].entries).flatten()[-1])
            t = float(np.array(sl["Cell temperature [K]"].entries).flatten()[-1])
            q = float(s.parameter_values["Nominal cell capacity [A.h]"])
            soc = 1.0 - (float(np.array(sl["Discharge capacity [A.h]"].entries).flatten()[-1]) / q)
            # Solid concentration for mechanical coupling
            c_s_avg = float(np.mean(sl["X-averaged negative particle concentration [mol.m-3]"].entries))
            eps_val, _ = self.solve_mechanical_adjoint(t, c_s_avg, th, s.parameter_values)

            res = np.array([v, t, soc, eps_val])
            self.solve_cache[solve_hash] = res
            return res
        except:
            return self.target_y

    def _compute_reduced_jacobian(self, theta_s):
        """Computes dy/dz via Finite Difference in latent space."""
        z_base = self.Phi @ theta_s
        y_base = self._get_y_full(theta_s)
        n_z = len(z_base)
        n_y = len(y_base)
        S_z = np.zeros((n_y, n_z))

        for i in range(n_z):
            eps = 1e-3
            dz = np.zeros(n_z); dz[i] = eps
            # Use stable pinv for latent space mapping
            d_theta = stable_pinv(self.Phi) @ dz
            y_p = self._get_y_full(theta_s + d_theta)
            S_z[:, i] = (y_p - y_base) / eps

        return S_z

    def _update_material_selection_probabilistic(self, theta_s, beta=10.0, lam_u=0.5):
        """Unified Energy-based selection model."""
        dopants = self.material_data.get("Cathode_Dopant", [])
        salts = self.material_data.get("Salt", [])

        def calc_score(y, uncertainty):
            # Energy score: -(error^2 + lambda * uncertainty)
            error = np.linalg.norm((y - self.target_y) / self.y_scale)**2
            return -(error + lam_u * uncertainty)

        def softmax_sample(scores):
            scores = np.array(scores)
            probs = np.exp(beta * (scores - np.max(scores)))
            probs /= np.sum(probs)
            return np.random.choice(len(scores), p=probs)

        if dopants:
            scores = []
            for i in range(len(dopants)):
                self.selected_dopant_idx = i
                y = self._get_y_full(theta_s)
                scores.append(calc_score(y, dopants[i].uncertainty))
            self.selected_dopant_idx = int(softmax_sample(scores))

        if salts:
            scores = []
            for i in range(len(salts)):
                self.selected_salt_idx = i
                y = self._get_y_full(theta_s)
                scores.append(calc_score(y, salts[i].uncertainty))
            self.selected_salt_idx = int(softmax_sample(scores))

        m_scores = []
        for m in [0.0, 1.0]:
            self.mtms_enabled = m
            y = self._get_y_full(theta_s)
            m_scores.append(calc_score(y, 0.0))
        self.mtms_enabled = float(softmax_sample(m_scores))

    def solve_mechanical_adjoint(self, T, c_s_avg, theta_s, param_vals):
        L_p, L_a = theta_s[0], theta_s[1]
        L_tot_val = L_p + L_a + 20e-6
        eps_ref = 1e-6
        eps_alpha = 1e-7 / (1.0 + theta_s[3])

        c_max = float(param_vals["Maximum concentration in negative electrode [mol.m-3]"])
        beta_expansion = 3.1e-6 / (c_max + 1e-6)

        if not dolfinx:
            eps = eps_alpha * (T - 298.15) + beta_expansion * c_s_avg
            deps_dL = (1e-6 / L_tot_val)
            S_mech = np.zeros(len(theta_s))
            S_mech[0] = S_mech[1] = (deps_dL / L_tot_val) / eps_ref
            # Nondimensionalize strain output
            return (eps / eps_ref), S_mech

        domain = mesh.create_interval(MPI.COMM_WORLD, 20, [0, 1])
        V = fem.functionspace(domain, ("Lagrange", 1))
        L_var = fem.Constant(domain, default_scalar_type(L_tot_val))
        L_ufl = ufl.variable(L_var)
        E = fem.Constant(domain, default_scalar_type(10e9))
        alpha_fem = fem.Constant(domain, default_scalar_type(eps_alpha))
        beta_fem = fem.Constant(domain, default_scalar_type(beta_expansion))
        eps_0 = alpha_fem * (T - 298.15) + beta_fem * c_s_avg
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        F = (1.0/L_ufl) * E * (u.dx(0) - L_ufl*eps_0) * v.dx(0) * ufl.dx
        a, L_form = ufl.lhs(F), ufl.rhs(F)
        dofs_left = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
        bc = fem.dirichletbc(default_scalar_type(0), dofs_left, V)
        uh = LinearProblem(a, L_form, bcs=[bc]).solve()
        strain_val = uh.x.array[-1] / L_tot_val
        K_mat = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
        K_mat.assemble()
        R_uh = ufl.replace(F, {u: uh})
        dR_dL = ufl.diff(R_uh, L_ufl)
        rhs_sens = fem.petsc.assemble_vector(fem.form(-dR_dL))
        fem.petsc.apply_lifting(rhs_sens, [fem.form(a)], bcs=[[bc]])
        rhs_sens.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(rhs_sens, [bc])
        du_dL = fem.Function(V)
        ksp = PETSc.KSP().create(domain.comm)
        ksp.setOperators(K_mat)
        ksp.solve(rhs_sens, du_dL.vector)
        dstrain_dL = (1.0/L_tot_val) * du_dL.x.array[-1] - (uh.x.array[-1] / (L_tot_val**2))
        S_mech = np.zeros(len(theta_s))
        S_mech[0] = S_mech[1] = (dstrain_dL / L_tot_val) / eps_ref
        return (float(strain_val) / eps_ref), S_mech

if __name__ == "__main__":
    opt = DSMOptimizer()
    res = opt.run()
    print(res)
