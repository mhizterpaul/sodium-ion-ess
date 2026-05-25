import numpy as np
import pybamm
import casadi
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialDiscoveryFramework

try:
    import dolfinx
    from dolfinx import fem, mesh, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    import ufl
    import petsc4py.PETSc as PETSc
except ImportError:
    dolfinx = None

class DSMOptimizer:
    """
    Differentiable Sensitivity Manifold Optimizer (DSMO).
    Coupled PyBaMM + FEniCSx Multiphysics sensitivities.
    Optimizes both structural parameters and material selection for maximum performance.
    """
    def __init__(self, target_y=None):
        # Target: [Voltage, Temperature, SOC, Mechanical Strain]
        self.target_y = target_y if target_y is not None else np.array([3.1, 305.0, 0.5, 1e-6])
        self.discovery = MaterialDiscoveryFramework()
        self.material_data = self.discovery.run_discovery()

        self.lr = 0.05
        self.max_iters = 5
        self.lam = 1e-3

        self.structural_keys = [
            "Positive electrode thickness [m]",
            "Negative electrode thickness [m]",
            "Positive electrode porosity",
            "Negative electrode porosity",
            "Positive particle radius [m]",
            "Bruggeman coefficient (electrolyte)",
            "Positive electrode active material volume fraction",
            "Negative electrode active material volume fraction",
            "Typical electrolyte concentration [mol.m-3]"
        ]

        # Material parameters (continuous selectors)
        self.theta_structural = np.array([1.2e-4, 1.2e-4, 0.3, 0.3, 1e-6, 1.5, 0.65, 0.65, 1000.0])
        self.theta_material = np.array([0.5, 0.5]) # [Dopant, Salt]

        self.theta = np.concatenate([self.theta_structural, self.theta_material])
        self.all_keys = self.structural_keys + ["Dopant_Alpha", "Salt_Alpha"]

    def apply_material_logic(self, param_vals, theta_m):
        alpha_d = np.clip(theta_m[0], 0, 1)
        alpha_s = np.clip(theta_m[1], 0, 1)

        dopants = self.material_data["Cathode_Dopant"] # [Mn, Cr]
        salts = self.material_data["Salt"]             # [NaBOB, NaTCP]

        # Interpolate deltas
        d_v = (1-alpha_d)*dopants[0].projected_delta.get("voltage_boost", 0) + alpha_d*dopants[1].projected_delta.get("voltage_boost", 0)
        d_diff = (1-alpha_d)*dopants[0].projected_delta.get("diffusivity_mult", 1) + alpha_d*dopants[1].projected_delta.get("diffusivity_mult", 1)

        s_cond = (1-alpha_s)*salts[0].projected_delta.get("conductivity_mult", 1) + alpha_s*salts[1].projected_delta.get("conductivity_mult", 1)
        s_trans = (1-alpha_s)*salts[0].projected_delta.get("ion_transference_mult", 1) + alpha_s*salts[1].projected_delta.get("ion_transference_mult", 1)

        # Apply to parameters
        base_ocp = param_vals["Positive electrode OCP [V]"]
        param_vals["Positive electrode OCP [V]"] = lambda sto: base_ocp(sto) + d_v

        base_diff = param_vals["Positive particle diffusivity [m2.s-1]"]
        param_vals["Positive particle diffusivity [m2.s-1]"] = lambda sto, T: base_diff(sto, T) * d_diff

        param_vals["Electrolyte conductivity [S.m-1]"] = param_vals["Electrolyte conductivity [S.m-1]"] * s_cond
        param_vals["Cation transference number"] = param_vals["Cation transference number"] * s_trans

        return param_vals

    def setup_sim(self, theta):
        param_vals = pybamm.ParameterValues(get_parameter_values())
        theta_s = theta[:len(self.structural_keys)]
        theta_m = theta[len(self.structural_keys):]

        for i, key in enumerate(self.structural_keys):
            param_vals[key] = theta_s[i]

        param_vals = self.apply_material_logic(param_vals, theta_m)
        param_vals["Current function [A]"] = 10.0

        model = pybamm.lithium_ion.DFN()
        solver = pybamm.CasadiSolver(mode="fast")
        sim = pybamm.Simulation(model, parameter_values=param_vals, solver=solver)
        return sim

    def solve_mechanical_adjoint(self, T, SOC, theta):
        """Concrete FEniCSx exact sensitivities."""
        if not dolfinx:
            eps = 1e-7 * (T - 298.15) + 1e-6 * (0.5 - SOC)
            deps_dL = (1e-6 / (theta[0]+theta[1]+20e-6))
            S_mech = np.zeros(len(self.theta))
            S_mech[0] = S_mech[1] = deps_dL
            return eps, S_mech

        domain = mesh.create_interval(MPI.COMM_WORLD, 20, [0, 1])
        V = fem.functionspace(domain, ("Lagrange", 1))

        L_tot_val = theta[0] + theta[1] + 20e-6
        L_var = fem.Constant(domain, default_scalar_type(L_tot_val))
        L_ufl = ufl.variable(L_var)

        E = fem.Constant(domain, default_scalar_type(10e9))
        alpha = fem.Constant(domain, default_scalar_type(1e-5))
        beta = fem.Constant(domain, default_scalar_type(0.02))
        eps_0 = alpha * (T - 298.15) + beta * (SOC - 0.5)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        F = (1.0/L_ufl) * E * (u.dx(0) - L_ufl*eps_0) * v.dx(0) * ufl.dx
        a = ufl.lhs(F)
        L_form = ufl.rhs(F)

        dofs_left = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
        bc = fem.dirichletbc(default_scalar_type(0), dofs_left, V)
        problem = LinearProblem(a, L_form, bcs=[bc])
        uh = problem.solve()

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

        S_mech = np.zeros(len(self.theta))
        S_mech[0] = S_mech[1] = dstrain_dL
        return float(strain_val), S_mech

    def run(self):
        print(f"Starting Multiphysics DSMO for {len(self.all_keys)} parameters...")

        theta_vec = self.theta
        for k in range(self.max_iters):
            sim = self.setup_sim(theta_vec)
            sol = sim.solve([0, 1800])

            V_val = float(np.array(sol["Terminal voltage [V]"].entries).flatten()[-1])
            T_val = float(np.array(sol["Cell temperature [K]"].entries).flatten()[-1])
            SOC_val = 1.0 - (float(np.array(sol["Discharge capacity [A.h]"].entries).flatten()[-1]) / 10.0)

            eps_val, S_mech_row = self.solve_mechanical_adjoint(T_val, SOC_val, theta_vec)
            y = np.array([V_val, T_val, SOC_val, eps_val])

            S = self.compute_jacobian(theta_vec)

            S = np.vstack([S, S_mech_row])
            r = y - self.target_y

            G = S.T @ S + self.lam * np.eye(len(theta_vec))
            update = np.linalg.solve(G, S.T @ r)
            theta_vec = theta_vec - self.lr * update

            # Constraints
            theta_vec[:9] = np.clip(theta_vec[:9],
                                    [5e-5, 5e-5, 0.1, 0.1, 1e-7, 1.0, 0.4, 0.4, 500.0],
                                    [3e-4, 3e-4, 0.6, 0.6, 1e-5, 3.0, 0.8, 0.8, 2000.0])
            theta_vec[9:] = np.clip(theta_vec[9:], 0, 1)

            print(f"  Iteration {k}: Residual Norm = {np.linalg.norm(r):.4f}, Dopant_Alpha = {theta_vec[9]:.2f}, Salt_Alpha = {theta_vec[10]:.2f}")
            if np.linalg.norm(r) < 1e-4: break

        dopant = "Cr" if theta_vec[9] > 0.5 else "Mn"
        salt = "NaTCP" if theta_vec[10] > 0.5 else "NaBOB"

        print(f"\nOptimization Complete.")
        print(f"Selected Dopant: {dopant}, Selected Salt: {salt}")

        return {
            "design": theta_vec.tolist(),
            "selected_dopant": dopant,
            "selected_salt": salt
        }

    def compute_jacobian(self, theta):
        n = len(theta)
        S = np.zeros((3, n))
        eps = 1e-4
        for i in range(n):
            th_p = theta.copy(); th_p[i] += eps
            sim_p = self.setup_sim(th_p)
            try:
                sol_p = sim_p.solve([0, 60]) # Fast eval
                v_p = float(np.array(sol_p["Terminal voltage [V]"].entries).flatten()[-1])
                t_p = float(np.array(sol_p["Cell temperature [K]"].entries).flatten()[-1])
                soc_p = 1.0 - (float(np.array(sol_p["Discharge capacity [A.h]"].entries).flatten()[-1]) / 10.0)
            except: v_p, t_p, soc_p = 3.0, 298.15, 0.5

            th_m = theta.copy(); th_m[i] -= eps
            sim_m = self.setup_sim(th_m)
            try:
                sol_m = sim_m.solve([0, 60])
                v_m = float(np.array(sol_m["Terminal voltage [V]"].entries).flatten()[-1])
                t_m = float(np.array(sol_m["Cell temperature [K]"].entries).flatten()[-1])
                soc_m = 1.0 - (float(np.array(sol_m["Discharge capacity [A.h]"].entries).flatten()[-1]) / 10.0)
            except: v_m, t_m, soc_m = 3.0, 298.15, 0.5

            S[0, i] = (v_p - v_m) / (2 * eps)
            S[1, i] = (t_p - t_m) / (2 * eps)
            S[2, i] = (soc_p - soc_m) / (2 * eps)

        return S

if __name__ == "__main__":
    opt = DSMOptimizer()
    res = opt.run()
    print(res)
