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
    Optimizes both structural parameters and material selection for maximum performance.
    """
    def __init__(self, target_efficiency=0.98):
        self.target_eff = target_efficiency
        self.discovery = MaterialDiscoveryFramework()
        self.material_data = self.discovery.run_discovery()

        self.lr = 0.1
        self.max_iters = 8
        self.lam = 1e-2

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

        # Material parameters:
        # theta_m[0]: Dopant interpolation (0: Mn, 1: Cr)
        # theta_m[1]: Salt interpolation (0: NaBOB, 1: NaTCP)
        self.theta_structural = np.array([1.2e-4, 1.2e-4, 0.3, 0.3, 1e-6, 1.5, 0.65, 0.65, 1000.0])
        self.theta_material = np.array([0.5, 0.5])

        self.theta = np.concatenate([self.theta_structural, self.theta_material])
        self.all_keys = self.structural_keys + ["Dopant_Alpha", "Salt_Alpha"]

    def apply_material_logic(self, param_vals, theta_m):
        alpha_d = np.clip(theta_m[0], 0, 1)
        alpha_s = np.clip(theta_m[1], 0, 1)

        dopants = self.material_data["Cathode_Dopant"] # [Mn, Cr]
        salts = self.material_data["Salt"]             # [NaBOB, NaTCP]

        # Interpolate deltas
        d_v = (1-alpha_d)*dopants[0].projected_delta["voltage_boost"] + alpha_d*dopants[1].projected_delta["voltage_boost"]
        d_diff = (1-alpha_d)*dopants[0].projected_delta["diffusivity_mult"] + alpha_d*dopants[1].projected_delta["diffusivity_mult"]

        s_cond = (1-alpha_s)*salts[0].projected_delta["conductivity_mult"] + alpha_s*salts[1].projected_delta["conductivity_mult"]
        s_trans = (1-alpha_s)*salts[0].projected_delta["ion_transference_mult"] + alpha_s*salts[1].projected_delta["ion_transference_mult"]

        # Apply to parameters
        # Wrap existing functions to include deltas
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

        # Structural
        for i, key in enumerate(self.structural_keys):
            param_vals[key] = theta_s[i]

        # Material
        param_vals = self.apply_material_logic(param_vals, theta_m)

        # Set a fixed current instead of InputParameter
        param_vals["Current function [A]"] = 10.0 # 1C for 10Ah cell

        model = pybamm.lithium_ion.DFN()
        solver = pybamm.CasadiSolver(mode="fast")
        sim = pybamm.Simulation(model, parameter_values=param_vals, solver=solver)
        return sim

    def run(self):
        print(f"Starting Performance-Based DSMO for {len(self.all_keys)} design parameters...")

        theta_vec = self.theta
        for k in range(self.max_iters):
            # Performance proxy: average discharge voltage
            sim = self.setup_sim(theta_vec)
            try:
                sol = sim.solve([0, 300]) # 5 minutes
                v_avg = np.mean(sol["Terminal voltage [V]"].entries)
                eff = v_avg / 3.4
            except Exception as e:
                print(f"Solve failed at iteration {k}: {e}")
                eff = 0.5

            # Sensitivity extraction (Finite Difference for material/structural mix)
            S = self.compute_jacobian(theta_vec)

            r = eff - self.target_eff
            # Manifold update
            J = S.T @ S + self.lam * np.eye(len(theta_vec))
            update = np.linalg.solve(J, S.T * r)

            theta_vec = theta_vec - self.lr * update.flatten()

            # Constraints
            theta_vec[:9] = np.clip(theta_vec[:9],
                                    [5e-5, 5e-5, 0.1, 0.1, 1e-7, 1.0, 0.4, 0.4, 500.0],
                                    [3e-4, 3e-4, 0.6, 0.6, 1e-5, 3.0, 0.8, 0.8, 2000.0])
            theta_vec[9:] = np.clip(theta_vec[9:], 0, 1) # Selector bounds

            print(f"  Iteration {k}: Metric = {eff:.4f}, Dopant_Alpha = {theta_vec[9]:.2f}, Salt_Alpha = {theta_vec[10]:.2f}")
            if abs(r) < 1e-4: break

        # Final Material Selection
        dopant = "Cr" if theta_vec[9] > 0.5 else "Mn"
        salt = "NaTCP" if theta_vec[10] > 0.5 else "NaBOB"

        print(f"\nOptimization Complete.")
        print(f"Selected Dopant: {dopant}")
        print(f"Selected Salt: {salt}")

        return {
            "design": theta_vec.tolist(),
            "selected_dopant": dopant,
            "selected_salt": salt
        }

    def compute_jacobian(self, theta):
        n = len(theta)
        S = np.zeros(n)
        eps = 1e-4

        for i in range(n):
            th_p = theta.copy(); th_p[i] += eps
            sim_p = self.setup_sim(th_p)
            try:
                sol_p = sim_p.solve([0, 60])
                eff_p = np.mean(sol_p["Terminal voltage [V]"].entries) / 3.4
            except: eff_p = 0.5

            th_m = theta.copy(); th_m[i] -= eps
            sim_m = self.setup_sim(th_m)
            try:
                sol_m = sim_m.solve([0, 60])
                eff_m = np.mean(sol_m["Terminal voltage [V]"].entries) / 3.4
            except: eff_m = 0.5

            S[i] = (eff_p - eff_m) / (2 * eps)

        return S.reshape(1, -1)

if __name__ == "__main__":
    opt = DSMOptimizer()
    res = opt.run()
    print(res)
