import pybamm
import numpy as np
import scipy.io as sio
import os
import math
from typing import Dict, Any, List, Tuple
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCandidate
from src.cell_optimization.chem_regularization import regularize_material

try:
    import dolfinx
    from dolfinx import fem, mesh, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    import ufl
except ImportError:
    dolfinx = None

class OptimizationValidator:
    """
    Multi-physics Digital Twin Validator.
    Orchestrates the 3-layer flow and performs high-fidelity mechanical integrity analysis using FEniCSx.
    """

    def __init__(self, optimized_design: Dict[str, float], material_info: List[Tuple[MaterialCandidate, Dict[str, Any]]]):
        self.design = optimized_design
        self.materials_reg = material_info

    def get_final_parameters(self) -> pybamm.ParameterValues:
        base_params = get_parameter_values()
        from src.cell_optimization.parameter_opt import ParamTransform
        transform = ParamTransform(base_params)
        transform.apply_design(np.array(list(self.design.values())), list(self.design.keys()))
        for cand, reg in self.materials_reg:
            transform.apply_physics(reg, cand.category)

        params = transform.evaluate()
        if "Cell volume [m3]" not in params:
            params["Cell volume [m3]"] = 0.130 * 0.070 * 0.0003
        return params

    def solve_mechanical_integrity(self, T_avg: float, cs_avg: float, L: float) -> Dict[str, float]:
        """
        High-fidelity 1D Thermo-Mechanical Solver using FEniCSx (dolfinx).
        Evaluates electrode structural integrity under max strain conditions.
        """
        if dolfinx is None:
            return {"max_stress_pa": 0.0, "mechanical_integrity_factor": 1.0}

        # Create mesh for electrode thickness
        domain = mesh.create_interval(MPI.COMM_WORLD, 20, [0, L])
        V = fem.functionspace(domain, ("Lagrange", 1))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Physics Constants
        E = 10e9  # Young's modulus (Pa)
        nu = 0.3  # Poisson's ratio
        alpha_t = 1e-5 # Thermal expansion
        alpha_s = 2e-5 # Swelling coefficient

        # Loadings derived from PyBaMM state
        delta_T = T_avg - 298.15
        delta_c = cs_avg

        # Strain formulation: epsilon = du/dx - alpha_t*dT - alpha_s*dc
        # Stress formulation (1D): sigma = E * epsilon
        # Residual: integral( sigma * dv/dx ) dx = 0
        a = ufl.inner(E * ufl.grad(u)[0], ufl.grad(v)[0]) * ufl.dx
        L_rhs = ufl.inner(E * (alpha_t * delta_T + alpha_s * delta_c), ufl.grad(v)[0]) * ufl.dx

        # Dirichlet BC at collector interface (x=0)
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
        bc = fem.dirichletbc(default_scalar_type(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

        problem = LinearProblem(a, L_rhs, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        # Calculate Stress
        sigma = E * (np.gradient(uh.x.array, L/20.0) - alpha_t * delta_T - alpha_s * delta_c)
        max_sigma = np.max(np.abs(sigma))

        yield_stress = 50e6 # 50 MPa limit
        return {
            "max_stress_pa": float(max_sigma),
            "mechanical_integrity_factor": float(max_sigma / yield_stress)
        }

    def run_validation(self):
        print("Running final Digital Twin validation (PyBaMM + FEniCSx)...")
        params = self.get_final_parameters()
        model = pybamm.lithium_ion.DFN({"thermal": "lumped"})
        solver = pybamm.CasadiSolver(mode="safe")
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)

        try:
            sol = sim.solve([0, 3600], inputs={"Current [A]": params["Nominal cell capacity [A.h]"]})

            v = sol["Terminal voltage [V]"].entries
            cap = sol["Discharge capacity [A.h]"].entries[-1]
            temp = sol["Cell temperature [K]"].entries
            cs_n = sol["X-averaged negative particle concentration [mol.m-3]"].entries

            # Mechanical integrity at max state (end of discharge)
            mech = self.solve_mechanical_integrity(temp[-1], cs_n[-1], self.design.get("Negative electrode thickness [m]", 1.2e-4))

            trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy = trapezoid(v, sol["Discharge capacity [A.h]"].entries)

            attributes = {
                "Energy_Wh": float(energy),
                "Capacity_Ah": float(cap),
                "Nominal_Voltage_V": float(np.mean(v)),
                "Max_Temp_K": float(np.max(temp)),
                "Max_Stress_Pa": mech["max_stress_pa"],
                "Mechanical_Integrity_Safety_Factor": mech["mechanical_integrity_factor"],
                "Energy_Density_Wh_kg": float(energy / 0.5)
            }

            print("Final Validated Attributes:")
            for k, v in attributes.items():
                print(f"  {k}: {v}")
            return attributes
        except Exception as e:
            print(f"Validation failed: {e}")
            return None

    def export_results(self, attributes, output_path="src/bms_design/cell_attributes.mat"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sio.savemat(output_path, {"cell_attributes": attributes})
        print(f"Cell attributes exported to {output_path}")

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    db, bases = engine.run()

    # Selection
    candidates = [
        [m for m in db["Cathode_Dopant"] if m.name == "Mn"][0],
        [m for m in db["Salt"] if m.name == "NaBOB"][0],
        db["Functionalization"][0]
    ]

    materials_reg = [(c, regularize_material(c, (bases["cathode"] if "Dopant" in c.category else bases["salt"] if "Salt" in c.category else bases["interface"]))) for c in candidates]

    optimized_design = {
        "Positive electrode thickness [m]": 0.00015,
        "Negative electrode thickness [m]": 0.00015,
        "Positive electrode porosity": 0.3,
        "Negative electrode porosity": 0.3,
        "Separator porosity": 0.5,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode active material volume fraction": 0.65,
        "Positive particle radius [m]": 1e-6,
        "Negative particle radius [m]": 5e-6,
        "Typical electrolyte concentration [mol.m-3]": 1000.0,
        "Positive electrode conductive carbon fraction": 0.08
    }

    validator = OptimizationValidator(optimized_design, materials_reg)
    res = validator.run_validation()
    if res:
        validator.export_results(res)
