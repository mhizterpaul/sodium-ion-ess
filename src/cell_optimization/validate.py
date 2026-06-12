import pybamm
import numpy as np
import scipy.io as sio
import os
import json
import traceback
from typing import Dict, Any, List, Tuple
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCategory, MaterialCandidate
from src.cell_optimization.chem_regularization import derive_coupled_deltas, regularize_salt_props, regularize_functionalization
from src.cell_optimization.parameter_opts import ParamTransform, OptimizerEngine

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
    High-fidelity validation using DFN model and dolfinx mechanics.
    """

    def __init__(self, optimized_design: Dict[str, float], combined_deltas: Dict[str, Any]):
        self.design = optimized_design
        self.deltas = combined_deltas

    def get_final_parameters(self) -> pybamm.ParameterValues:
        base_params = get_parameter_values()
        pt = ParamTransform(pybamm.ParameterValues(base_params))

        # Apply physics
        pt.apply_physics_deltas(self.deltas)

        # Apply design vector
        from src.cell_optimization.parameter_opts import DESIGN_SPACE
        pt.apply_design_vector(
            np.array([self.design[k] for k in DESIGN_SPACE if k in self.design]),
            [k for k in DESIGN_SPACE if k in self.design]
        )

        p = pt.get_parameter_values()
        # Ensure thermal parameters are set
        if "Cell volume [m3]" not in p:
            p["Cell volume [m3]"] = 0.13 * 0.07 * 0.01
        if "Cell cooling surface area [m2]" not in p:
            p["Cell cooling surface area [m2]"] = 2 * (0.13 * 0.07 + 0.13 * 0.01 + 0.07 * 0.01)
        return p

    def solve_mechanical_integrity(self, T_avg: float, cs_avg: float, L: float) -> Dict[str, float]:
        """
        High-fidelity 1D Thermo-Mechanical Solver using FEniCSx (dolfinx).
        Evaluates electrode structural integrity under max strain conditions.
        """
        # Ensure T_avg and cs_avg are scalars
        T_val = float(np.mean(T_avg))
        cs_val = float(np.mean(cs_avg))

        if dolfinx is None:
            # Simple physical proxy if dolfinx is missing
            E = 10e9
            alpha_t = 1e-5
            alpha_s = 2e-5
            strain = alpha_t * (T_val - 298.15) + alpha_s * cs_val
            stress = E * strain
            return {"max_stress_pa": float(stress), "mechanical_integrity_factor": float(stress / 50e6)}

        # Create mesh for electrode thickness
        domain = mesh.create_interval(MPI.COMM_WORLD, 20, [0, L])
        V = fem.functionspace(domain, ("Lagrange", 1))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Physics Constants
        E = 10e9  # Young's modulus (Pa)
        alpha_t = 1e-5 # Thermal expansion
        alpha_s = 2e-5 # Swelling coefficient

        # Loadings
        delta_T = T_val - 298.15
        delta_c = cs_val

        a = ufl.inner(E * ufl.grad(u)[0], ufl.grad(v)[0]) * ufl.dx
        L_rhs = ufl.inner(E * (alpha_t * delta_T + alpha_s * delta_c), ufl.grad(v)[0]) * ufl.dx

        # Dirichlet BC at collector interface (x=0)
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
        bc = fem.dirichletbc(default_scalar_type(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

        problem = LinearProblem(a, L_rhs, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        # Calculate Stress (simplified 1D)
        sigma = E * (np.gradient(uh.x.array, L/20.0) - alpha_t * delta_T - alpha_s * delta_c)
        max_sigma = np.max(np.abs(sigma))

        return {
            "max_stress_pa": float(max_sigma),
            "mechanical_integrity_factor": float(max_sigma / 50e6)
        }

    def run_validation(self):
        print("Running high-fidelity DFN validation...")
        params = self.get_final_parameters()
        model = pybamm.lithium_ion.DFN({"thermal": "lumped"})

        sim = pybamm.Simulation(model, parameter_values=params)

        try:
            sol = sim.solve([0, 3600], inputs={"Current [A]": params["Nominal cell capacity [A.h]"]})

            v = sol["Terminal voltage [V]"].data
            cap = sol["Discharge capacity [A.h]"].data[-1]
            temp = sol["Volume-averaged cell temperature [K]"].data
            cs_n = sol["X-averaged negative particle concentration [mol.m-3]"].data

            # Mechanical integrity
            mech = self.solve_mechanical_integrity(temp[-1], cs_n[-1], self.design.get("Negative electrode thickness [m]", 100e-6))

            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy = trapz_func(v * sol["Current [A]"].data, sol["Time [s]"].data) / 3600

            attributes = {
                "energy_wh": float(energy),
                "capacity_ah": float(cap),
                "voltage_avg": float(np.mean(v)),
                "max_temp_k": float(np.max(temp)),
                "max_stress_pa": mech["max_stress_pa"],
                "mechanical_integrity_factor": mech["mechanical_integrity_factor"]
            }

            print("Validation complete.")
            print(json.dumps(attributes, indent=2))
            return attributes
        except Exception as e:
            print(f"Validation failed: {e}")
            traceback.print_exc()
            return None

if __name__ == "__main__":
    from src.cell_optimization.parameter_opts import run_workflow
    best_design_result = run_workflow()
    if best_design_result:
        validator = OptimizationValidator(
            best_design_result.get("design_specs", {}),
            best_design_result.get("combined_deltas", {})
        )
        validator.run_validation()
