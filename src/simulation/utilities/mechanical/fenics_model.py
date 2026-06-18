"""Thermoelastic Strain Model (3D) in FEniCSx.

Solves the thermo-chemo-mechanical PDE:
∇·σ = 0
σ = C : (ε - ε_th - ε_soc)
ε = 0.5 * (∇u + ∇u^T)
ε_th = α(T - T0)
ε_soc = β(SOC - SOC0)
"""

import numpy as np
from typing import Any, Dict, Optional
from dataclasses import dataclass

try:
    import dolfinx
    from dolfinx import fem, mesh, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    import ufl
except ImportError:
    dolfinx = None

@dataclass
class ThermoelasticStrainModel:
    critical_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.critical_thresholds is None:
            self.critical_thresholds = {"NFPP": 2e-3, "hard_carbon": 1e-3, "SEI": 5e-4}

    def solve_strain(self, pybamm_sol: Any, params: Any) -> Dict[str, Any]:
        """Solves for the displacement and strain field."""
        if dolfinx is None:
            # Physics-based proxy for fallback (linear expansion)
            T = np.max(pybamm_sol["Cell temperature [K]"].data)
            soc = 1.0 - (pybamm_sol["Discharge capacity [A.h]"].data[-1] / params["Nominal cell capacity [A.h]"])
            strain = 1e-5 * (T - 298.15) + 0.02 * soc
            return {"max_strain": float(strain)}

        # Electrode dimensions (Pouch section) from paper.md and cell_alpha.py
        L = params.get("Electrode height [m]", 0.130)
        W = params.get("Electrode width [m]", 0.070)
        H_p = params.get("Positive electrode thickness [m]", 100e-6)
        H_n = params.get("Negative electrode thickness [m]", 120e-6)
        H_s = params.get("Separator thickness [m]", 25e-6)
        H = H_p + H_n + H_s # Total stack height for mechanical PDE

        domain = mesh.create_box(MPI.COMM_WORLD, [[0, 0, 0], [L, W, H]], [10, 10, 3])
        V = fem.functionspace(domain, ("CG", 1, (3,)))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Map DFN outputs to FEniCS
        Q = fem.functionspace(domain, ("CG", 1))
        T_max = np.max(pybamm_sol["Cell temperature [K]"].data)
        soc_val = 1.0 - (pybamm_sol["Discharge capacity [A.h]"].data[-1] / params["Nominal cell capacity [A.h]"])

        T_field = fem.Function(Q)
        T_field.interpolate(lambda x: np.full(x.shape[1], T_max))
        s_field = fem.Function(Q)
        s_field.interpolate(lambda x: np.full(x.shape[1], soc_val))

        # Material parameters
        E = fem.Constant(domain, default_scalar_type(params.get("Negative electrode Young's modulus [Pa]", 10e9)))
        nu = fem.Constant(domain, default_scalar_type(0.3))
        alpha = fem.Constant(domain, default_scalar_type(1e-5)) # Thermal expansion
        beta = fem.Constant(domain, default_scalar_type(0.02)) # SOC expansion
        T_ref = fem.Constant(domain, default_scalar_type(298.15))

        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u, T, s):
            eps_inel = (alpha * (T - T_ref) + beta * s) * ufl.Identity(3)
            return lmbda * ufl.tr(epsilon(u) - eps_inel) * ufl.Identity(3) + 2 * mu * (epsilon(u) - eps_inel)

        a = ufl.inner(sigma(u, T_field, s_field), epsilon(v)) * ufl.dx
        L_form = ufl.dot(fem.Constant(domain, default_scalar_type((0, 0, 0))), v) * ufl.dx

        # BC: Fixed at one face
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0))
        bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

        problem = LinearProblem(a, L_form, bcs=[bc])
        uh = problem.solve()

        # Extract strain
        strain_expr = fem.Expression(ufl.sqrt(ufl.inner(epsilon(uh), epsilon(uh))), Q.element.interpolation_points())
        strains = fem.Function(Q)
        strains.interpolate(strain_expr)

        return {"max_strain": float(np.max(strains.x.array))}
