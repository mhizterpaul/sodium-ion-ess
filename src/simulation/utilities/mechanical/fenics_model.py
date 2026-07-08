"""Thermoelastic Strain Model (3D) in FEniCSx.

Solves the thermo-chemo-mechanical PDE:
∇·σ = 0
σ = C : (ε - ε_th - ε_soc)
ε = 0.5 * (∇u + ∇u^T)
ε_th = α(T - T0)
ε_soc = β(SOC - SOC0)
"""

import numpy as np
import pybamm
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

    def solve_strain(self, pybamm_sol: Any, params: Any, **kwargs) -> Dict[str, Any]:
        """Solves for the displacement and strain field with 1D-3D field mapping (Issue 7, 8)."""
        # Note: Rate-dependent scaling removed (Issue 9) as DFN concentration fields
        # already account for rate-induced internal gradients.

        if dolfinx is None:
            # Physics-based proxy for fallback (intercalation strain)
            try:
                T_all = pybamm_sol["Volume-averaged cell temperature [K]"].entries
            except (KeyError, pybamm.ModelError, AttributeError):
                T_all = pybamm_sol["Cell temperature [K]"].entries

            T_max = np.max(T_all)

            # Use particle-level stoichiometry change (Issue 6, 7)
            try:
                sto_p = pybamm_sol["X-averaged positive electrode surface stoichiometry"].entries
                sto_n = pybamm_sol["X-averaged negative electrode surface stoichiometry"].entries
                delta_sto = max(np.max(sto_p) - np.min(sto_p), np.max(sto_n) - np.min(sto_n))
            except (KeyError, pybamm.ModelError, AttributeError):
                cap_ah = pybamm_sol["Discharge capacity [A.h]"].entries
                nom_cap = params["Nominal cell capacity [A.h]"]
                soc_all = 1.0 - (cap_ah / nom_cap)
                delta_sto = np.max(soc_all) - np.min(soc_all)

            strain = (1e-5 * (T_max - 298.15) + 0.02 * delta_sto)
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

        # Map DFN fields (T(x), c_s(x)) to FEniCS 3D mesh (Issue 7, 8)
        Q = fem.functionspace(domain, ("CG", 1))

        try:
             # Extract spatial distribution at final time step (Issue 7, 8)
             # Use variables that provide spatial resolution across the stack
             T_spatial = pybamm_sol["Cell temperature [K]"].entries[:, -1]
             T_interp = lambda x: np.interp(x[2], np.linspace(0, H, len(T_spatial)), T_spatial)

             sto_p_spatial = pybamm_sol["Positive electrode surface stoichiometry"].entries[:, -1]
             sto_n_spatial = pybamm_sol["Negative electrode surface stoichiometry"].entries[:, -1]

             def stoichiometry_mapping(x):
                  # Map 1D electrode stoichiometric fields to 3D mesh zones (Issue 8)
                  val = np.zeros(x.shape[1])
                  # Anode zone (0 to H_n)
                  mask_n = x[2] <= H_n
                  val[mask_n] = np.interp(x[2, mask_n], np.linspace(0, H_n, len(sto_n_spatial)), sto_n_spatial)
                  # Cathode zone (H_n + H_s to H)
                  mask_p = x[2] >= (H_n + H_s)
                  val[mask_p] = np.interp(x[2, mask_p], np.linspace(H_n + H_s, H, len(sto_p_spatial)), sto_p_spatial)
                  return val

        except (KeyError, pybamm.ModelError, AttributeError):
             # Scalar fallback with physical grounding
             T_max = np.max(pybamm_sol["Cell temperature [K]"].entries)
             cap_ah = pybamm_sol["Discharge capacity [A.h]"].entries
             soc_final = 1.0 - (cap_ah[-1] / params["Nominal cell capacity [A.h]"])
             T_interp = lambda x: np.full(x.shape[1], T_max)
             stoichiometry_mapping = lambda x: np.full(x.shape[1], soc_final)

        T_field = fem.Function(Q)
        T_field.interpolate(T_interp)
        s_field = fem.Function(Q)
        s_field.interpolate(stoichiometry_mapping)

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

    def compute_endurance_metric(self, max_strain: float) -> Dict[str, float]:
        """
        Estimates cycle life (N_crit) using Coffin-Manson relationship.
        """
        # ε_p = ε_f' * (2N)^c -> N = 0.5 * (ε_p / ε_f')^(1/c)
        # For typical battery materials: ε_f' ~ 0.1, c ~ -0.5
        eps_f = 0.1
        c = -0.5
        n_crit = 0.5 * (max_strain / eps_f) ** (1/c) if max_strain > 0 else 1e12
        return {"n_crit": float(n_crit)}
