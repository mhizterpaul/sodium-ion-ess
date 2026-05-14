"""FEniCSx Mechanics Module.

Provides FEniCSx dolfinx.fem.FunctionSpace compatibility layer for mechanics parameters.
Parameters from nfpp_sodium_ion can be directly used to configure FEniCSx problems.
"""

from .thermal_expansion import ThermalExpansionModel
from .swelling_coefficients import SwellingCoefficientModel
from .elastic_moduli import ElasticModuliModel

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import nfpp_hardcarbon.src.cell_parameters.data.mechanics as mechanics
    from mechanical.fem import Function, FunctionSpace, VectorFunctionSpace
    from mechanical.mesh import Mesh, create_rectangle, CellType
    import ufl
    import numpy as np
except ImportError:  # pragma: no cover
    mechanics = None
    Function = None
    FunctionSpace = None
    VectorFunctionSpace = None
    Mesh = None
    create_rectangle = None
    CellType = None
    ufl = None
    np = None


@dataclass
class ElasticModuliSpace:
    """Elastic moduli parameters as FEniCSx-compatible fields."""

    youngs_modulus_pa: float = 2.0e9
    poisson_ratio: float = 0.3

    def get_lame_parameters(self) -> tuple:
        """Compute Lamé parameters.

        Returns:
            (lambda, mu) for use in elasticity problems
        """
        E = self.youngs_modulus_pa
        nu = self.poisson_ratio
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lam, mu


@dataclass
class ThermalExpansionSpace:
    """Thermal expansion coefficient as FEniCSx expression.

    Provides temperature-dependent thermal strain for coupling to
    mechanical problem in form compatible with ufl definitions.
    """

    alpha_ref: float = 1e-5  # Reference thermal expansion [1/K]
    reference_temperature_k: float = 298.15

    def thermal_strain(self, temperature_field: Any) -> Any:
        """Compute thermal strain from temperature field.

        Args:
            temperature_field: ufl/dolfinx function or scalar

        Returns:
            Thermal strain suitable for use in elasticity equations
        """
        return self.alpha_ref * (temperature_field - self.reference_temperature_k)


@dataclass
class SwellingCoefficientSpace:
    """SOC-driven swelling as FEniCSx-compatible field.

    Swelling from insertion/deinsertion couples to mechanical problem.
    """

    swelling_coefficient: float = 5e-5  # [dimensionless/SOC fraction]

    def swelling_strain(self, soc_field: Any) -> Any:
        """Compute swelling strain from SOC field.

        Args:
            soc_field: State of charge [0,1] as ufl/dolfinx function

        Returns:
            Swelling-induced strain
        """
        return self.swelling_coefficient * soc_field


class ThermoelasticProblem:
    """FEniCSx thermoelastic continuum problem setup.

    Assembles and solves coupled thermoelastic equations with parameter
    fields from nfpp_sodium_ion package.
    """

    def __init__(
        self,
        mesh: Optional[Any] = None,
        elastic_moduli: Optional[ElasticModuliSpace] = None,
        thermal_expansion: Optional[ThermalExpansionSpace] = None,
        swelling: Optional[SwellingCoefficientSpace] = None
    ):
        """Initialize thermoelastic problem.

        Args:
            mesh: dolfinx.mesh.Mesh or None (creates reference mesh)
            elastic_moduli: ElasticModuliSpace with E and ν
            thermal_expansion: ThermalExpansionSpace with α
            swelling: SwellingCoefficientSpace with swelling coefficient
        """
        if mechanics is None:
            raise ImportError("dolfinx is required for thermoelastic problem")

        self.mesh = mesh or self._create_reference_mesh()
        self.elastic_moduli = elastic_moduli or ElasticModuliSpace()
        self.thermal_expansion = thermal_expansion or ThermalExpansionSpace()
        self.swelling = swelling or SwellingCoefficientSpace()

        # Function spaces
        self.V = VectorFunctionSpace(self.mesh, ("CG", 1))  # Displacement
        self.Q = FunctionSpace(self.mesh, ("CG", 1))        # Scalar fields (T, SOC)

    def _create_reference_mesh(self) -> Any:
        """Create 1D reference domain [0, L] for electrode.

        Returns:
            dolfinx mesh
        """
        L = 1.0  # Reference thickness [m]
        mesh = mechanics.mesh.create_interval(mechanics.MPI.COMM_WORLD, 100, [0.0, L])
        return mesh

    def setup_bilinear_form(self, u: Any, v: Any) -> Any:
        """Set up bilinear form for elasticity equation.

        Args:
            u: Trial function (displacement)
            v: Test function

        Returns:
            ufl bilinear form
        """
        if ufl is None:
            raise ImportError("ufl is required")

        # Strain-displacement relation
        def strain(w):
            return ufl.sym(ufl.grad(w))

        # Constitutive relation (Hooke's law)
        lam, mu = self.elastic_moduli.get_lame_parameters()

        def sigma(w):
            return lam * ufl.tr(strain(w)) * ufl.Identity(len(w)) + 2 * mu * strain(w)

        # Bilinear form
        a = ufl.inner(sigma(u), strain(v)) * ufl.dx
        return a

    def setup_rhs_thermal_swelling(
        self,
        v: Any,
        temperature_field: Any,
        soc_field: Any
    ) -> Any:
        """Set up RHS including thermal expansion and swelling loads.

        Args:
            v: Test function
            temperature_field: Temperature T(x,t)
            soc_field: SOC profile

        Returns:
            ufl linear form (RHS)
        """
        if ufl is None:
            raise ImportError("ufl is required")

        # Thermal strain
        eps_th = self.thermal_expansion.thermal_strain(temperature_field)

        # Swelling strain
        eps_sw = self.swelling.swelling_strain(soc_field)

        # Total inelastic strain
        eps_in = eps_th + eps_sw

        lam, mu = self.elastic_moduli.get_lame_parameters()
        sigma_in = lam * ufl.tr(eps_in) * ufl.Identity(1) + 2 * mu * eps_in

        # RHS: stress-divergence of inelastic part
        L = -ufl.inner(sigma_in, ufl.sym(ufl.grad(v))) * ufl.dx
        return L


class ParameterCompatibleMechanicsInterface:
    """Interface for using nfpp_sodium_ion parameters directly in FEniCSx.

    Maps parameter objects to FEniCSx function space configurations.
    """

    @staticmethod
    def from_parameter_set(parameter_set: Dict[str, Any]) -> ThermoelasticProblem:
        """Create thermoelastic problem from nfpp_sodium_ion parameter set.

        Args:
            parameter_set: Dict from load_default_parameter_set()

        Returns:
            Configured ThermoelasticProblem ready for FEniCSx
        """
        elastic_part = parameter_set.get("elastic_moduli")
        thermal_part = parameter_set.get("thermal_expansion")
        swelling_part = parameter_set.get("swelling_coefficients")

        elastic_space = None
        if elastic_part:
            elastic_space = ElasticModuliSpace(
                youngs_modulus_pa=elastic_part.youngs_modulus_pa,
                poisson_ratio=elastic_part.poisson_ratio
            )

        thermal_space = None
        if thermal_part:
            thermal_space = ThermalExpansionSpace(
                alpha_ref=thermal_part.alpha_ref
            )

        swelling_space = None
        if swelling_part:
            swelling_space = SwellingCoefficientSpace(
                swelling_coefficient=swelling_part.swelling_coefficient
            )

        return ThermoelasticProblem(
            elastic_moduli=elastic_space,
            thermal_expansion=thermal_space,
            swelling=swelling_space
        )


__all__ = [
    "ThermalExpansionModel",
    "SwellingCoefficientModel",
    "ElasticModuliModel",
    "ElasticModuliSpace",
    "ThermalExpansionSpace",
    "SwellingCoefficientSpace",
    "ThermoelasticProblem",
    "ParameterCompatibleMechanicsInterface",
]
