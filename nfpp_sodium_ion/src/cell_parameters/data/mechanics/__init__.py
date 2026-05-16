from .elastic_moduli import ElasticModuliModel
from .swelling_coefficients import SwellingCoefficientsModel
from .thermal_expansion import ThermalExpansionModel

__all__ = [
    "ElasticModuliModel",
    "SwellingCoefficientsModel",
    "ThermalExpansionModel",
]

# Note: FEniCSx infrastructure code previously here was simplified to support parameter-only focus.
# If full structural simulation is needed, refer to src/simulation/thermoelastic_strain.py
