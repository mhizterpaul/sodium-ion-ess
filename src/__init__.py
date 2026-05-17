"""Simulation and Model Infrastructure.

Provides the core simulation drivers and platform infrastructure for
electrochemical-thermal and structural mechanics modeling.
"""

from .simulation.electrochemical_thermal import ElectrochemicalThermalDriverModel
from .simulation.thermal_field import ThermalFieldModel
from .simulation.thermoelastic_strain import ThermoelasticStrainModel

__all__ = [
    "ElectrochemicalThermalDriverModel",
    "ThermalFieldModel",
    "ThermoelasticStrainModel",
]

# Utility classes and platform infrastructure for downstream simulation
class SimulationPlatform:
    """Infrastructure layer for managing coupled simulations."""

    @staticmethod
    def initialize_coupled_problem(parameter_set):
        """Prepares the coupled electrochemical-thermal-mechanical environment."""
        # Initialize models
        electro_thermal = ElectrochemicalThermalDriverModel()
        thermal_field = ThermalFieldModel()
        mechanics = ThermoelasticStrainModel()

        return {
            "electro_thermal": electro_thermal,
            "thermal_field": thermal_field,
            "mechanics": mechanics,
            "parameters": parameter_set
        }
