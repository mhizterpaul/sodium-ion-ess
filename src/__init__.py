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
    "SimulationPlatform",
]

class SimulationPlatform:
    """Infrastructure layer for managing coupled simulations."""

    def __init__(self, parameter_set):
        self.params = parameter_set
        self.electro_thermal = ElectrochemicalThermalDriverModel()
        self.thermal_field = ThermalFieldModel()
        self.mechanics = ThermoelasticStrainModel()

    def run_coupled_simulation(self, times):
        """Executes the coupled DFN-Thermal-Strain pipeline."""
        # 1. Electrochemical-Thermal resolution
        model_dict = self.electro_thermal.build_model(self.params)
        et_sol = self.electro_thermal.simulate(model_dict, times)

        # 2. Thermal Field propagation
        # (Placeholder for T(x,t) resolution)

        # 3. Thermoelastic Strain evaluation
        mech_model = self.mechanics.build_model(self.params)

        return {
            "electro_thermal": et_sol,
            "mechanics": mech_model,
        }
