"""Thermal property definitions."""
from .heat_generation import HeatGenerationModel
from .heat_capacity import HeatCapacityModel
from .thermal_conductivity import ThermalConductivityModel

__all__ = ["HeatGenerationModel", "HeatCapacityModel", "ThermalConductivityModel"]
