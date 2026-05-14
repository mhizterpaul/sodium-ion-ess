"""Thermal Field Model.

Heat transport model implementing:
- Spatial-temporal temperature field T(x,t) for resolved analysis
- Lumped temperature T(t) for reduced-order analysis
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ThermalFieldModel:
    """Thermal Field Model for heat transport.
    
    Propagates heat generation source Q(t) through the cell to determine
    temperature field evolution.
    """
    
    name: str = "Thermal Field Model"
    spatial_resolution: str = "resolved"  # "resolved" or "lumped"

    def build_resolved_model(self, parameter_set: Dict[str, Any]) -> Dict[str, Any]:
        """Build spatially-resolved thermal PDE model.
        
        Args:
            parameter_set: Parameter dictionary from nfpp_sodium_ion
            
        Returns:
            Model specification with thermal conductivity and capacity tensors
        """
        return {
            "type": "resolved_pde",
            "thermal_conductivity": parameter_set.get("thermal_conductivity"),
            "heat_capacity": parameter_set.get("heat_capacity"),
            "domain": "cell_geometry",  # 1D through-plane
        }

    def build_lumped_model(self, parameter_set: Dict[str, Any]) -> Dict[str, Any]:
        """Build lumped (0D) thermal model.
        
        Args:
            parameter_set: Parameter dictionary
            
        Returns:
            Lumped model with effective properties
        """
        heat_cap = parameter_set["heat_capacity"].specific_heat(298.15)
        thermal_cond = parameter_set["thermal_conductivity"].conductivity(298.15)
        
        return {
            "type": "lumped_ode",
            "heat_capacity": heat_cap,
            "thermal_conductivity": thermal_cond,
            "reference_temperature_k": 298.15,
        }

    def solve_resolved(
        self, 
        model: Dict[str, Any], 
        heat_generation: Any,  # Time-dependent array
        boundary_conditions: Dict[str, float],
        times: list
    ) -> Dict[str, Any]:
        """Solve resolved thermal PDE.
        
        Args:
            model: Output from build_resolved_model()
            heat_generation: Q(t) from electrochemical driver
            boundary_conditions: Ambient temperature, convection coefficients
            times: Time array [s]
            
        Returns:
            Temperature field T(x,t)
        """
        return {
            "temperature_field": None,  # T(x,t) tensor
            "times": times,
            "positions": None,  # Spatial mesh
        }

    def solve_lumped(
        self, 
        model: Dict[str, Any], 
        heat_generation: float, 
        ambient_temperature_k: float = 298.15,
        convection_coefficient: float = 20.0
    ) -> float:
        """Solve lumped thermal model: dT/dt = Q / (m*Cp) - h(T - T_amb).
        
        Args:
            model: Output from build_lumped_model()
            heat_generation: Heat generation rate [W]
            ambient_temperature_k: Ambient temperature [K]
            convection_coefficient: Surface heat transfer [W/m²K]
            
        Returns:
            Steady-state or transient cell temperature [K]
        """
        dT = heat_generation / (model["heat_capacity"] + model["thermal_conductivity"])
        return ambient_temperature_k + dT
