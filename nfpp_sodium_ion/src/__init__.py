"""NFPP Sodium-ion parameter set for PyBaMM."""

from .cell_parameters.data.base.cell import CellParameters
from .cell_parameters.data.base.chemistry import ChemistryParameters
from .cell_parameters.data.base.constants import Constants
from .cell_parameters.data.electrodes.nfpp_cathode import NfppCathodeParameters
from .cell_parameters.data.electrodes.hard_carbon_anode import HardCarbonAnodeParameters
from .cell_parameters.data.electrodes.separator import SeparatorParameters
from .cell_parameters.data.electrolyte.na_pfp_dfo import NaPfpDfoParameters
from .cell_parameters.data.transport.diffusivity import DiffusivityModel
from .cell_parameters.data.transport.conductivity import ConductivityModel
from .cell_parameters.data.thermal.heat_generation import HeatGenerationModel
from .cell_parameters.data.thermal.heat_capacity import HeatCapacityModel
from .cell_parameters.data.thermal.thermal_conductivity import ThermalConductivityModel
from .cell_parameters.data.kinetics.reaction_rates import ReactionRateModel
from .cell_parameters.data.kinetics.exchange_current_density import ExchangeCurrentDensityModel
from .cell_parameters.data.degradation.sei_growth import SeiGrowthModel
from .cell_parameters.data.degradation.cei_growth import CeiGrowthModel
from .cell_parameters.data.degradation.loss_of_lithium_equivalent import LossOfSodiumEquivalentModel


def get_parameter_values():
    """NFPP Sodium-ion parameter set for PyBaMM.

    Returns a dictionary of parameter values compatible with PyBaMM's DFN model.
    """
    return {
        "chemistry": "sodium_ion",
        "citation": "NFPP Sodium-ion base parameter package",
        "cell": CellParameters(),
        "chemistry_params": ChemistryParameters(),
        "constants": Constants(),
        "cathode": NfppCathodeParameters(),
        "anode": HardCarbonAnodeParameters(),
        "separator": SeparatorParameters(),
        "electrolyte": NaPfpDfoParameters(),
        "diffusivity": DiffusivityModel(),
        "conductivity": ConductivityModel(),
        "heat_generation": HeatGenerationModel(),
        "heat_capacity": HeatCapacityModel(),
        "thermal_conductivity": ThermalConductivityModel(),
        "reaction_rates": ReactionRateModel(),
        "exchange_current_density": ExchangeCurrentDensityModel(),
        "sei_growth": SeiGrowthModel(),
        "cei_growth": CeiGrowthModel(),
        "loss_of_sodium_equivalent": LossOfSodiumEquivalentModel(),
    }
