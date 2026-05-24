import pybamm
from nfpp_sodium_ion.src.cell_parameters.data.electrodes.nfpp_cathode import NfppCathodeParameters
from nfpp_sodium_ion.src.cell_parameters.data.electrodes.hard_carbon_anode import HardCarbonAnodeParameters
from nfpp_sodium_ion.src.cell_parameters.data.electrolyte.na_pfp_dfo import NaPfpDfoParameters
from nfpp_sodium_ion.src.cell_parameters.data.thermal.heat_capacity import HeatCapacityModel
from nfpp_sodium_ion.src.cell_parameters.data.thermal.thermal_conductivity import ThermalConductivityModel
from nfpp_sodium_ion.src.cell_parameters.data.transport.conductivity import ConductivityModel
from nfpp_sodium_ion.src.cell_parameters.data.transport.diffusivity import DiffusivityModel
from nfpp_sodium_ion.src.cell_parameters.data.mechanics.elastic_moduli import ElasticModuliModel
from nfpp_sodium_ion.src.cell_parameters.data.mechanics.thermal_expansion import ThermalExpansionModel
from nfpp_sodium_ion.src.cell_parameters.data.mechanics.swelling_coefficients import SwellingCoefficientModel
from .cell_alpha import get_parameter_values as get_base_parameters

def get_parameter_values(updates=None):
    """
    Assembles the full NFPP sodium-ion parameter set.
    """
    params = get_base_parameters()
    if updates:
        params.update(updates, check_already_exists=False)
    return pybamm.ParameterValues(params)
