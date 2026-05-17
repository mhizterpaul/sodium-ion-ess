import numpy as np
import pybamm
from ..calibration.derivation import get_derived_parameters
from .data.electrodes.nfpp_cathode import NfppCathodeParameters
from .data.electrodes.hard_carbon_anode import HardCarbonAnodeParameters
from .data.mechanics.elastic_moduli import ElasticModuliModel
from .data.base.cell import CellParameters

def nfpp_diffusivity_literature(sto, T):
    D_ref = 1.0 * 10 ** (-14)
    E_D_s = 30000
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius

def nfpp_ocp_literature(sto):
    u_eq = 3.11 - 0.5 * sto + 0.1 * np.exp(-100 * sto) - 0.1 * np.exp(-100 * (1 - sto))
    return u_eq

def hard_carbon_diffusivity_literature(sto, T):
    D_ref = 5.0 * 10 ** (-15)
    E_D_s = 40000
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius

def hard_carbon_ocp_literature(sto):
    u_eq = 0.1 * np.exp(-20 * sto) + 0.05 * (1 - sto)
    return u_eq

def exchange_current_density_sodium(c_e, c_s_surf, c_s_max, T):
    m_ref = 2e-5
    E_r = 35000
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5

def current_function(t):
    """
    Returns the current at time t.
    Supports variable current profiles via InputParameter.
    """
    return pybamm.InputParameter("Current [A]")

def get_parameter_values():
    cathode = NfppCathodeParameters()
    anode = HardCarbonAnodeParameters()
    cell = CellParameters()
    elastic = ElasticModuliModel()
    derived = get_derived_parameters()

    return {
        "chemistry": "sodium_ion",
        "Negative current collector thickness [m]": cell.anode_collector_thickness_um * 1e-6,
        "Negative electrode thickness [m]": 0.00012,
        "Separator thickness [m]": cell.separator_thickness_um * 1e-6,
        "Positive electrode thickness [m]": 0.0001,
        "Positive current collector thickness [m]": cell.cathode_collector_thickness_um * 1e-6,
        "Electrode height [m]": 0.130,
        "Electrode width [m]": 0.070,
        "Number of electrodes connected in parallel to make a cell": float(derived["n_layers_10ah"]),
        "Nominal cell capacity [A.h]": cell.capacity_ah,
        "Current function [A]": current_function,
        "Contact resistance [Ohm]": 0,
        "Negative electrode conductivity [S.m-1]": 256.0,
        "Maximum concentration in negative electrode [mol.m-3]": derived["c_max_n"],
        "Negative particle diffusivity [m2.s-1]": hard_carbon_diffusivity_literature,
        "Negative electrode OCP [V]": hard_carbon_ocp_literature,
        "Negative electrode porosity": 0.3,
        "Negative electrode active material volume fraction": derived["eps_am_n"],
        "Negative particle radius [m]": 5e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode exchange-current density [A.m-2]": exchange_current_density_sodium,
        "Negative electrode density [kg.m-3]": anode.density_kg_m3,
        "Negative electrode Young's modulus [Pa]": 10.0e9,
        "Negative electrode Poisson's ratio": 0.3,
        "Negative electrode OCP entropic change [V.K-1]": 0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Positive electrode conductivity [S.m-1]": 50.0,
        "Maximum concentration in positive electrode [mol.m-3]": derived["c_max_p"],
        "Positive particle diffusivity [m2.s-1]": nfpp_diffusivity_literature,
        "Positive electrode OCP [V]": nfpp_ocp_literature,
        "Positive electrode porosity": 0.3,
        "Positive electrode active material volume fraction": derived["eps_am_p"],
        "Positive particle radius [m]": 1e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode exchange-current density [A.m-2]": exchange_current_density_sodium,
        "Positive electrode density [kg.m-3]": cathode.density_kg_m3,
        "Positive electrode Young's modulus [Pa]": elastic.youngs_modulus_pa,
        "Positive electrode Poisson's ratio": elastic.poisson_ratio,
        "Positive electrode OCP entropic change [V.K-1]": 0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Separator porosity": 0.5,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 1000.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        "Initial concentration in electrolyte [mol.m-3]": 1200.0,
        "Cation transference number": 0.45,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": 5e-10,
        "Electrolyte conductivity [S.m-1]": 1.0,
        "Reference temperature [K]": 298.15,
        "Ambient temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
        "Lower voltage cut-off [V]": 2.0,
        "Upper voltage cut-off [V]": 3.8,
        "Initial concentration in negative electrode [mol.m-3]": 0.95 * derived["c_max_n"],
        "Initial concentration in positive electrode [mol.m-3]": 0.05 * derived["c_max_p"],
    }
