import re
import math
import numpy as np
from typing import Dict, Set, List, Optional, Any
from pymatgen.core import Composition

KT = 0.0259 # eV at 300K

def thermo_norm(x, ref=0.0):
    # Boltzmann scaling: nondimensionalizing energy residuals
    return (x - ref) / KT

def stoich_norm(formula: str) -> dict:
    try:
        comp = Composition(formula)
        total = sum(comp.values())
        if total == 0: return {}
        return {k: v / total for k, v in comp.items()}
    except Exception:
        return {}

def geom_norm(props, base_props):
    vp = props.get("volume_per_atom", 1.0)
    vb = base_props.get("volume_per_atom", 1.0)
    return {
        "volume_ratio": vp / vb,
        "strain": (vp - vb) / vb
    }

def compute_chemical_realization(
    base_formula: str,
    proxy_formula: str,
    base_props: Dict[str, float],
    proxy_props: Dict[str, float]
) -> float:
    """How safely can this proxy perturb the base material?"""
    try:
        c_base = stoich_norm(base_formula)
        c_proxy = stoich_norm(proxy_formula)
        shared = set(c_base) & set(c_proxy)
        overlap = sum(min(c_base.get(e, 0), c_proxy.get(e, 0)) for e in shared)

        # Physical Residual Mismatch (nondimensionalized)
        dE = abs(thermo_norm(proxy_props.get("formation_energy", 0), base_props.get("formation_energy", 0)))
        dV = abs(geom_norm(proxy_props, base_props)["strain"]) / 0.05

        # Realization score based on overlap and residual attenuation
        realization = np.tanh(overlap * 3.0) * np.exp(-0.5 * (dE + dV))
        return float(np.clip(realization, 0.01, 1.0))
    except Exception:
        return 0.1

def derive_coupled_deltas(base_props, proxy_props, base_formula, proxy_formula) -> dict:
    # 2.3 Physical residuals
    dE = thermo_norm(proxy_props.get("formation_energy", 0), base_props.get("formation_energy", 0))
    dG = (proxy_props.get("band_gap", 0) - base_props.get("band_gap", 0)) / KT
    dV = geom_norm(proxy_props, base_props)["strain"]

    # Stability improvement dS (positive means more stable)
    dS = (base_props.get("stability", 0) - proxy_props.get("stability", 0)) / KT

    # Realization Factor
    R = compute_chemical_realization(base_formula, proxy_formula, base_props, proxy_props)

    # 2.4 Physically grounded coupling rules
    # Voltage shift follows Nernstian/Energy relation
    voltage_boost = -dE * KT * R
    # Conductivity follows Arrhenius/Band-gap relation: exp(-Eg/2kT)
    conductivity_log_delta = -0.5 * dG * R
    # Diffusion/Kinetics coupled to volume and energy barrier shifts
    diffusivity_log_delta = (dV - 0.2 * dG) * R
    reaction_rate_log_delta = (0.1 * dE - 0.3 * dG) * R
    stability_shift = dS * R

    return {
        "thermodynamic": {
            "voltage_boost": float(voltage_boost),
            "stability_shift": float(stability_shift)
        },
        "kinetic": {
            "exchange_current_log_delta": float(reaction_rate_log_delta)
        },
        "transport": {
            "diffusivity_log_delta": float(diffusivity_log_delta),
            "conductivity_log_delta": float(conductivity_log_delta)
        },
        "structural": {
            "strain": float(dV * R)
        }
    }

def regularize_salt_props(base_salt_props: Dict[str, float], candidate_salt_props: Dict[str, float]) -> Dict[str, Any]:
    """Derive electrolyte deltas from physical residuals (dG for conductivity, dV for viscosity)."""
    try:
        dG = (candidate_salt_props.get("band_gap", 0) - base_salt_props.get("band_gap", 0)) / KT
        dV = (candidate_salt_props.get("volume_per_atom", 1.0) - base_salt_props.get("volume_per_atom", 1.0)) / 1.0

        return {
            "transport": {
                "electrolyte_conductivity_log_delta": float(-0.5 * dG),
                "electrolyte_diffusivity_log_delta": float(-1.0 * dV)
            }
        }
    except Exception: return {"transport": {}}

def regularize_functionalization(base_props: Dict[str, float], candidate_props: Dict[str, float]) -> Dict[str, Any]:
    """MTMS Functionalization regularized via weighted network dilution model."""
    try:
        # Connectivity fraction for MTMS (phi = 0.75)
        phi = 0.75
        alpha_v = 1.0
        alpha_d = 1.5

        dS = (base_props.get("stability", 0) - candidate_props.get("stability", 0)) / KT

        # Mapping to SEI kinetics and transport using network dilution exponents
        return {
            "kinetic": {
                "sei_growth_log_delta": float(-dS * phi),
                "exchange_current_log_delta": float(0.1 * dS)
            },
            "transport": {
                "sei_resistivity_log_delta": float(-alpha_d * dS * phi)
            },
            "thermodynamic": {
                "initial_sodium_loss_delta": float(-alpha_v * dS * phi)
            }
        }
    except Exception: return {"kinetic": {}, "transport": {}, "thermodynamic": {}}
