import re
import math
import numpy as np
from typing import Dict, Set, List, Optional, Any
from pymatgen.core import Composition

KT = 0.0259 # eV at 300K

def thermo_norm(x, ref=0.0):
    # Use material-energy scale as requested
    return (x - ref) / max(abs(ref), 0.1)

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

        # Physical Residual Mismatch
        dE = abs(thermo_norm(proxy_props.get("formation_energy", 0), base_props.get("formation_energy", 0)))
        dV = abs(geom_norm(proxy_props, base_props)["strain"]) / 0.05

        # Realization score
        realization = np.tanh(overlap * 3.0) * np.exp(-0.5 * (dE + dV))
        return float(np.clip(realization, 0.01, 1.0))
    except Exception:
        return 0.1

def derive_coupled_deltas(base_props, proxy_props, base_formula, proxy_formula) -> dict:
    # 2.3 Physical residuals
    dE = thermo_norm(proxy_props.get("formation_energy", 0), base_props.get("formation_energy", 0))
    dG = (proxy_props.get("band_gap", 0) - base_props.get("band_gap", 0)) / 1.0
    dV = geom_norm(proxy_props, base_props)["strain"]

    # Stability: lower is better (Energy above hull).
    # If proxy has LOWER energy above hull than base, it is MORE stable.
    # Positive dS now indicates IMPROVEMENT (MORE stable).
    dS = (base_props.get("stability", 0) - proxy_props.get("stability", 0)) / 0.2

    # Realization Factor
    R = compute_chemical_realization(base_formula, proxy_formula, base_props, proxy_props)

    # 2.4 Physics coupling rules
    voltage_boost = -0.05 * dE * R
    diffusivity_log_delta = (0.5 * dV - 0.2 * dG) * R
    reaction_rate_log_delta = (0.1 * dE - 0.3 * dG) * R
    stability_shift = dS * R

    # 2.5 Structured channels
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
            "conductivity_log_delta": -0.5 * dG * R
        },
        "structural": {
            "strain": float(dV * R)
        }
    }

def regularize_salt_props(base_salt_props: Dict[str, float], candidate_salt_props: Dict[str, float]) -> Dict[str, Any]:
    try:
        c_ratio = candidate_salt_props["conductivity"] / base_salt_props["conductivity"]
        v_ratio = candidate_salt_props["viscosity"] / base_salt_props["viscosity"]
        t_diff = candidate_salt_props["transference_number"] - base_salt_props["transference_number"]
        return {
            "transport": {
                "electrolyte_conductivity_log_delta": float(np.log(c_ratio)),
                "electrolyte_diffusivity_log_delta": float(-np.log(v_ratio)),
                "transference_number_delta": float(t_diff)
            }
        }
    except Exception: return {"transport": {}}

def regularize_functionalization(candidate_props: Dict[str, float]) -> Dict[str, Any]:
    return {
        "kinetic": {
            "sei_growth_log_delta": float(np.log(candidate_props.get("sei_growth_factor", 1.0))),
            "exchange_current_log_delta": float(np.log(candidate_props.get("exchange_current_factor", 1.0)))
        },
        "transport": {
            "sei_resistivity_log_delta": float(np.log(candidate_props.get("resistance_growth_factor", 1.0)))
        },
        "thermodynamic": {
            "initial_sodium_loss_delta": float(candidate_props.get("initial_sodium_loss_factor", 1.0) - 1.0)
        }
    }
