import re
import math
import numpy as np
from typing import Dict, Set, List, Optional, Any
from pymatgen.core import Composition

KT = 0.0259 # eV at 300K

def thermo_norm(x, ref=0.0):
    # Boltzmann scaling: nondimensionalizing energy residuals
    return (x - ref) / KT

def normalized_residual(value, scale):
    if scale <= 0:
        return 0.0
    return value / scale

def stoich_norm(formula: str) -> dict:
    try:
        comp = Composition(formula)
        total = sum(comp.values())
        if total == 0: return {}
        return {k: v / total for k, v in comp.items()}
    except Exception:
        return {}

def activation_energy_proxy(base_props: Dict[str, float], cand_props: Dict[str, float]) -> float:
    """
    Transport-grounded Activation Energy Proxy Model (Issue 3.2).
    Ea = 0.5*dr + 0.3*dV + 0.2*dS
    """
    radius_term = abs(cand_props.get("ionic_radius", 1.0) - base_props.get("ionic_radius", 1.0))
    volume_term = abs(cand_props.get("volume_per_atom", 1.0) - base_props.get("volume_per_atom", 1.0))
    stability_term = abs(cand_props.get("stability", 0.0) - base_props.get("stability", 0.0))

    return float(0.5 * radius_term + 0.3 * volume_term + 0.2 * stability_term)

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
        vp = proxy_props.get("volume_per_atom", 1.0)
        vb = base_props.get("volume_per_atom", 1.0)
        dV = abs(vp - vb) / vb / 0.05

        realization = np.tanh(overlap * 3.0) * np.exp(-0.5 * (dE + dV))
        return float(np.clip(realization, 0.01, 1.0))
    except Exception:
        return 0.1

def derive_coupled_deltas(base_props, proxy_props, base_formula, proxy_formula) -> dict:
    # 2.3 Physical residuals normalized by KT (Issue 16)
    dE = thermo_norm(proxy_props.get("formation_energy", 0), base_props.get("formation_energy", 0))

    stability_delta = proxy_props.get("stability", 0) - base_props.get("stability", 0)
    stability_scale = max(abs(base_props.get("stability", 1.0)), abs(proxy_props.get("stability", 1.0)), 1e-12)
    dS = normalized_residual(stability_delta, stability_scale)

    vp = proxy_props.get("volume_per_atom", 1.0)
    vb = base_props.get("volume_per_atom", 1.0)
    dV = (vp - vb) / vb

    # Realization Factor: Regularizes deltas for different chemistries
    R = compute_chemical_realization(base_formula, proxy_formula, base_props, proxy_props)

    # 2.4 Physically grounded coupling rules attenuated by Realization
    # Correct electrochemical mapping (Issue 3.3)
    F = 96485.0
    NA = 6.02214076e23
    # dE * KT is the energy difference in eV/atom
    energy_joule = dE * KT * 1.602176634e-19
    voltage_boost = -(energy_joule * NA / F) * R

    # Ionic Conductivity Model using Ea proxy
    Ea_base = activation_energy_proxy(base_props, base_props)
    Ea_proxy = activation_energy_proxy(base_props, proxy_props)
    conductivity_log_delta = (Ea_base - Ea_proxy) / KT * R

    diffusivity_log_delta = (dV - 0.1 * abs(dE)) * R
    reaction_rate_log_delta = (0.1 * dE + 0.1 * dS) * R
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

def regularize_salt_props(base_salt_formula: str, candidate_salt_formula: str, base_salt_props: Dict[str, float], candidate_salt_props: Dict[str, float]) -> Dict[str, Any]:
    """Derive electrolyte deltas attenuated by chemical realization."""
    try:
        R = compute_chemical_realization(base_salt_formula, candidate_salt_formula, base_salt_props, candidate_salt_props)

        # Ea proxy-based conductivity
        Ea_base = activation_energy_proxy(base_salt_props, base_salt_props)
        Ea_can = activation_energy_proxy(base_salt_props, candidate_salt_props)
        dEa_norm = (Ea_base - Ea_can) / KT

        v_can = candidate_salt_props.get("volume_per_atom", 1.0)
        v_base = base_salt_props.get("volume_per_atom", 1.0)
        dV = (v_can - v_base) / v_base

        return {
            "transport": {
                "electrolyte_conductivity_log_delta": float(dEa_norm * R),
                "electrolyte_diffusivity_log_delta": float(-1.0 * dV * R)
            }
        }
    except Exception: return {"transport": {}}

def regularize_functionalization(base_int_formula: str, candidate_func_formula: str, base_props: Dict[str, float], candidate_props: Dict[str, float]) -> Dict[str, Any]:
    """MTMS Functionalization via film growth surrogate and chemical realization."""
    try:
        R = compute_chemical_realization(base_int_formula, candidate_func_formula, base_props, candidate_props)
        dS = (base_props.get("stability", 0) - candidate_props.get("stability", 0)) / KT

        # Mapping to SEI kinetics (Issue 6)
        return {
            "kinetic": {
                "sei_growth_log_delta": float(-0.8 * dS * R),
                "exchange_current_log_delta": float(0.2 * dS * R)
            },
            "transport": {
                "sei_resistivity_log_delta": float(-0.5 * dS * R)
            },
            "thermodynamic": {
                "initial_sodium_loss_delta": float(-0.3 * dS * R)
            },
            "mechanical": {
                # Modulus degradation factor (Issue 1 - from reviewer feedback)
                "modulus_degradation_factor": float(1.0 - 0.3 * np.clip(1.0 - np.exp(-abs(dS)*R), 0.0, 1.0))
            }
        }
    except Exception: return {"kinetic": {}, "transport": {}, "thermodynamic": {}, "mechanical": {}}

def mechanical_stability_metric(stresses: Optional[List[float]] = None) -> float:
    """Von Mises yield proxy for fast Stage 1 mechanical ranking."""
    if stresses and len(stresses) >= 2:
        s1 = stresses[0]
        s2 = stresses[1]
        s3 = 0.0
        von_mises = np.sqrt(0.5 * ((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2))
        return float(-von_mises)
    elif stresses:
        return float(-max(np.abs(stresses)))
    return -1e-6
