import re
import math
import numpy as np
from typing import Dict, Set, List, Optional, Any

KT = 0.0259 # eV at 300K

def parse_stoich(formula: str) -> Dict[str, float]:
    """Parses a chemical formula into element counts."""
    tokens = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', formula)
    result = {}
    for element, amount in tokens:
        result[element] = float(amount) if amount else 1.0
    return result

def thermo_norm(x, ref):
    """Material-energy scale normalization."""
    return (x - ref) / max(abs(ref), 0.1)

def stoich_norm(formula_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalizes stoichiometric counts to sum to 1."""
    total = sum(formula_dict.values())
    if total == 0: return {k: 0.0 for k in formula_dict}
    return {k: v / total for k, v in formula_dict.items()}

def stoich_distance(base: Dict[str, float], proxy: Dict[str, float]) -> float:
    """Computes the difference between normalized stoichiometries."""
    keys = set(base) | set(proxy)
    return sum(abs(base.get(k, 0.0) - proxy.get(k, 0.0)) for k in keys)

def geom_norm(props, base_props):
    return {
        "volume_ratio": props["volume_per_atom"] / base_props["volume_per_atom"],
        "strain": (props["volume_per_atom"] - base_props["volume_per_atom"]) / (base_props["volume_per_atom"] + 1e-9)
    }

def apply_connectivity_scaling(props: Dict[str, float], phi: float = 0.75) -> Dict[str, float]:
    """Physically grounded connectivity-based scaling for organosiloxanes."""
    scaled = props.copy()
    a_density = 1.0
    if "volume_per_atom" in scaled:
        scaled["volume_per_atom"] = scaled["volume_per_atom"] / (phi**a_density)
    if "formation_energy" in scaled:
        scaled["formation_energy"] = scaled["formation_energy"] * phi
    if "stability" in scaled:
        scaled["stability"] = scaled["stability"] / (phi + 1e-9)
    return scaled

def compute_chemical_realization(
    base_formula: str,
    proxy_formula: str,
    base_props: Dict[str, float],
    proxy_props: Dict[str, float]
) -> float:
    """How safely can this proxy perturb the base material?"""
    def safe(x, ref=0.0):
        try: return float(x)
        except: return ref

    base_s = stoich_norm(parse_stoich(base_formula))
    proxy_s = stoich_norm(parse_stoich(proxy_formula))
    stoich_penalty = stoich_distance(base_s, proxy_s)

    e_base = set(base_s.keys())
    e_proxy = set(proxy_s.keys())
    r_chem = len(e_base & e_proxy) / max(len(e_base | e_proxy), 1)

    dE = thermo_norm(safe(proxy_props.get("formation_energy")), safe(base_props.get("formation_energy")))
    dV = (safe(proxy_props.get("volume_per_atom")) - safe(base_props.get("volume_per_atom"))) / (safe(base_props.get("volume_per_atom")) + 1e-9)

    z = (3.0 * r_chem - 1.5 * abs(dE) - 1.0 * abs(dV) - 2.0 * stoich_penalty)
    z = np.clip(z, -10, 10)
    return float(1.0 / (1.0 + np.exp(-z)))

def get_regularized_residuals(
    base_props: Dict[str, float],
    proxy_props: Dict[str, float],
    realization: float = 1.0
) -> Dict[str, float]:
    """
    Core Regularization: Computes attenuated physical residuals.
    All physics-to-performance mapping (delta derivation) happens in parameter_opt.py.
    """
    r = float(realization)

    dE = thermo_norm(proxy_props["formation_energy"], base_props["formation_energy"]) * r
    dG = ((proxy_props["band_gap"] - base_props["band_gap"]) / 1.0) * r
    dV = geom_norm(proxy_props, base_props)["strain"] * r
    dS = ((base_props["stability"] - proxy_props["stability"]) / 0.2) * r

    return {
        "dE": float(dE),
        "dG": float(dG),
        "dV": float(dV),
        "dS": float(dS)
    }

def regularize_material(candidate_obj: Any, base_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Consolidated Regularization Entry Point.
    Returns attenuated physical residuals and realization metadata.
    """
    props = candidate_obj.properties
    base_props = base_dict["properties"]
    base_formula = base_dict["formula"]

    is_network = candidate_obj.category == "Functionalization_Proxy"
    if is_network:
        props = apply_connectivity_scaling(props, phi=0.75)

    realization = compute_chemical_realization(base_formula, candidate_obj.composition, base_props, props)
    residuals = get_regularized_residuals(base_props, props, realization=realization)

    return {
        "residuals": residuals,
        "realization": realization,
        "proxy_uncertainty": (1.0 - realization)**2,
        "is_network": is_network
    }
