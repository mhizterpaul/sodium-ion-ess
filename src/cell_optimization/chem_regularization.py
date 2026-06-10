import re
import math
import numpy as np
from typing import Dict, Set, List, Optional, Any

KT = 0.0259 # eV at 300K

def parse_formula(formula: str) -> Set[str]:
    """
    Extracts elemental symbols from a chemical formula.
    Example: 'Na4Fe3P4O15' -> {'Na', 'Fe', 'P', 'O'}
    """
    return set(re.findall(r'[A-Z][a-z]?', formula))

def thermo_norm(x, ref=0.0):
    return (x - ref) / KT

def stoich_norm(formula_dict: dict) -> dict:
    total = sum(formula_dict.values())
    if total == 0: return {k: 0 for k in formula_dict}
    return {k: v / total for k, v in formula_dict.items()}

def geom_norm(props, base_props):
    return {
        "volume_ratio": props["volume_per_atom"] / base_props["volume_per_atom"],
        "strain": (props["volume_per_atom"] - base_props["volume_per_atom"]) / base_props["volume_per_atom"]
    }

def compute_chemical_realization(
    base_formula: str,
    proxy_formula: str,
    base_props: Dict[str, float],
    proxy_props: Dict[str, float]
) -> float:
    """
    Stable bounded realization score in [0,1]
    using additive logit fusion instead of multiplicative collapse.
    """

    def safe(x, ref=0.0):
        try:
            return float(x)
        except:
            return ref

    # --- chemical overlap ---
    e_base = parse_formula(base_formula)
    e_proxy = parse_formula(proxy_formula)

    r_chem = len(e_base & e_proxy) / max(len(e_base | e_proxy), 1)

    # --- normalized physics residuals (log-stabilized via tanh) ---
    dv = np.tanh(
        (safe(proxy_props.get("volume_per_atom")) -
         safe(base_props.get("volume_per_atom"))) /
        (safe(base_props.get("volume_per_atom")) + 1e-9)
    )

    de = np.tanh(
        (safe(proxy_props.get("band_gap")) -
         safe(base_props.get("band_gap"))) /
        (safe(base_props.get("band_gap")) + 1e-6)
    )

    df = np.tanh(
        safe(proxy_props.get("formation_energy")) -
        safe(base_props.get("formation_energy"))
    )

    # energy norm instead of raw quadratic collapse
    phys_energy = (0.5 * dv**2 + 0.3 * de**2 + 0.2 * df**2)

    # logit-space fusion (stable weighting)
    z = 3.0 * r_chem - 2.5 * phys_energy

    z = np.clip(z, -10, 10)
    return float(1.0 / (1.0 + np.exp(-z)))

def derive_coupled_deltas(
    base_props: Dict[str, float],
    proxy_props: Dict[str, float],
    base_params: Any
) -> Dict[str, Dict[str, float]]:
    """
    Physics-only transformation layer.
    Replaces latent vector model with physical residual state.
    """
    dE = thermo_norm(proxy_props["formation_energy"], base_props["formation_energy"])
    dG = (proxy_props["band_gap"] - base_props["band_gap"]) / 1.0
    dV = geom_norm(proxy_props, base_props)["strain"]
    dS = (proxy_props["stability"] - base_props["stability"]) / 0.2

    # Physics coupling rules
    voltage_boost = -0.05 * dE
    diffusivity_log_delta = 0.5 * dV - 0.2 * dG
    reaction_rate_log_delta = 0.1 * dE - 0.3 * dG
    stability_shift = dS

    def clip_log(x):
        return float(np.clip(x, -5, 5))

    return {
        "thermodynamic": {
            "voltage_boost": float(voltage_boost),
            "stability_shift": float(stability_shift)
        },
        "kinetic": {
            "reaction_rate_log_delta": clip_log(reaction_rate_log_delta)
        },
        "transport": {
            "diffusivity_log_delta": clip_log(diffusivity_log_delta)
        },
        "structural": {
            "volume_expansion_coeff": float(dV)
        }
    }
