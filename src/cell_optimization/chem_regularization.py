import re
import math
import numpy as np
from typing import Dict, Set, List

def parse_formula(formula: str) -> Set[str]:
    """
    Extracts elemental symbols from a chemical formula.
    Example: 'Na4Fe3P4O15' -> {'Na', 'Fe', 'P', 'O'}
    """
    return set(re.findall(r'[A-Z][a-z]?', formula))

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

# Latent Physics Metric Gz (Curvature preference in physics directions)
GZ_METRIC = np.diag([10.0, 5.0, 2.0, 1.0])

def derive_coupled_deltas(
    base_props: Dict[str, float],
    proxy_props: Dict[str, float],
    base_v: float,
    realization: float
) -> Dict[str, Dict[str, float]]:
    """
    Physically constrained channel model with linear coupling matrix.
    """

    # --- normalized latent vector ---
    z = np.array([
        proxy_props["formation_energy"] - base_props["formation_energy"],
        proxy_props["volume_per_atom"] - base_props["volume_per_atom"],
        proxy_props["band_gap"] - base_props["band_gap"],
        proxy_props["stability"] - base_props["stability"]
    ])

    # normalize vector (critical stability fix)
    z = z / (np.linalg.norm(z) + 1e-9)

    # linear coupling matrix (replaces ad-hoc scalars)
    W = np.array([
        [ 1.2,  0.3, -0.2,  0.5],
        [-0.4,  1.0,  0.2, -0.1],
        [ 0.1, -0.3,  1.1,  0.0],
        [ 0.0,  0.2, -0.1,  1.3]
    ])

    dy = W @ z

    scale = float(realization)

    def clip(x): return float(np.tanh(x))

    return {
        "thermodynamic": {
            "voltage_boost": clip(dy[0]) * scale,
            "stability_shift": clip(dy[3]) * scale
        },
        "kinetic": {
            "reaction_rate_log_delta": clip(dy[1] - 0.5 * dy[2]) * scale
        },
        "transport": {
            "diffusivity_log_delta": clip(dy[2] - 0.3 * dy[1]) * scale
        },
        "structural": {
            "volume_expansion_coeff": clip(dy[1]) * scale
        }
    }
