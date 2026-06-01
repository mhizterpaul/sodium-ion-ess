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

    # --- normalized physics residuals ---
    dv = (safe(proxy_props.get("volume_per_atom"))
          - safe(base_props.get("volume_per_atom"))) / (safe(base_props.get("volume_per_atom")) + 1e-9)

    de = (safe(proxy_props.get("band_gap"))
          - safe(base_props.get("band_gap"))) / (safe(base_props.get("band_gap")) + 1e-6)

    df = safe(proxy_props.get("formation_energy")) - safe(base_props.get("formation_energy"))

    # --- quadratic physics penalty (stable form) ---
    phys_penalty = -(1.0 * dv**2 + 0.7 * de**2 + 0.3 * df**2)

    # --- LOGIT fusion (key fix) ---
    a, b = 2.0, 3.0
    z = a * (r_chem - 0.5) + b * phys_penalty

    # sigmoid with numerical stability
    z = np.clip(z, -20, 20)
    return float(1.0 / (1.0 + np.exp(-z)))

# Calibrated Projection Matrix M (Identity baseline for identifiability)
M_PROJECTION = np.eye(4, dtype=float)

# Latent Physics Metric Gz (Curvature preference in physics directions)
# Weights: Energy(10), Volume(5), Bandgap(2), Stability(1)
GZ_METRIC = np.diag([10.0, 5.0, 2.0, 1.0])

def derive_coupled_deltas(
    base_props: Dict[str, float],
    proxy_props: Dict[str, float],
    base_v: float,
    realization: float
) -> Dict[str, Dict[str, float]]:
    """
    Physically constrained channel model with proper scaling and clipping.
    """

    # --- normalized latent vector ---
    z = np.array([
        (proxy_props["formation_energy"] - base_props["formation_energy"]) / 2.0,
        (proxy_props["volume_per_atom"] - base_props["volume_per_atom"]) / 10.0,
        (proxy_props["band_gap"] - base_props["band_gap"]) / 3.0,
        (proxy_props["stability"] - base_props["stability"]) / 1.0
    ])

    dy = M_PROJECTION @ z

    # helper clamp (critical for DFN stability)
    def clip(x, lim=5.0):
        return float(np.clip(x, -lim, lim))

    return {
        "thermodynamic": {
            "voltage_boost": clip(-dy[0]) * realization * 0.05,  # << scaled properly
            "stability_shift": clip(dy[3]) * realization
        },
        "kinetic": {
            "reaction_rate_log_delta": clip(0.1 * dy[0] - 0.2 * dy[2]) * realization
        },
        "transport": {
            "diffusivity_log_delta": clip(1.0 * dy[1] - 0.3 * dy[2]) * realization
        },
        "structural": {
            "volume_expansion_coeff": clip(dy[1]) * realization
        }
    }
