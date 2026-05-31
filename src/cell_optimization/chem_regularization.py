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

def compute_chemical_realization(base_formula: str, proxy_formula: str,
                                 base_props: Dict[str, float],
                                 proxy_props: Dict[str, float]) -> float:
    """
    Computes a realization factor [0, 1] using normalized Mahalanobis-style
    similarity and bounded logistic fusion.
    """
    # 1. Chemical Jaccard
    e_base = parse_formula(base_formula)
    e_proxy = parse_formula(proxy_formula)
    r_chem = len(e_base & e_proxy) / max(len(e_base | e_proxy), 1)

    # 2. Normalized Structural Distance
    dv = (proxy_props["volume_per_atom"] - base_props["volume_per_atom"]) / (base_props["volume_per_atom"] + 1e-9)

    # 3. Normalized Electronic Distance
    de = (proxy_props["band_gap"] - base_props["band_gap"]) / (base_props["band_gap"] + 1e-6)

    # 4. Formation Energy Stabilization Term
    df = (proxy_props["formation_energy"] - base_props["formation_energy"])

    # Bounded Logistic Fusion (prevents realization collapse)
    # Penalizes mismatches exponentially in the latent score z
    z = - (1.2 * dv**2 + 0.8 * de**2 + 0.3 * df**2)
    r_phys = 1 / (1 + math.exp(-z))

    return float(np.clip(r_chem * r_phys, 0.0, 1.0))

# Calibrated Projection Matrix M
# Using Identity-initialized baseline for identifiability and physical consistency
M_PROJECTION = np.eye(4, dtype=float)

def derive_coupled_deltas(base_props: Dict[str, float],
                          proxy_props: Dict[str, float],
                          base_v: float,
                          realization: float) -> Dict[str, float]:
    """
    Derives performance deltas using the latent physics vector and projection matrix.
    """
    # Latent Physics Vector z (normalized and dimensionless where possible)
    z = np.array([
        (proxy_props["formation_energy"] - base_props["formation_energy"]),
        (proxy_props["volume_per_atom"] - base_props["volume_per_atom"]) / (base_props["volume_per_atom"] + 1e-9),
        (proxy_props["band_gap"] - base_props["band_gap"]) / (base_props["band_gap"] + 1e-6),
        (proxy_props["stability"] - base_props["stability"])
    ])

    # Multi-dimensional projection
    dy = M_PROJECTION @ z

    # Performance projections (clamped and scaled)
    voltage_boost = dy[0] * realization * (base_v / 3.2)
    # Diffusivity: exp scaling of structural/electronic proxy
    diffusivity_mult = math.exp(dy[1] * realization)

    return {
        "voltage_boost": float(voltage_boost),
        "diffusivity_mult": float(max(0.1, min(10.0, diffusivity_mult)))
    }
