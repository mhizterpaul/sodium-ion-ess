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
    z = - (1.2 * dv**2 + 0.8 * de**2 + 0.3 * df**2)
    r_phys = 1 / (1 + math.exp(-z))

    return float(np.clip(r_chem * r_phys, 0.0, 1.0))

# Calibrated Projection Matrix M (Identity baseline for identifiability)
M_PROJECTION = np.eye(4, dtype=float)

# Latent Physics Metric Gz (Curvature preference in physics directions)
# Weights: Energy(10), Volume(5), Bandgap(2), Stability(1)
GZ_METRIC = np.diag([10.0, 5.0, 2.0, 1.0])

def derive_coupled_deltas(base_props: Dict[str, float],
                          proxy_props: Dict[str, float],
                          base_v: float,
                          realization: float) -> Dict[str, Dict[str, float]]:
    """
    Derives performance deltas using typed parameter channels and dimensionless
    latent physics vector.

    Returns:
        Dict organized by channels: thermodynamic, kinetic, transport, structural.
    """
    # Dimensionless Latent Physics Vector z
    # Normalized by characteristic scales: Ef~2eV, V~10A^3, Eg~3eV, Stab~1eV
    z = np.array([
        (proxy_props["formation_energy"] - base_props["formation_energy"]) / 2.0,
        (proxy_props["volume_per_atom"] - base_props["volume_per_atom"]) / 10.0,
        (proxy_props["band_gap"] - base_props["band_gap"]) / 3.0,
        (proxy_props["stability"] - base_props["stability"]) / 1.0
    ])

    # Multi-dimensional projection
    dy = M_PROJECTION @ z

    # Map projected physics to typed channels with consistent scaling logic
    # thermodynamic: additive (theta' = theta + beta*delta)
    # kinetic/transport: log-space multiplicative (theta' = theta * exp(alpha*delta))

    channels = {
        "thermodynamic": {
            # Voltage shift: V ~ -dEf (Thermodynamic consistency)
            "voltage_boost": -dy[0] * realization * (base_v / 3.2),
            "stability_shift": dy[3] * realization
        },
        "kinetic": {
            # ln(i0/i0_ref) proportional to dEf/dEg mismatch
    # Clipped to prevent exponential explosion
    "reaction_rate_log_delta": np.clip((0.1 * dy[0] - 0.2 * dy[2]) * realization, -5, 5)
        },
        "transport": {
            # ln(D/D0) proportional to volume and electronic distances
    "diffusivity_log_delta": np.clip((1.2 * dy[1] - 0.5 * dy[2]) * realization, -5, 5)
        },
        "structural": {
            "volume_expansion_coeff": dy[1] * realization
        }
    }

    return channels
