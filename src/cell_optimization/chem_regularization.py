import re
import math
from typing import Dict, Set

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
    Computes a realization factor [0, 1] based on chemical, structural,
    and electronic similarity between a base material and a proxy.
    """
    # 1. Chemical Similarity (Elemental Overlap)
    e_base = parse_formula(base_formula)
    e_proxy = parse_formula(proxy_formula)
    r_chem = len(e_base & e_proxy) / len(e_base | e_proxy) if (e_base | e_proxy) else 0.0

    # 2. Structural Similarity (Volume-per-atom mismatch)
    v_b = base_props["volume_per_atom"]
    v_p = proxy_props["volume_per_atom"]
    # Handle zero volume case just in case
    r_struct = math.exp(-abs(v_p - v_b) / (v_b + 1e-9))

    # 3. Electronic Similarity (Bandgap mismatch)
    eg_b = base_props["band_gap"]
    eg_p = proxy_props["band_gap"]
    epsilon = 1e-6
    r_electronic = math.exp(-abs(eg_p - eg_b) / (eg_b + epsilon))

    return r_chem * r_struct * r_electronic

def derive_coupled_deltas(base_props: Dict[str, float],
                          proxy_props: Dict[str, float],
                          base_v: float,
                          realization: float) -> Dict[str, float]:
    """
    Derives correlated performance perturbations using a reduced-order latent descriptor.
    """
    # de_diff: difference in formation energy per atom
    de_diff = proxy_props["formation_energy"] - base_props["formation_energy"]

    # vol_ratio: relative volume change
    vol_ratio = proxy_props["volume_per_atom"] / (base_props["volume_per_atom"] + 1e-9)

    # eg_ratio: relative bandgap change
    eg_b = base_props["band_gap"]
    eg_p = proxy_props["band_gap"]
    eg_ratio = eg_p / (eg_b + 1e-6)

    # Latent descriptor z = w1*dEf + w2*vol_ratio + w3*bandgap_ratio
    # Weights selected based on physical heuristics for polyanionic cathodes
    w1, w2, w3 = -0.5, 0.3, -0.2
    z = w1 * de_diff + w2 * (vol_ratio - 1.0) + w3 * (eg_ratio - 1.0)

    # Performance projections
    # a1 for voltage boost, a2 for diffusivity exponent
    a1 = 0.15 * (base_v / 3.2)
    a2 = 0.8

    voltage_boost = a1 * z * realization
    # Diffusivity follows exponential scaling: D/D0 = exp(a2 * z * realization)
    diffusivity_mult = math.exp(a2 * z * realization)

    # Clamping for physical consistency
    return {
        "voltage_boost": float(voltage_boost),
        "diffusivity_mult": float(max(0.1, min(10.0, diffusivity_mult)))
    }
