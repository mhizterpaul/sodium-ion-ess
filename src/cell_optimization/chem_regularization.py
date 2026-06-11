import re
import math
import numpy as np
from typing import Dict, Set, List, Optional, Any

KT = 0.0259 # eV at 300K

def parse_stoich(formula: str) -> Dict[str, float]:
    """
    Parses a chemical formula into element counts.
    Example: 'Na4Fe3P4O15' -> {'Na': 4.0, 'Fe': 3.0, 'P': 4.0, 'O': 15.0}
    """
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
    """
    Physically grounded connectivity-based scaling for organosiloxanes (MTMS)
    derived from network solids (SiO2).
    P_mtms = phi^alpha * P_sio2
    """
    scaled = props.copy()
    # Connectivity exponents
    a_density = 1.0
    a_modulus = 2.5 # Not directly used in PyBaMM but good for completeness
    a_transport = 1.5

    # Scaling properties
    # Density/Volume: Lower connectivity -> higher volume per atom
    if "volume_per_atom" in scaled:
        scaled["volume_per_atom"] = scaled["volume_per_atom"] / (phi**a_density)

    # Energy: scaling formation energy by connectivity
    if "formation_energy" in scaled:
        scaled["formation_energy"] = scaled["formation_energy"] * phi

    # Stability: lower connectivity typically reduces stability (higher energy above hull)
    if "stability" in scaled:
        scaled["stability"] = scaled["stability"] / (phi + 1e-9)

    return scaled

def compute_chemical_realization(
    base_formula: str,
    proxy_formula: str,
    base_props: Dict[str, float],
    proxy_props: Dict[str, float]
) -> float:
    """
    How safely can this proxy perturb the base material?
    Uses stoichiometry, chemical overlap, and physics residuals.
    """
    def safe(x, ref=0.0):
        try: return float(x)
        except: return ref

    # --- Stoichiometry and chemical overlap ---
    base_s = stoich_norm(parse_stoich(base_formula))
    proxy_s = stoich_norm(parse_stoich(proxy_formula))

    stoich_penalty = stoich_distance(base_s, proxy_s)

    e_base = set(base_s.keys())
    e_proxy = set(proxy_s.keys())
    r_chem = len(e_base & e_proxy) / max(len(e_base | e_proxy), 1)

    # --- Physical residuals ---
    dE = thermo_norm(safe(proxy_props.get("formation_energy")), safe(base_props.get("formation_energy")))
    dV = (safe(proxy_props.get("volume_per_atom")) - safe(base_props.get("volume_per_atom"))) / (safe(base_props.get("volume_per_atom")) + 1e-9)

    # Realization equation: higher is better
    z = (
        3.0 * r_chem
        - 1.5 * abs(dE)
        - 1.0 * abs(dV)
        - 2.0 * stoich_penalty
    )

    z = np.clip(z, -10, 10)
    return float(1.0 / (1.0 + np.exp(-z)))

def derive_coupled_deltas(
    base_props: Dict[str, float],
    proxy_props: Dict[str, float],
    is_network: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Universal physics transformation layer.
    Converts raw material property differences into physical performance deltas.
    """
    dE = thermo_norm(proxy_props["formation_energy"], base_props["formation_energy"])
    dG = (proxy_props["band_gap"] - base_props["band_gap"]) / 1.0
    dV = geom_norm(proxy_props, base_props)["strain"]
    dS = (base_props["stability"] - proxy_props["stability"]) / 0.2

    # Physics coupling rules

    # 1. Energetic/Thermodynamic
    voltage_boost = -0.01 * dE # Small correction
    stability_shift = dS
    initial_loss_mult = math.exp(np.clip(0.2 * dS, -5, 5))

    # 2. Kinetic (Arrhenius-derived)
    activation_delta = 0.2 * dV + 0.1 * dG

    # For network solids (like MTMS-derived), we apply a further connectivity-based attenuation
    # if not already handled by input properties.
    diffusivity_log_delta = -activation_delta / (KT + 1e-9)

    reaction_rate_log_delta = 0.1 * dE - 0.3 * dG
    sei_growth_mult = math.exp(np.clip(0.5 * dE - 0.2 * dS, -5, 5))
    negative_exchange_log_delta = 0.4 * dS - 0.1 * dG

    # 3. Transport/Secondary
    transport_log_delta = -0.5 * dE + 0.2 * dV
    interfacial_log_delta = -0.8 * dS + 0.3 * dG

    def clip_log(x):
        return float(np.clip(x, -5, 5))

    return {
        "thermodynamic": {
            "voltage_boost": float(voltage_boost),
            "stability_shift": float(stability_shift),
            "initial_loss_mult": float(initial_loss_mult)
        },
        "kinetic": {
            "reaction_rate_log_delta": clip_log(reaction_rate_log_delta),
            "sei_growth_mult": float(sei_growth_mult),
            "negative_exchange_log_delta": clip_log(negative_exchange_log_delta)
        },
        "transport": {
            "diffusivity_log_delta": clip_log(diffusivity_log_delta),
            "conductivity_mult": float(math.exp(clip_log(transport_log_delta))),
            "ion_transference_mult": 1.0 + 0.1 * float(np.tanh(transport_log_delta)),
            "resistance_drift_mult": float(math.exp(clip_log(interfacial_log_delta)))
        },
        "structural": {
            "volume_expansion_coeff": float(dV)
        }
    }
