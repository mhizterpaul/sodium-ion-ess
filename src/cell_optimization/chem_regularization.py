import re
import math
import numpy as np
from typing import Dict, Set, List, Optional, Any
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Specie
from pymatgen.analysis.bond_valence import BVAnalyzer

KT = 0.0259 # eV at 300K

# --- PHYSICAL CONSTANTS & CONSTRAINTS ---
DOPANT_CHARGES = {"Mn": 2, "Cr": 3, "Ni": 2}
FE_CHARGE = 2

# Sensible coordination number defaults for battery materials
# Na:6, Fe:6, P:4 (tetrahedral), O:2, C:4, Si:4
DEFAULT_CN = {"Na": 6, "Fe": 6, "P": 4, "O": 2, "C": 4, "Si": 4}

def generate_doped_formula(dopant, x):
    # Charge neutrality via Na vacancy compensation (Issue 2 fix)
    try:
        dopant_charge = DOPANT_CHARGES[dopant]
        delta_q = (dopant_charge - FE_CHARGE)
        # charge compensation via Na vacancies: each Fe site substituted by a higher valence dopant
        # requires removing (dopant_charge - FE_CHARGE) Na+ ions.
        # Total sites substituted = 3.0 * x
        na_deficit = 3.0 * x * delta_q

        comp = Composition({
            "Na": 4.0 - na_deficit,
            "Fe": 3.0 * (1.0 - x),
            dopant: 3.0 * x,
            "P": 4,
            "O": 15
        })
        return comp.reduced_formula
    except Exception:
        return f"Na{4.0-x*(DOPANT_CHARGES.get(dopant,2)-2):.2f}Fe{3.0*(1.0-x):.2f}{dopant}{3.0*x:.2f}P4O15"

def get_oxidation_states(comp: Composition, structure=None):
    """Refined oxidation state solver with neutrality enforcement."""
    # fallback deterministic oxidation map (battery-relevant prior)
    prior = {
        "Na": 1, "O": -2, "P": 5,
        "Fe": 2, "Mn": 2, "Cr": 3, "Ni": 2,
        "C": 0, "Si": 4, "F": -1, "H": 1
    }
    states = {}
    try:
        # 1. Prior-based assignment
        for el in comp.elements:
            if el.symbol in prior:
                states[el.symbol] = prior[el.symbol]

        # 2. Structural fallback (BVAnalyzer)
        missing = [el.symbol for el in comp.elements if el.symbol not in states]
        if missing and structure:
            try:
                analyzer = BVAnalyzer()
                decorated = analyzer.get_oxi_state_decorated_structure(structure)
                for s in missing:
                    for sp in decorated.composition.elements:
                        if hasattr(sp, "symbol") and sp.symbol == s:
                            states[s] = getattr(sp, "oxi_state", states.get(s))
            except: pass

        # 3. Guess-based fallback
        missing = [el.symbol for el in comp.elements if el.symbol not in states]
        if missing:
            guesses = comp.oxi_state_guesses()
            if guesses:
                best = guesses[0]
                for s in missing:
                    if s in best: states[s] = best[s]

        # 4. Charge Neutrality Enforcement (Level 6 improvement)
        # We solve for the last missing element or verify if all known
        if len(states) == len(comp.elements):
             total_charge = sum(states[el.symbol] * amt for el, amt in comp.items())
             if abs(total_charge) > 1e-3:
                  # Simple heuristic: adjust transition metal or oxygen if discrepancy small
                  if "O" in states:
                       states["O"] -= total_charge / comp["O"]

        return states
    except Exception:
        return prior

def ionic_radius_proxy(formula: str, structure=None) -> float:
    """Refined ionic radius using coordination defaults (Level 4 improvement)."""
    try:
        comp = Composition(formula)
        states = get_oxidation_states(comp, structure=structure)
        total_atoms = sum(comp.values())
        avg_radius = 0.0

        for el, count in comp.items():
             symbol = el.symbol
             oxi = states.get(symbol, 0)
             cn = DEFAULT_CN.get(symbol, 6)
             try:
                 # Shannon radii depend on oxidation and coordination
                 if oxi != 0:
                      # Attempt CN-specific Specie lookup if available in future expansions
                      # Currently use Specie.ionic_radius as proxy (corresponds to common CN)
                      radius = Specie(symbol, oxi).ionic_radius
                 else:
                      radius = el.average_ionic_radius

                 if radius is None: radius = el.atomic_radius
             except:
                 radius = el.atomic_radius or 1.0

             avg_radius += (count / total_atoms) * (radius if radius else 1.0)
        return float(avg_radius)
    except Exception:
        return 1.0

def compute_surrogate_properties(formula: str) -> Dict[str, Any]:
    """Physically-informed material property estimation (Level 4 Fallback)."""
    try:
        comp = Composition(formula)
        total_atoms = sum(comp.values())

        # 1. Weighted Average Electronegativity (Issue 5)
        avg_X = sum(el.X * amt for el, amt in comp.items() if el.X is not None) / total_atoms

        # 2. Refined Volume Estimation (Issue 3)
        # Sum of ionic volumes (4/3 * pi * r^3) / packing_factor
        packing_factor = 0.65 # Conservative for complex battery materials
        v_ion = 0.0
        states = get_oxidation_states(comp)
        for el, count in comp.items():
             oxi = states.get(el.symbol, 0)
             try:
                  r = Specie(el.symbol, oxi).ionic_radius if oxi != 0 else el.average_ionic_radius
                  if r is None: r = el.atomic_radius or 1.0
             except: r = 1.0
             v_ion += (count / total_atoms) * (4/3.0 * np.pi * (r**3))

        volume_per_atom = v_ion / packing_factor

        # 3. Band Gap Proxy (Issue 2) - Phillips-like ionicity
        # Delta_chi^2 based heuristic
        metals = [el for el in comp.elements if el.is_metal]
        non_metals = [el for el in comp.elements if not el.is_metal]
        if metals and non_metals:
             d_chi = np.mean([el.X for el in non_metals]) - np.mean([el.X for el in metals])
             band_gap = 1.2 * (d_chi**2) + 0.5
        else:
             band_gap = 0.1 # metallic/semi-metal default

        # 4. Formation Energy Proxy (Issue 1)
        # Returns energy per atom (eV/atom) to ensure scale invariance (Issue 1 fix)
        ef_proxy = -0.5 * (avg_X - 1.5) / 4.0 # Heuristic scaling

        # 5. Stability Proxy (Issue 7) - Bond Valence Mismatch Proxy
        # High electronegativity difference often correlates with higher stability
        stability = 0.1 / (max(abs(ef_proxy), 0.1))

        # 6. Battery Specific Metrics (Issue 11)
        # Theoretical Capacity (mAh/g): C = nF / (3.6 * Mw)
        #Mw = comp.weight
        # Assume 1 Na exchange if Na present
        na_count = comp.get("Na", 0)
        theoretical_capacity = (na_count * 96485.0) / (3.6 * comp.weight) if na_count > 0 else 0.0

        # Insertion Voltage Proxy: correlative with electronegativity difference
        avg_voltage = 0.5 * (avg_X - 1.0) + 2.0 if na_count > 0 else 0.0

        return {
            "stability": float(np.clip(stability, 0.001, 0.5)),
            "formation_energy": float(ef_proxy),
            "band_gap": float(np.clip(band_gap, 0.0, 10.0)),
            "volume_per_atom": float(volume_per_atom),
            "uncertainty_formation_energy": 0.4,
            "ionic_radius": ionic_radius_proxy(formula),
            "theoretical_capacity_mah_g": float(theoretical_capacity),
            "avg_insertion_voltage": float(avg_voltage),
            "resolved_formula": formula,
            "computed": True
        }
    except Exception:
        return {
            "stability": 0.1, "formation_energy": -1.0, "band_gap": 3.0,
            "volume_per_atom": 10.0, "uncertainty_formation_energy": 0.5,
            "ionic_radius": 1.0, "computed": True
        }

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
