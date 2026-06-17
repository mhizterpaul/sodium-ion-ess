import json
import os
import re
import math
import numpy as np
import logging
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from pymatgen.core import Composition, Element
from pymatgen.core.periodic_table import Specie
from pymatgen.analysis.bond_valence import BVAnalyzer

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None

try:
    from mp_api.client import MPRester
except ImportError:
    MPRester = None

from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

KT = 0.0259 # eV at 300K

class MaterialCategory(Enum):
    CATHODE_DOPANT = auto()
    SALT = auto()
    FUNCTIONALIZATION = auto()

# --- FORMULAS FROM PAPER.MD ---
BASE_CATHODE_PRIORITIES = ["Na4Fe3(PO4)2P2O7", "Na2FeP2O7", "NaFeP2O7"]
BASE_SALT_FORMULA = "NaPF6"
BASE_INTERFACE_FORMULA = "C2H4O"
DOPANTS = ["Mn", "Cr", "Ni"]
DOPANT_CHARGES = {"Mn": 2, "Cr": 3, "Ni": 2}
FE_CHARGE = 2

# New salt/functionalization candidates from paper.md
SALTS = {
    "NaBOB": "NaBC4O8",
    "NaTCP": "NaC4N3"
}
FUNCTIONALIZATION = {
    "MTMS": "C4H12O3Si"
}

OQMD_URL = "https://oqmd.org/oqmdapi/formationenergy"
CACHE_FILE = "material_cache.json"

@dataclass
class MaterialCandidate:
    name: str
    category: MaterialCategory
    composition: str
    properties: Dict[str, Any]
    provenance: str = "OQMD"

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
    # fallback deterministic oxidation map (battery-relevant prior)
    prior = {
        "Na": 1, "O": -2, "P": 5,
        "Fe": 2, "Mn": 2, "Cr": 3, "Ni": 2,
        "C": 0, "Si": 4, "F": -1
    }
    states = {}
    try:
        # Always prioritize deterministic prior table first (Key constraint)
        for el in comp.elements:
            symbol = el.symbol
            if symbol in prior:
                states[symbol] = prior[symbol]

        # If any elements missing from prior, use structure-based BVAnalyzer or guesses
        missing_symbols = [el.symbol for el in comp.elements if el.symbol not in states]
        if missing_symbols:
            if structure:
                try:
                    analyzer = BVAnalyzer()
                    decorated = analyzer.get_oxi_state_decorated_structure(structure)
                    for s in missing_symbols:
                        amt_dict = decorated.composition.get_el_amt_dict()
                        if s in amt_dict:
                            # Summing amounts might be wrong if there are multiple sites with same element but different oxi states
                            # But Specie(symbol, oxi) needs a single oxi. We take the most common one.
                            # Decorated structure elements are usually Specie.
                            for sp in decorated.composition.elements:
                                if hasattr(sp, "symbol") and sp.symbol == s:
                                    states[s] = getattr(sp, "oxi_state", states.get(s))
                except Exception:
                    pass

            # Secondary fallback: guesses
            if any(s not in states for s in missing_symbols):
                guesses = comp.oxi_state_guesses()
                if guesses:
                    best_guess = guesses[0]
                    for s in missing_symbols:
                        if s not in states and s in best_guess:
                            states[s] = best_guess[s]
        return states
    except Exception:
        return prior

def ionic_radius_proxy(formula: str, structure=None) -> float:
    """Computes a weighted average ionic radius for the composition using Specie data."""
    try:
        comp = Composition(formula)
        states = get_oxidation_states(comp, structure=structure)

        total_atoms = sum(comp.values())
        avg_radius = 0.0
        for el, count in comp.items():
             symbol = el.symbol
             oxi = states.get(symbol, 0)
             try:
                 # Deterministic radius selection logic (Issue 2.3)
                 radius = el.average_ionic_radius if oxi == 0 else Specie(symbol, oxi).ionic_radius
                 if radius is None:
                     radius = el.atomic_radius
             except Exception:
                 radius = el.atomic_radius or 1.0
             avg_radius += (count / total_atoms) * (radius if radius else 1.0)
        return float(avg_radius)
    except Exception:
        return 1.0

class MaterialMappingEngine:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.mp_key = os.environ.get("MP_API_KEY")
        self.cache = self._load_cache()
        self.session = self._setup_session() if requests else None
        self.base_params = get_parameter_values()
        self._run_result = None

    def _setup_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            logging.warning(f"Failed to save cache: {e}")

    def _resolve_material(self, formula: str, source_override: Optional[str] = None, chemsys: Optional[str] = None) -> Tuple[Optional[Dict[str, float]], str, str]:
        try:
             canonical_formula = Composition(formula).reduced_formula
        except Exception:
             canonical_formula = formula

        cache_key = hashlib.md5(f"{canonical_formula}|{source_override}|{chemsys}".encode()).hexdigest()
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            return entry["props"], entry.get("source", "UNKNOWN"), entry.get("formula", canonical_formula)

        props, source, resolved_formula = None, "NONE", canonical_formula

        def process_docs(docs):
            if not docs: return None, "NONE", canonical_formula
            ALPHA = 20.0
            filtered_docs = [d for d in docs if float(d.energy_above_hull) <= 0.1]
            if not filtered_docs: return None, "NONE", canonical_formula
            stabilities = np.array([float(d.energy_above_hull) for d in filtered_docs])
            weights = np.exp(-ALPHA * stabilities)
            weights /= np.sum(weights)
            best_formula = filtered_docs[0].formula_pretty
            best_structure = getattr(filtered_docs[0], "structure", None)
            p = {
                "stability": float(np.sum(weights * stabilities)),
                "formation_energy": float(np.sum(weights * np.array([float(d.formation_energy_per_atom) for d in filtered_docs]))),
                "band_gap": float(np.sum(weights * np.array([float(d.band_gap if d.band_gap is not None else 0.0) for d in filtered_docs]))),
                "volume_per_atom": float(np.sum(weights * np.array([float(d.volume / d.nsites if d.nsites else 1.0) for d in filtered_docs]))),
                "uncertainty_formation_energy": float(np.std([float(d.formation_energy_per_atom) for d in filtered_docs])),
                "ionic_radius": ionic_radius_proxy(best_formula, structure=best_structure),
                "resolved_formula": best_formula
            }
            return p, "MATERIALS_PROJECT", p["resolved_formula"]

        if MPRester and self.mp_key and (source_override == "MP" or source_override is None):
            try:
                with MPRester(api_key=self.mp_key) as mpr:
                    if chemsys:
                        docs = mpr.materials.summary.search(chemsys=chemsys, fields=['formula_pretty', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites', 'structure'])
                    else:
                        docs = mpr.materials.summary.search(formula=canonical_formula, fields=['formula_pretty', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites', 'structure'])
                    if docs: props, source, resolved_formula = process_docs(docs)
            except Exception as e: logging.exception(f"MP resolution failed: {e}")

        if props is None and self.session and (source_override == "OQMD" or source_override is None):
            try:
                params = {"composition": canonical_formula, "limit": 10, "fields": "composition,delta_e,stability,band_gap,volume,natoms"}
                r = self.session.get(OQMD_URL, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    if data:
                        class Doc:
                            def __init__(self, d):
                                self.energy_above_hull = d.get("stability", 0.1)
                                self.formation_energy_per_atom = d.get("delta_e", 0.0)
                                self.band_gap = d.get("band_gap", 0.0)
                                self.volume = d.get("volume", 1.0)
                                self.nsites = d.get("natoms", 1.0)
                                self.formula_pretty = d.get("composition", canonical_formula)
                        docs = [Doc(d) for d in data]
                        props, source, resolved_formula = process_docs(docs)
                        source = "OQMD"
            except Exception as e: logging.exception(f"OQMD resolution failed: {e}")

        if props:
            self.cache[cache_key] = {"props": props, "source": source, "formula": resolved_formula}
            self._save_cache()
            return props, source, resolved_formula
        return None, "NONE", canonical_formula

    def run(self) -> Tuple[Dict[MaterialCategory, List[MaterialCandidate]], Dict[str, Any]]:
        if self._run_result is not None: return self._run_result
        print(f"Executing API-Based Material Resolution (Layer 1)...")
        system = {cat: [] for cat in MaterialCategory}
        bases, seen = {}, set()
        SOURCES = ["MP", "OQMD"]
        for f in BASE_CATHODE_PRIORITIES:
            for src_name in SOURCES:
                p, src, rf = self._resolve_material(f, source_override=src_name)
                if p: bases["cathode"] = {"formula": rf, "properties": p, "source": src}; break
            if "cathode" in bases: break
        for src_name in SOURCES:
            p, src, rf = self._resolve_material(BASE_SALT_FORMULA, source_override=src_name)
            if p: bases["salt"] = {"formula": rf, "properties": p, "source": src}; break
        for src_name in SOURCES:
            p, src, rf = self._resolve_material(BASE_INTERFACE_FORMULA, source_override=src_name)
            if p: bases["interface"] = {"formula": rf, "properties": p, "source": src}; break
        if not all(k in bases for k in ["cathode", "salt", "interface"]): return system, {}
        for d in DOPANTS:
            for x in [0.05, 0.1, 0.15]:
                formula = generate_doped_formula(d, x)
                chemsys = f"Na-Fe-{d}-P-O"
                for src_name in SOURCES:
                    p, src, rf = self._resolve_material(formula=formula, chemsys=chemsys, source_override=src_name)
                    if p and rf not in seen:
                        system[MaterialCategory.CATHODE_DOPANT].append(MaterialCandidate(name=f"{d}-doped-x{x}", category=MaterialCategory.CATHODE_DOPANT, composition=rf, properties=p, provenance=src))
                        seen.add(rf); break
        for name, formula in SALTS.items():
            for src_name in SOURCES:
                p, src, rf = self._resolve_material(formula, source_override=src_name)
                if p and rf not in seen:
                    system[MaterialCategory.SALT].append(MaterialCandidate(name=name, category=MaterialCategory.SALT, composition=rf, properties=p, provenance=src))
                    seen.add(rf); break
        for name, formula in FUNCTIONALIZATION.items():
            for src_name in SOURCES:
                p, src, rf = self._resolve_material(formula, source_override=src_name)
                if p and rf not in seen:
                    system[MaterialCategory.FUNCTIONALIZATION].append(MaterialCandidate(name=name, category=MaterialCategory.FUNCTIONALIZATION, composition=rf, properties=p, provenance=src))
                    seen.add(rf); break
        self._run_result = (system, bases)
        return self._run_result
