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
from pymatgen.core import Composition

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
    # Normalized site-fraction substitution on Fe site (Issue 1)
    try:
        comp = Composition({
            "Na": 4,
            "Fe": 3.0 * (1.0 - x),
            dopant: 3.0 * x,
            "P": 4,
            "O": 15
        })
        return comp.reduced_formula
    except Exception:
        return f"Na4Fe{3.0*(1.0-x):.2f}{dopant}{3.0*x:.2f}P4O15"

def enforce_charge_balance(dopant_charge, x, fe_charge=2):
    # Charge neutrality enforcement (Issue 2)
    delta = x * (dopant_charge - fe_charge)
    return abs(delta) < 0.1

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
        # Canonicalize formula for cache safety (Issue 9)
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
            docs = sorted(docs, key=lambda d: d.energy_above_hull)[:5]

            # Boltzmann weighting tied to KT (Issue 8)
            T_eff = KT
            e_hull = np.array([float(d.energy_above_hull) for d in docs])
            weights = np.exp(-e_hull / T_eff)
            weights /= np.sum(weights)

            p = {
                "stability": float(np.sum(weights * e_hull)),
                "formation_energy": float(np.sum(weights * np.array([float(d.formation_energy_per_atom) for d in docs]))),
                "band_gap": float(np.sum(weights * np.array([float(d.band_gap if d.band_gap is not None else 0.0) for d in docs]))),
                "volume_per_atom": float(np.sum(weights * np.array([float(d.volume / d.nsites if d.nsites else 1.0) for d in docs]))),
                "uncertainty_formation_energy": float(np.std([float(d.formation_energy_per_atom) for d in docs])),
                "resolved_formula": docs[0].formula_pretty
            }
            return p, "MATERIALS_PROJECT", p["resolved_formula"]

        if MPRester and self.mp_key and (source_override == "MP" or source_override is None):
            try:
                with MPRester(api_key=self.mp_key) as mpr:
                    if chemsys:
                        docs = mpr.materials.summary.search(chemsys=chemsys, fields=['formula_pretty', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites'])
                    else:
                        docs = mpr.materials.summary.search(formula=canonical_formula, fields=['formula_pretty', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites'])

                    if docs:
                        props, source, resolved_formula = process_docs(docs)
            except Exception as e:
                 logging.exception(f"MP resolution failed for {canonical_formula}: {e}")

        if props is None and self.session and (source_override == "OQMD" or source_override is None):
            try:
                params = {"composition": canonical_formula, "limit": 5, "fields": "composition,delta_e,stability,band_gap,volume,natoms"}
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
            except Exception as e:
                 logging.exception(f"OQMD resolution failed for {canonical_formula}: {e}")

        if props:
            self.cache[cache_key] = {"props": props, "source": source, "formula": resolved_formula}
            self._save_cache()
            return props, source, resolved_formula
        return None, "NONE", canonical_formula

    def run(self) -> Tuple[Dict[MaterialCategory, List[MaterialCandidate]], Dict[str, Any]]:
        if self._run_result is not None:
             return self._run_result

        print(f"Executing API-Based Material Resolution (Layer 1)...")
        system = {cat: [] for cat in MaterialCategory}
        bases = {}
        seen = set() # Issue 10

        # Explicit Resolution Policy (Issue 3)
        SOURCES = ["MP", "OQMD"]

        for f in BASE_CATHODE_PRIORITIES:
            for src_name in SOURCES:
                p, src, rf = self._resolve_material(f, source_override=src_name)
                if p:
                    bases["cathode"] = {"formula": rf, "properties": p, "source": src}
                    break
            if "cathode" in bases: break

        for src_name in SOURCES:
            p, src, rf = self._resolve_material(BASE_SALT_FORMULA, source_override=src_name)
            if p:
                bases["salt"] = {"formula": rf, "properties": p, "source": src}
                break

        for src_name in SOURCES:
            p, src, rf = self._resolve_material(BASE_INTERFACE_FORMULA, source_override=src_name)
            if p:
                bases["interface"] = {"formula": rf, "properties": p, "source": src}
                break

        if not all(k in bases for k in ["cathode", "salt", "interface"]):
            return system, {}

        for d in DOPANTS:
            # Issue 2: Charge balance before resolving
            for x in [0.05, 0.1, 0.15]:
                if not enforce_charge_balance(DOPANT_CHARGES[d], x): continue
                formula = generate_doped_formula(d, x)
                chemsys = f"Na-Fe-{d}-P-O"

                for src_name in SOURCES:
                    p, src, rf = self._resolve_material(formula=formula, chemsys=chemsys, source_override=src_name)
                    if p:
                        if rf not in seen:
                            system[MaterialCategory.CATHODE_DOPANT].append(MaterialCandidate(name=f"{d}-doped-x{x}", category=MaterialCategory.CATHODE_DOPANT, composition=rf, properties=p, provenance=src))
                            seen.add(rf)
                        break
                if f"{d}-doped-x{x}" in [c.name for c in system[MaterialCategory.CATHODE_DOPANT]]: break

        for name, formula in SALTS.items():
            for src_name in SOURCES:
                p, src, rf = self._resolve_material(formula, source_override=src_name)
                if p:
                    if rf not in seen:
                        system[MaterialCategory.SALT].append(MaterialCandidate(name=name, category=MaterialCategory.SALT, composition=rf, properties=p, provenance=src))
                        seen.add(rf)
                    break

        for name, formula in FUNCTIONALIZATION.items():
            for src_name in SOURCES:
                p, src, rf = self._resolve_material(formula, source_override=src_name)
                if p:
                    if rf not in seen:
                        system[MaterialCategory.FUNCTIONALIZATION].append(MaterialCandidate(name=name, category=MaterialCategory.FUNCTIONALIZATION, composition=rf, properties=p, provenance=src))
                        seen.add(rf)
                    break

        self._run_result = (system, bases)
        return self._run_result
