import json
import os
import re
import math
import numpy as np
import hashlib
import traceback
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
from src.cell_optimization.chem_regularization import (
    KT,
    generate_doped_formula,
    get_oxidation_states,
    ionic_radius_proxy,
    compute_surrogate_properties
)

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

class MaterialMappingEngine:
    def __init__(self):
        self.mp_key = os.environ.get("MP_API_KEY")
        self.cache = self._load_cache()
        self.session = self._setup_session() if requests else None
        self.base_params = get_parameter_values()
        self._run_result = None

    def _setup_session(self):
        session = requests.Session()
        # Increased backoff and added read retries to address OQMD instability (Issue 12)
        # We explicitly retry on all sorts of errors including read timeouts.
        retries = Retry(
            total=10,
            backoff_factor=4,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
            allowed_methods=["GET"],
            respect_retry_after_header=True,
            read=5  # Explicitly retry on read timeouts
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        # Added headers to avoid potential blocks
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
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
            print(f"WARNING: Failed to save cache: {e}")

    def _resolve_material(self, formula: str, source_override: Optional[str] = None, chemsys: Optional[str] = None) -> Tuple[Optional[Dict[str, float]], str, str]:
        """Hierarchical material resolution (Level 1-4)."""
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

        # Level 1: MP Exact Formula Search
        if props is None and MPRester and self.mp_key and (source_override == "MP" or source_override is None):
            try:
                with MPRester(api_key=self.mp_key) as mpr:
                    docs = mpr.materials.summary.search(formula=canonical_formula, fields=['formula_pretty', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites', 'structure'])
                    if docs:
                        props, source, resolved_formula = process_docs(docs)
            except Exception as e:
                print(f"ERROR: Level 1 (MP Exact) failed for {canonical_formula}: {e}\n{traceback.format_exc()}")

        # Level 2: OQMD Exact Formula Search
        if props is None and self.session and (source_override == "OQMD" or source_override is None):
            try:
                params = {"composition": canonical_formula, "limit": 5, "format": "json", "fields": "name,delta_e,stability,band_gap,volume,natoms"}
                r = self.session.get(OQMD_URL, params=params, timeout=(15, 60))
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    if data:
                        class Doc:
                            def __init__(self, d):
                                self.energy_above_hull = d.get("stability") if d.get("stability") is not None else 0.1
                                self.formation_energy_per_atom = d.get("delta_e") if d.get("delta_e") is not None else 0.0
                                self.band_gap = d.get("band_gap") if d.get("band_gap") is not None else 0.0
                                self.volume = d.get("volume") if d.get("volume") is not None else 1.0
                                self.nsites = d.get("natoms") if d.get("natoms") is not None else 1.0
                                self.formula_pretty = d.get("name", canonical_formula)
                        docs = [Doc(d) for d in data]
                        props, source, resolved_formula = process_docs(docs)
                        source = "OQMD"
            except Exception as e:
                print(f"ERROR: Level 2 (OQMD Exact) failed for {canonical_formula}: {e}\n{traceback.format_exc()}")

        # Level 3: Materials-System Search (Na-C-H-O etc.)
        if props is None and source_override is None:
            try:
                if not chemsys:
                    chemsys = "-".join(sorted([el.symbol for el in Composition(canonical_formula).elements]))

                # Try MP System
                if MPRester and self.mp_key:
                    with MPRester(api_key=self.mp_key) as mpr:
                        docs = mpr.materials.summary.search(chemsys=chemsys, fields=['formula_pretty', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites', 'structure'])
                        if docs:
                            props, source, resolved_formula = process_docs(docs)

                # Try OQMD System
                if props is None and self.session:
                    params = {"composition": chemsys, "limit": 10, "format": "json", "fields": "name,delta_e,stability,band_gap,volume,natoms"}
                    r = self.session.get(OQMD_URL, params=params, timeout=(15, 120))
                    if r.status_code == 200:
                        data = r.json().get("data", [])
                        if data:
                            class Doc:
                                def __init__(self, d):
                                    self.energy_above_hull = d.get("stability") if d.get("stability") is not None else 0.1
                                    self.formation_energy_per_atom = d.get("delta_e") if d.get("delta_e") is not None else 0.0
                                    self.band_gap = d.get("band_gap") if d.get("band_gap") is not None else 0.0
                                    self.volume = d.get("volume") if d.get("volume") is not None else 1.0
                                    self.nsites = d.get("natoms") if d.get("natoms") is not None else 1.0
                                    self.formula_pretty = d.get("name", canonical_formula)
                            docs = [Doc(d) for d in data]
                            props, source, resolved_formula = process_docs(docs)
                            source = "OQMD_SYSTEM"
            except Exception as e:
                print(f"ERROR: Level 3 (System Search) failed for {chemsys}: {e}\n{traceback.format_exc()}")

        # Level 4 Fallback: Physics-based property estimation
        if props is None and source_override is None:
            props = compute_surrogate_properties(canonical_formula)
            source = "COMPUTED"
            resolved_formula = canonical_formula

        if props:
            self.cache[cache_key] = {"props": props, "source": source, "formula": resolved_formula}
            self._save_cache()
            return props, source, resolved_formula
        return None, "NONE", canonical_formula

    def run(self) -> Tuple[Dict[MaterialCategory, List[MaterialCandidate]], Dict[str, Any]]:
        if self._run_result is not None: return self._run_result
        print(f"Executing Hierarchical Material Resolution (Level 1-4)...")
        system = {cat: [] for cat in MaterialCategory}
        bases, seen = {}, set()

        # Resolve Base Materials
        for f in BASE_CATHODE_PRIORITIES:
            p, src, rf = self._resolve_material(f)
            if p:
                bases["cathode"] = {"formula": rf, "properties": p, "source": src}
                break

        p, src, rf = self._resolve_material(BASE_SALT_FORMULA)
        if p: bases["salt"] = {"formula": rf, "properties": p, "source": src}

        p, src, rf = self._resolve_material(BASE_INTERFACE_FORMULA)
        if p: bases["interface"] = {"formula": rf, "properties": p, "source": src}

        if not all(k in bases for k in ["cathode", "salt", "interface"]):
            missing = [k for k in ["cathode", "salt", "interface"] if k not in bases]
            err_msg = f"Critical Failure: Failed to resolve base materials: {missing}. Aborting pipeline."
            print(f"ERROR: {err_msg}")
            raise RuntimeError(err_msg)

        print(f"INFO: Resolved base materials: {list(bases.keys())}")

        # Resolve Candidates
        for d in DOPANTS:
            for x in [0.05, 0.1, 0.15]:
                formula = generate_doped_formula(d, x)
                chemsys = f"Na-Fe-{d}-P-O"
                p, src, rf = self._resolve_material(formula=formula, chemsys=chemsys)
                if p and rf not in seen:
                    system[MaterialCategory.CATHODE_DOPANT].append(MaterialCandidate(name=f"{d}-doped-x{x}", category=MaterialCategory.CATHODE_DOPANT, composition=rf, properties=p, provenance=src))
                    seen.add(rf)

        for name, formula in SALTS.items():
            p, src, rf = self._resolve_material(formula)
            if p and rf not in seen:
                system[MaterialCategory.SALT].append(MaterialCandidate(name=name, category=MaterialCategory.SALT, composition=rf, properties=p, provenance=src))
                seen.add(rf)

        for name, formula in FUNCTIONALIZATION.items():
            p, src, rf = self._resolve_material(formula)
            if p and rf not in seen:
                system[MaterialCategory.FUNCTIONALIZATION].append(MaterialCandidate(name=name, category=MaterialCategory.FUNCTIONALIZATION, composition=rf, properties=p, provenance=src))
                seen.add(rf)
        for cat in MaterialCategory:
            print(f"INFO: Resolved {len(system[cat])} candidates for {cat.name}")

        self._run_result = (system, bases)
        return self._run_result
