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

class MaterialCategory(Enum):
    CATHODE_DOPANT = auto()
    SALT = auto()
    FUNCTIONALIZATION = auto()

# --- FORMULAS FROM PAPER.MD ---
BASE_CATHODE_PRIORITIES = ["Na4Fe3(PO4)2P2O7", "Na2FeP2O7", "NaFeP2O7"]
BASE_SALT_FORMULA = "NaPF6"
BASE_INTERFACE_FORMULA = "C2H4O"
DOPANTS = ["Mn", "Cr", "Ni"]

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
        logging.basicConfig(level=logging.INFO)
        # MP API Key must be configured via environment variable
        self.mp_key = os.environ.get("MP_API_KEY")
        self.cache_version = self._generate_cache_version()
        self.cache = self._load_cache()
        self.session = self._setup_session() if requests else None
        self.base_params = get_parameter_values()

    def _generate_cache_version(self) -> str:
        # Dynamic cache version based on formulas
        all_formulas = "".join(BASE_CATHODE_PRIORITIES + [BASE_SALT_FORMULA, BASE_INTERFACE_FORMULA] + DOPANTS + list(SALTS.values()) + list(FUNCTIONALIZATION.values()))
        return hashlib.md5(all_formulas.encode()).hexdigest()[:8]

    def _setup_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    data = json.load(f)
                    if data.get("version") == self.cache_version:
                         return data.get("entries", {})
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump({"version": self.cache_version, "entries": self.cache}, f, indent=2)
        except IOError as e:
            logging.warning(f"Failed to save cache: {e}")

    def _resolve_material(self, formula: str, source_override: Optional[str] = None, chemsys: Optional[str] = None) -> Tuple[Optional[Dict[str, float]], str]:
        cache_key = f"RESOLVE:{formula if not chemsys else chemsys}"
        if cache_key in self.cache:
            return self.cache[cache_key]["props"], self.cache[cache_key].get("source", "UNKNOWN")

        props, source = None, "NONE"
        if MPRester and self.mp_key and (source_override == "MP" or source_override is None):
            try:
                with MPRester(api_key=self.mp_key) as mpr:
                    if chemsys:
                        docs = mpr.materials.summary.search(chemsys=chemsys, fields=['formula_pretty', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites'])
                    else:
                        docs = mpr.materials.summary.search(formula=formula, fields=['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites'])

                    if docs:
                        docs.sort(key=lambda d: d.energy_above_hull)
                        best = docs[0]
                        props = {
                            "stability": float(best.energy_above_hull),
                            "formation_energy": float(best.formation_energy_per_atom),
                            "band_gap": float(best.band_gap if best.band_gap is not None else 0.0),
                            "volume_per_atom": float(best.volume / best.nsites if best.nsites else 1.0)
                        }
                        source = "MATERIALS_PROJECT"
            except Exception: pass

        if props is None and self.session and (source_override == "OQMD" or source_override is None):
            try:
                params = {"composition": formula, "limit": 1, "fields": "delta_e,stability,band_gap,volume,natoms"}
                r = self.session.get(OQMD_URL, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    if data:
                        best = data[0]
                        props = {
                            "stability": float(best.get("stability", 0.1)),
                            "formation_energy": float(best.get("delta_e", 0.0)),
                            "band_gap": float(best.get("band_gap", 0.0)),
                            "volume_per_atom": float(best.get("volume", 1.0)) / float(best.get("natoms", 1.0))
                        }
                        source = "OQMD"
            except Exception: pass

        if props:
            self.cache[cache_key] = {"props": props, "source": source}
            self._save_cache()
            return props, source
        return None, "NONE"

    def run(self) -> Tuple[Dict[MaterialCategory, List[MaterialCandidate]], Dict[str, Any]]:
        print(f"Executing API-Based Material Resolution (Layer 1)...")
        system = {cat: [] for cat in MaterialCategory}
        bases = {}

        # 1. Resolve Cathode Base
        for f in BASE_CATHODE_PRIORITIES:
            p, src = self._resolve_material(f, source_override="MP")
            if p:
                bases["cathode"] = {"formula": f, "properties": p, "source": src}
                break

        # 2. Resolve Salt Base (NaPF6)
        p_salt_base, src_salt_base = self._resolve_material(BASE_SALT_FORMULA)
        if p_salt_base:
            bases["salt"] = {"formula": BASE_SALT_FORMULA, "properties": p_salt_base, "source": src_salt_base}

        # 3. Resolve Interface Base
        p_int, src_int = self._resolve_material(BASE_INTERFACE_FORMULA)
        if p_int:
            bases["interface"] = {"formula": BASE_INTERFACE_FORMULA, "properties": p_int, "source": src_int}

        if not all(k in bases for k in ["cathode", "salt", "interface"]):
            return system, {}

        # 4. Resolve Cathode Dopants
        for d in DOPANTS:
            chemsys = f"Na-Fe-{d}-P-O"
            p, src = self._resolve_material(formula=f"Na4Fe2.7{d}0.3P4O15", chemsys=chemsys)
            if not p:
                chemsys = f"Na-{d}-P-O"
                p, src = self._resolve_material(formula=f"Na{d}PO4", chemsys=chemsys)

            if p:
                system[MaterialCategory.CATHODE_DOPANT].append(MaterialCandidate(name=f"{d}-doped", category=MaterialCategory.CATHODE_DOPANT, composition=f"Doped-{d}", properties=p, provenance=src))

        # 5. Resolve Salts (NaBOB, NaTCP) from API
        for name, formula in SALTS.items():
             p, src = self._resolve_material(formula)
             if p:
                 system[MaterialCategory.SALT].append(MaterialCandidate(name=name, category=MaterialCategory.SALT, composition=formula, properties=p, provenance=src))

        # 6. Resolve Functionalization (MTMS) from API
        for name, formula in FUNCTIONALIZATION.items():
             p, src = self._resolve_material(formula)
             if p:
                 system[MaterialCategory.FUNCTIONALIZATION].append(MaterialCandidate(name=name, category=MaterialCategory.FUNCTIONALIZATION, composition=formula, properties=p, provenance=src))

        return system, bases
