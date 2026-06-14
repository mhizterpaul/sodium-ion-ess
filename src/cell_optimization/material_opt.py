import json
import os
import re
import math
import numpy as np
import logging
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

# --- DATABASES ---
SALT_DATABASE = {
    "NaPF6": {"conductivity": 0.8, "transference_number": 0.35, "viscosity": 4.5},
    "NaBOB": {"conductivity": 0.6, "transference_number": 0.45, "viscosity": 7.2},
    "NaTCP": {"conductivity": 0.5, "transference_number": 0.50, "viscosity": 12.0}
}

MTMS_DATABASE = {
    "MTMS": {
        "sei_growth_factor": 0.75,
        "resistance_growth_factor": 0.85,
        "exchange_current_factor": 1.2,
        "initial_sodium_loss_factor": 0.9
    }
}

ALLOWED_SALTS = list(SALT_DATABASE.keys())
ALLOWED_FUNCTIONALIZATION = list(MTMS_DATABASE.keys())

BASE_CATHODE_PRIORITIES = ["Na4Fe3(PO4)2P2O7", "Na2FeP2O7", "NaFeP2O7"]
BASE_SALT_FORMULA = "NaPF6"
BASE_INTERFACE_FORMULA = "C2H4O"
DOPANTS = ["Mn", "Cr", "Ni"]

OQMD_URL = "https://oqmd.org/oqmdapi/formationenergy"
CACHE_FILE = "material_cache.json"
CACHE_VERSION = "v23"

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
        self.cache = self._load_cache()
        self.session = self._setup_session() if requests else None
        # Provide base_params for Layer 3
        self.base_params = get_parameter_values()
        self.mp_key = os.environ.get("MP_API_KEY")

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
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            logging.warning(f"Failed to save cache: {e}")

    def _resolve_material(self, formula: str, source_override: Optional[str] = None, chemsys: Optional[str] = None) -> Tuple[Optional[Dict[str, float]], str]:
        cache_key = f"RESOLVE:{formula if not chemsys else chemsys}:{CACHE_VERSION}"
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
        print(f"Executing Strict Material Resolution (Layer 1)...")
        system = {cat: [] for cat in MaterialCategory}
        bases = {}

        for f in BASE_CATHODE_PRIORITIES:
            p, src = self._resolve_material(f, source_override="MP")
            if p:
                bases["cathode"] = {"formula": f, "properties": p, "source": src}
                break

        p_salt, src_salt = self._resolve_material(BASE_SALT_FORMULA)
        if p_salt:
            bases["salt"] = {"formula": BASE_SALT_FORMULA, "properties": p_salt, "source": src_salt, "solution": SALT_DATABASE[BASE_SALT_FORMULA]}

        p_int, src_int = self._resolve_material(BASE_INTERFACE_FORMULA)
        if p_int: bases["interface"] = {"formula": BASE_INTERFACE_FORMULA, "properties": p_int, "source": src_int}

        if not all(k in bases for k in ["cathode", "salt", "interface"]):
            return system, {}

        for d in DOPANTS:
            chemsys = f"Na-Fe-{d}-P-O"
            p, src = self._resolve_material(formula=f"Na4Fe2.7{d}0.3P4O15", chemsys=chemsys)
            if not p:
                chemsys = f"Na-{d}-P-O"
                p, src = self._resolve_material(formula=f"Na{d}PO4", chemsys=chemsys)

            if p:
                system[MaterialCategory.CATHODE_DOPANT].append(MaterialCandidate(name=f"{d}-doped", category=MaterialCategory.CATHODE_DOPANT, composition=f"Doped-{d}", properties=p, provenance=src))

        for name in ALLOWED_SALTS:
            system[MaterialCategory.SALT].append(MaterialCandidate(name=name, category=MaterialCategory.SALT, composition=name, properties=SALT_DATABASE[name], provenance="LITERATURE"))

        for name in ALLOWED_FUNCTIONALIZATION:
            system[MaterialCategory.FUNCTIONALIZATION].append(MaterialCandidate(name=name, category=MaterialCategory.FUNCTIONALIZATION, composition=name, properties=MTMS_DATABASE[name], provenance="LITERATURE"))

        return system, bases
