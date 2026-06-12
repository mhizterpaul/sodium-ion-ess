import json
import os
import re
import math
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

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

# --- CONSTRAINED CHEMICAL SPACE ---
# Scientific Fix: Dedicated Salt Database for solution-phase properties
SALT_DATABASE = {
    "NaPF6": {"conductivity": 0.8, "transference_number": 0.35, "viscosity": 4.5, "oxidation_window": 4.5},
    "NaBOB": {"conductivity": 0.6, "transference_number": 0.45, "viscosity": 7.2, "oxidation_window": 4.2},
    "NaTCP": {"conductivity": 0.5, "transference_number": 0.50, "viscosity": 12.0, "oxidation_window": 4.8}
}

ALLOWED_SALTS = list(SALT_DATABASE.keys())
ALLOWED_FUNCTIONALIZATION = {"MTMS": ["C4H12O3Si", "CH3Si(OCH3)3"]}
MTMS_PROXY = "SiO2"

# Cascading Resolve Priorities
BASE_CATHODE_PRIORITIES = ["Na4Fe3(PO4)2P2O7", "Na2FeP2O7", "NaFeP2O7"]
BASE_SALT_FORMULA = "NaPF6"
BASE_INTERFACE_FORMULA = "C2H4O"
DOPANTS = ["Mn", "Cr", "Ni"]

# --- API CONFIG ---
OQMD_URL = "https://oqmd.org/oqmdapi/formationenergy"
CACHE_FILE = "material_cache.json"
CACHE_VERSION = "v16"

@dataclass
class MaterialCandidate:
    name: str
    category: str
    composition: str
    properties: Dict[str, float]
    database_uncertainty: float = 0.0
    provenance: str = "OQMD"

class MaterialMappingEngine:
    """Strict Data Layer: Handles ingestion, resolution, and caching ONLY."""
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.cache = self._load_cache()
        self.session = self._setup_session() if requests else None
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

    def _valid_props(self, p: Dict[str, Any]) -> bool:
        required = ["stability", "formation_energy", "band_gap", "volume_per_atom"]
        for k in required:
            if k not in p: return False
            try:
                v = float(p[k])
                if not np.isfinite(v): return False
            except (TypeError, ValueError):
                return False
        return True

    def _resolve_material(self, formula: str, source_override: Optional[str] = None) -> Tuple[Optional[Dict[str, float]], str]:
        cache_key = f"RESOLVE:{formula}:{CACHE_VERSION}"
        if cache_key in self.cache:
            return self.cache[cache_key]["props"], self.cache[cache_key].get("source", "UNKNOWN")

        props_mp = None
        if MPRester and self.mp_key and (source_override == "MP" or source_override is None):
            try:
                with MPRester(api_key=self.mp_key) as mpr:
                    docs = mpr.materials.summary.search(
                        formula=formula,
                        fields=['material_id', 'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites']
                    )
                    if docs:
                        docs.sort(key=lambda d: d.energy_above_hull)
                        best = docs[0]
                        props_mp = {
                            "stability": float(best.energy_above_hull),
                            "formation_energy": float(best.formation_energy_per_atom),
                            "band_gap": float(best.band_gap if best.band_gap is not None else 0.0),
                            "volume_per_atom": float(best.volume / best.nsites if best.nsites else 15.0),
                            "natoms": float(best.nsites),
                            "u": 0.05
                        }
            except Exception as e:
                logging.warning(f"MP query failed for {formula}: {e}")

        props_oqmd = None
        if self.session and (source_override == "OQMD" or source_override is None):
            try:
                params = {"composition": formula, "limit": 10, "fields": "delta_e,stability,band_gap,volume,natoms"}
                r = self.session.get(OQMD_URL, params=params, timeout=15)
                r.raise_for_status()
                data = r.json().get("data", [])
                if data:
                    data.sort(key=lambda x: float(x.get("stability", 1e9)))
                    best = data[0]
                    props_oqmd = {
                        "stability": float(best.get("stability", 0.1)),
                        "formation_energy": float(best.get("delta_e", 0.0)),
                        "band_gap": float(best.get("band_gap", 0.0)),
                        "volume_per_atom": float(best.get("volume", 1.0)) / float(best.get("natoms", 1.0)),
                        "natoms": float(best.get("natoms", 1.0)),
                        "u": 0.1
                    }
            except Exception as e:
                logging.warning(f"OQMD query failed for {formula}: {e}")

        props, source = None, "NONE"
        if props_mp and props_oqmd:
            w_m = 1.0 / (props_mp["u"]**2 + 1e-9)
            w_o = 1.0 / (props_oqmd["u"]**2 + 1e-9)
            common_keys = {"formation_energy", "band_gap", "stability", "volume_per_atom"}
            props = {k: (w_m * props_mp[k] + w_o * props_oqmd[k]) / (w_m + w_o) for k in common_keys}
            props["natoms"] = props_mp["natoms"]
            props["u"] = 0.5 * (props_mp["u"] + props_oqmd["u"])
            source = "MP+OQMD"
        elif props_mp:
            props, source = props_mp, "MATERIALS_PROJECT"
            props["u"] = 0.05
        elif props_oqmd:
            props, source = props_oqmd, "OQMD"
            props["u"] = 0.1

        if props and self._valid_props(props):
            self.cache[cache_key] = {"props": props, "source": source}
            self._save_cache()
            return props, source

        return None, "NONE"

    def run(self) -> Tuple[Dict[str, List[MaterialCandidate]], Dict[str, Any]]:
        print(f"Executing Strict Material Resolution (Scientific Descriptors Fix)...")
        system = {"Cathode_Dopant": [], "Salt": [], "Functionalization": []}
        bases = {}

        # --- Base Resolution ---
        for f in BASE_CATHODE_PRIORITIES:
            p, src = self._resolve_material(f, source_override="MP")
            if p:
                bases["cathode"] = {"formula": f, "properties": p, "source": src}
                break

        p_salt, src_salt = self._resolve_material(BASE_SALT_FORMULA)
        if p_salt:
            bases["salt"] = {"formula": BASE_SALT_FORMULA, "properties": p_salt, "source": src_salt}
            # Solution properties
            bases["salt"]["solution"] = SALT_DATABASE[BASE_SALT_FORMULA]

        p_int, src_int = self._resolve_material(BASE_INTERFACE_FORMULA)
        if p_int: bases["interface"] = {"formula": BASE_INTERFACE_FORMULA, "properties": p_int, "source": src_int}

        if not all(k in bases for k in ["cathode", "salt", "interface"]):
            logging.error("Failed to resolve base material properties. Aborting.")
            return system, bases

        # --- Dopants (Scientific Fix: Relevant Doped Formulas) ---
        # Instead of NaMnPO4, use doped NFPP proxies
        for d in DOPANTS:
            # Try exact doped formula first
            f = f"Na4Fe2.7{d}0.3P4O15" # Proxy for doped phase
            p, src = self._resolve_material(f, source_override="MP")
            if not p:
                # Fallback to searching the chemsys Na-Fe-{d}-P-O for the best match
                f = f"Na{d}PO4" # Legacy fallback
                p, src = self._resolve_material(f, source_override="MP")

            if p:
                system["Cathode_Dopant"].append(MaterialCandidate(
                    name=d, category="Cathode_Dopant", composition=f,
                    properties=p, database_uncertainty=p.get("u", 0.1), provenance=src))

        # --- Salts (Scientific Fix: Solution Descriptors) ---
        for name in ALLOWED_SALTS:
            # We don't query DFT for salts anymore, use SALT_DATABASE
            system["Salt"].append(MaterialCandidate(
                name=name, category="Salt", composition=name,
                properties=SALT_DATABASE[name], database_uncertainty=0.01, provenance="LITERATURE_SOLUTION"))

        # --- Functionalization (MTMS) ---
        for name, formulas in ALLOWED_FUNCTIONALIZATION.items():
            hit = False
            for formula in formulas:
                p, src = self._resolve_material(formula)
                if p:
                    system["Functionalization"].append(MaterialCandidate(
                        name=name, category="Functionalization", composition=formula,
                        properties=p, database_uncertainty=p.get("u", 0.1), provenance=src))
                    hit = True
                    break
            if not hit:
                p, src = self._resolve_material(MTMS_PROXY)
                if p:
                    system["Functionalization"].append(MaterialCandidate(
                        name=name, category="Functionalization_Proxy", composition=MTMS_PROXY,
                        properties=p, database_uncertainty=p.get("u", 0.1), provenance=src))

        return system, bases

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    db, bases = engine.run()
    print(f"\nBases Resolved: {list(bases.keys())}")
    for cat, items in db.items():
        print(f"Category {cat}: {len(items)} candidates")
