import json
import os
import re
import math
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from src.cell_optimization.chem_regularization import (
    compute_chemical_realization,
    derive_coupled_deltas,
    apply_connectivity_scaling,
    KT
)

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

import pybamm
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

# --- CONSTRAINED CHEMICAL SPACE ---
ALLOWED_SALTS = {"NaBOB": "C4BNaO8", "NaTCP": "C5H3Cl3NNaO"}
ALLOWED_FUNCTIONALIZATION = {"MTMS": ["C4H12O3Si", "CH3Si(OCH3)3"]}
MTMS_PROXY = "SiO2"

# Cascading Resolve Priorities
BASE_CATHODE_PRIORITIES = ["Na4Fe3P4O15", "NaFePO4"]
BASE_SALT_FORMULA = "NaPF6"
BASE_INTERFACE_FORMULA = "C2H4O" # Meaningful baseline for SEI/MTMS modification
DOPANTS = ["Mn", "Cr", "Ni"]

# --- SCIENTIFIC & API CONFIG ---
OQMD_URL = "https://oqmd.org/oqmdapi/formationenergy"
CACHE_FILE = "material_cache.json"
CACHE_VERSION = "v14"

REQUIRED_CHANNELS = {"thermodynamic", "kinetic", "transport", "structural"}

@dataclass
class MaterialCandidate:
    name: str
    category: str
    composition: str
    properties: Dict[str, float]
    projected_delta: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    realization: float = 1.0
    database_uncertainty: float = 0.0
    proxy_uncertainty: float = 0.0
    provenance: str = "OQMD"

    def __post_init__(self):
        """Strict schema validation for projected_delta across all categories."""
        is_valid = (
            isinstance(self.projected_delta, dict)
            and REQUIRED_CHANNELS.issubset(self.projected_delta.keys())
            and all(isinstance(self.projected_delta[k], dict) for k in REQUIRED_CHANNELS)
        )
        if not is_valid:
            logging.error(f"MaterialCandidate {self.name} ({self.category}) has malformed projected_delta schema.")

    def to_pybamm_delta(self) -> Dict[str, Any]:
        """Maps derived deltas to PyBaMM parameter names."""
        mapping = {}
        td = self.projected_delta.get("thermodynamic", {})
        kt = self.projected_delta.get("kinetic", {})
        tr = self.projected_delta.get("transport", {})

        if self.category == "Cathode_Dopant":
            mapping["Positive electrode OCP [V]"] = ("additive", td.get("voltage_boost", 0.0))
            mapping["Positive particle diffusivity [m2.s-1]"] = ("multiplier", math.exp(tr.get("diffusivity_log_delta", 0.0)))
            mapping["Positive electrode exchange-current density [A.m-2]"] = ("multiplier", math.exp(kt.get("reaction_rate_log_delta", 0.0)))
        elif self.category == "Salt":
            mapping["Electrolyte conductivity [S.m-1]"] = ("multiplier", tr.get("conductivity_mult", 1.0))
            mapping["Cation transference number"] = ("multiplier", tr.get("ion_transference_mult", 1.0))
        elif self.category == "Functionalization":
            mapping["SEI reaction exchange current density [A.m-2]"] = ("multiplier", kt.get("sei_growth_mult", 1.0))
            mapping["Initial concentration in negative electrode [mol.m-3]"] = ("multiplier", td.get("initial_loss_mult", 1.0))
            mapping["SEI resistivity [Ohm.m]"] = ("multiplier", tr.get("resistance_drift_mult", 1.0))
            mapping["Negative electrode exchange-current density [A.m-2]"] = ("multiplier", math.exp(kt.get("negative_exchange_log_delta", 0.0)))
        return mapping

class MaterialMappingEngine:
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
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        pass

    def _valid_props(self, p: Dict[str, Any]) -> bool:
        required = ["stability", "formation_energy", "band_gap", "volume_per_atom"]
        for k in required:
            if k not in p: return False
            try:
                v = float(p[k])
            except: return False
            if not np.isfinite(v): return False
        return True

    def _resolve_material(self, formula: str, source_override: Optional[str] = None) -> tuple[Optional[Dict[str, float]], float, str]:
        cache_key = f"RESOLVE:{formula}:{CACHE_VERSION}"
        if cache_key in self.cache:
            return self.cache[cache_key]["props"], self.cache[cache_key]["conf"], self.cache[cache_key].get("source", "UNKNOWN")

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

        props, conf, source = None, 0.0, "NONE"
        if props_mp and props_oqmd:
            w_m = 1.0 / (props_mp["u"]**2 + 1e-9)
            w_o = 1.0 / (props_oqmd["u"]**2 + 1e-9)
            common_keys = {"formation_energy", "band_gap", "stability", "volume_per_atom"}
            props = {k: (w_m * props_mp[k] + w_o * props_oqmd[k]) / (w_m + w_o) for k in common_keys}
            props["natoms"] = props_mp["natoms"]
            props["u"] = 0.5 * (props_mp["u"] + props_oqmd["u"])
            conf, source = 0.95, "MP+OQMD"
        elif props_mp:
            props, conf, source = props_mp, 1.0, "MATERIALS_PROJECT"
        elif props_oqmd:
            props, conf, source = props_oqmd, 0.9, "OQMD"

        if props and self._valid_props(props):
            self.cache[cache_key] = {"props": props, "conf": conf, "source": source}

        return props, conf, source

    def run(self):
        print(f"Executing Physics-Constrained Materials Mapping (Uncertainty-Aware)...")
        system = {"Cathode_Dopant": [], "Salt": [], "Functionalization": []}

        # --- Base Cathode Resolution ---
        base_cathode = None
        base_cathode_formula = None
        for f in BASE_CATHODE_PRIORITIES:
            base_cathode, _, _ = self._resolve_material(f, source_override="MP")
            if base_cathode:
                base_cathode_formula = f
                break

        base_salt, _, _ = self._resolve_material(BASE_SALT_FORMULA)
        base_interface, _, _ = self._resolve_material(BASE_INTERFACE_FORMULA)

        if not all([base_cathode, base_salt, base_interface]):
            logging.error("Failed to resolve required base material properties. Aborting mapping.")
            return system

        # --- Dopants ---
        for d in DOPANTS:
            proxy_formula = f"Na{d}PO4"
            proxy_props, conf, src = self._resolve_material(proxy_formula, source_override="MP")
            if not proxy_props: continue

            realization = compute_chemical_realization(base_cathode_formula, proxy_formula, base_cathode, proxy_props)
            deltas = derive_coupled_deltas(base_cathode, proxy_props, realization=realization)
            system["Cathode_Dopant"].append(MaterialCandidate(
                name=d, category="Cathode_Dopant", composition=proxy_formula,
                properties=proxy_props, projected_delta=deltas, confidence=conf,
                realization=realization, database_uncertainty=proxy_props.get("u", 0.1),
                proxy_uncertainty=(1-realization)**2, provenance=src))

        # --- Salts ---
        for name, formula in ALLOWED_SALTS.items():
            props, conf, src = self._resolve_material(formula)
            if not props: continue

            realization = compute_chemical_realization(BASE_SALT_FORMULA, formula, base_salt, props)
            deltas = derive_coupled_deltas(base_salt, props, realization=realization)
            system["Salt"].append(MaterialCandidate(
                name=name, category="Salt", composition=formula, properties=props,
                projected_delta=deltas, confidence=conf, realization=realization,
                database_uncertainty=props.get("u", 0.05), proxy_uncertainty=(1-realization)**2, provenance=src))

        # --- Functionalization (MTMS) ---
        for name, formulas in ALLOWED_FUNCTIONALIZATION.items():
            resolved_props = None
            resolved_formula = None
            source_was_proxy = False
            for formula in formulas:
                resolved_props, conf, src = self._resolve_material(formula)
                if resolved_props:
                    resolved_formula = formula
                    break

            if not resolved_props:
                resolved_props, conf, src = self._resolve_material(MTMS_PROXY)
                if resolved_props:
                    resolved_props = apply_connectivity_scaling(resolved_props, phi=0.75)
                    resolved_formula = MTMS_PROXY
                    source_was_proxy = True

            if not resolved_props: continue

            realization = compute_chemical_realization(BASE_INTERFACE_FORMULA, resolved_formula, base_interface, resolved_props)
            deltas = derive_coupled_deltas(base_interface, resolved_props, realization=realization, is_network=source_was_proxy)
            system["Functionalization"].append(MaterialCandidate(
                name=name, category="Functionalization", composition=resolved_formula, properties=resolved_props,
                projected_delta=deltas, confidence=conf, realization=realization,
                database_uncertainty=0.01, proxy_uncertainty=(1-realization)**2, provenance=src))

        return system

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    res = engine.run()
    for cat, cands in res.items():
        print(f"\nCategory: {cat}")
        for c in cands:
            print(f"  - {c.name} (R: {c.realization:.2f}, Proxy-U: {c.proxy_uncertainty:.2e}): {c.composition}")
