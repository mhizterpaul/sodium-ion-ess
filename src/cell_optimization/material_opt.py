import json
import os
import re
import math
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from src.cell_optimization.chem_regularization import compute_chemical_realization, derive_coupled_deltas

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
ALLOWED_FUNCTIONALIZATION = {"MTMS": "C4H12O3Si"}
BASE_CATHODE_FORMULA = "Na4Fe3P4O15"
BASE_SALT_FORMULA = "NaPF6"
DOPANTS = ["Mn", "Cr", "Ni"]

# --- SCIENTIFIC & API CONFIG ---
OQMD_URL = "https://oqmd.org/oqmdapi/formationenergy"
CACHE_FILE = "material_cache.json"
CACHE_VERSION = "v3"
KT = 0.0259 # eV at 300K

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
    uncertainty: float = 0.0
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
        if self.category == "Cathode_Dopant":
            td = self.projected_delta.get("thermodynamic", {})
            kt = self.projected_delta.get("kinetic", {})
            tr = self.projected_delta.get("transport", {})
            mapping["Positive electrode OCP [V]"] = ("additive", td.get("voltage_boost", 0.0))
            mapping["Positive particle diffusivity [m2.s-1]"] = ("multiplier", math.exp(tr.get("diffusivity_log_delta", 0.0)))
            mapping["Positive electrode exchange-current density [A.m-2]"] = ("multiplier", math.exp(kt.get("reaction_rate_log_delta", 0.0)))
        elif self.category == "Salt":
            tr = self.projected_delta.get("transport", {})
            mapping["Electrolyte conductivity [S.m-1]"] = ("multiplier", tr.get("conductivity_mult", 1.0))
            mapping["Cation transference number"] = ("multiplier", tr.get("ion_transference_mult", 1.0))
        elif self.category == "Functionalization":
            td = self.projected_delta.get("thermodynamic", {})
            kt = self.projected_delta.get("kinetic", {})
            tr = self.projected_delta.get("transport", {})
            mapping["SEI reaction exchange current density [A.m-2]"] = ("multiplier", kt.get("sei_growth_mult", 1.0))
            mapping["Initial concentration in negative electrode [mol.m-3]"] = ("multiplier", td.get("initial_loss_mult", 1.0))
            mapping["SEI resistivity [Ohm.m]"] = ("multiplier", tr.get("resistance_drift_mult", 1.0))
        return mapping

class PhysicsModels:
    """Transformation layer for material properties to performance deltas."""

    @staticmethod
    def stability_decomposition(props: Dict[str, float]) -> tuple[float, float]:
        s_thermo = props["stability"]
        s_kinetic = props["band_gap"] / (props["volume_per_atom"] + 1e-6)
        return s_thermo, s_kinetic

    @staticmethod
    def safe_ocp(ocp_func: Any, x: float = 0.5) -> float:
        try:
            v = ocp_func(x)
            return float(getattr(v, "value", v))
        except Exception:
            return 3.2 * 0.8

    @staticmethod
    def cathode_perturbation(proxy_props: Dict[str, float], base_props: Dict[str, float], base_params: Any, base_formula: str, proxy_formula: str) -> tuple[Dict[str, Any], float]:
        """Calculates cathode deltas with semantically correct anchor naming."""
        realization = compute_chemical_realization(base_formula, proxy_formula, base_props, proxy_props)
        deltas = derive_coupled_deltas(base_props, proxy_props)
        return deltas, realization

    @staticmethod
    def salt_dissociation(props: Dict[str, float], base_props: Dict[str, float]) -> Dict[str, Any]:
        """Refined salt model returning structured channels."""
        s_thermo, _ = PhysicsModels.stability_decomposition(props)
        s_base_thermo, _ = PhysicsModels.stability_decomposition(base_props)

        ef_diff = props["formation_energy"] - base_props["formation_energy"]
        sigma_mult = math.exp(np.clip(-ef_diff / (2 * KT), -10, 10))
        sigma_mult = min(max(sigma_mult, 0.1), 10.0)

        delta_s = s_thermo - s_base_thermo
        dissociation = 1.0 / (1.0 + math.exp(np.clip(25.0 * delta_s, -20, 20)))

        return {
            "thermodynamic": {"stability_shift": delta_s},
            "kinetic": {},
            "transport": {
                "conductivity_mult": sigma_mult * dissociation,
                "ion_transference_mult": 1.0 + (0.15 * dissociation)
            },
            "structural": {}
        }

    @staticmethod
    def anode_interface(props: Dict[str, float]) -> Dict[str, Any]:
        """Anode functionalization model returning structured channels."""
        s_thermo, _ = PhysicsModels.stability_decomposition(props)
        sei_growth = 0.5 + 0.5 * math.exp(np.clip(-s_thermo * 8.0, -20, 20))
        r_sei = 1.0 + 0.4 * (1.0 - math.exp(np.clip(-s_thermo, -20, 20)))
        loss = 0.7 + 0.3 * (1.0 - math.exp(np.clip(-s_thermo, -20, 20)))

        return {
            "thermodynamic": {"initial_loss_mult": loss},
            "kinetic": {"sei_growth_mult": sei_growth},
            "transport": {"resistance_drift_mult": r_sei},
            "structural": {}
        }

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

    def _resolve_material(self, formula: str, category_baseline: str) -> tuple[Optional[Dict[str, float]], float, str]:
        cache_key = f"RESOLVE:{formula}:{CACHE_VERSION}"
        if cache_key in self.cache:
            return self.cache[cache_key]["props"], self.cache[cache_key]["conf"], self.cache[cache_key].get("source", "UNKNOWN")

        props_oqmd = None
        if self.session:
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
                        "natoms": float(best.get("natoms", 1.0))
                    }
            except Exception as e:
                logging.warning(f"OQMD query failed for {formula}: {e}")

        props_mp = None
        if MPRester and self.mp_key:
            try:
                with MPRester(api_key=self.mp_key) as mpr:
                    docs = mpr.materials.summary.search(formula=formula, fields=['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites'])
                    if docs:
                        docs.sort(key=lambda d: d.energy_above_hull)
                        best = docs[0]
                        props_mp = {
                            "stability": best.energy_above_hull,
                            "formation_energy": best.formation_energy_per_atom,
                            "band_gap": best.band_gap,
                            "volume_per_atom": best.volume / best.nsites if best.nsites else 15.0,
                            "natoms": best.nsites
                        }
            except Exception as e:
                logging.warning(f"MP query failed for {formula}: {e}")

        props, conf, source = None, 0.0, "NONE"
        if props_oqmd and props_mp:
            props = {k: 0.5*(props_oqmd[k] + props_mp[k]) for k in props_oqmd}
            conf = 0.95
            props["uncertainty"] = abs(props_oqmd["formation_energy"] - props_mp["formation_energy"])
            source = "OQMD+MP"
        elif props_oqmd:
            props, conf, source = props_oqmd, 1.0, "OQMD"
            props["uncertainty"] = 0.05
        elif props_mp:
            props, conf, source = props_mp, 0.9, "MATERIALS_PROJECT"
            props["uncertainty"] = 0.1

        if props and self._valid_props(props):
            self.cache[cache_key] = {"props": props, "conf": conf, "source": source}

        return props, conf, source

    def run(self):
        print("Executing Physics-Driven Materials Mapping...")
        system = {"Cathode_Dopant": [], "Salt": [], "Functionalization": []}
        physics = PhysicsModels()

        base_cathode, _, _ = self._resolve_material(BASE_CATHODE_FORMULA, "Cathode")
        base_salt, _, _ = self._resolve_material(BASE_SALT_FORMULA, "Salt")

        if not base_cathode:
            base_cathode = {"stability": 0.05, "formation_energy": -2.1, "band_gap": 2.5, "volume_per_atom": 15.5, "natoms": 26}
        if not base_salt:
            base_salt = {"stability": 0.01, "formation_energy": -1.8, "band_gap": 5.0, "volume_per_atom": 25.0, "natoms": 7}

        dopant_proxies = {
            "Mn": "Na4Fe2.7Mn0.3P4O15",
            "Cr": "Na4Fe2.7Cr0.3P4O15",
            "Ni": "Na4Fe2.7Ni0.3P4O15"
        }
        for d, proxy_formula in dopant_proxies.items():
            proxy_props, conf, src = self._resolve_material(proxy_formula, "Cathode")
            if not proxy_props:
                fallback_formula = f"Na{d}PO4"
                proxy_props, conf, src = self._resolve_material(fallback_formula, "Cathode")

            if not proxy_props: continue

            deltas, realization = physics.cathode_perturbation(proxy_props, base_cathode, self.base_params, BASE_CATHODE_FORMULA, proxy_formula)

            u_proxy = proxy_props.get("uncertainty", 0.1)
            u_base = base_cathode.get("uncertainty", 0.05)
            w_p = 1.0 / (u_proxy**2 + 1e-6)
            w_b = 1.0 / (u_base**2 + 1e-6)

            common_keys = {"formation_energy", "band_gap", "stability", "volume_per_atom"}
            corrected_props = {
                k: (w_p * proxy_props[k] + w_b * base_cathode[k]) / (w_p + w_b)
                for k in common_keys
                if k in proxy_props and k in base_cathode and np.isfinite(proxy_props[k]) and np.isfinite(base_cathode[k])
            }

            system["Cathode_Dopant"].append(MaterialCandidate(
                name=d, category="Cathode_Dopant", composition=proxy_formula,
                properties=corrected_props, projected_delta=deltas, confidence=conf,
                realization=realization, uncertainty=u_proxy, provenance=src))

        for name, formula in ALLOWED_SALTS.items():
            props, conf, src = self._resolve_material(formula, "Salt")
            if not props: continue
            deltas = physics.salt_dissociation(props, base_salt)
            system["Salt"].append(MaterialCandidate(
                name=name, category="Salt", composition=formula, properties=props,
                projected_delta=deltas, confidence=conf, realization=1.0, uncertainty=props.get("uncertainty", 0.05), provenance=src))

        for name, formula in ALLOWED_FUNCTIONALIZATION.items():
            props, conf, src = self._resolve_material(formula, "Anode")
            if not props: continue
            deltas = physics.anode_interface(props)
            system["Functionalization"].append(MaterialCandidate(
                name=name, category="Functionalization", composition=formula, properties=props,
                projected_delta=deltas, confidence=conf, realization=1.0, uncertainty=0.01, provenance=src))

        return system

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    res = engine.run()
    for cat, cands in res.items():
        print(f"\nCategory: {cat}")
        for c in cands:
            print(f"  - {c.name} (Conf: {c.confidence:.1f}, Source: {c.provenance}): {c.projected_delta}")
