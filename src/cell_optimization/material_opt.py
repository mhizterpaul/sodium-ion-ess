import json
import os
import re
import math
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
CACHE_VERSION = "v2"
KT = 0.0259 # eV at 300K



@dataclass
class MaterialCandidate:
    name: str
    category: str
    composition: str
    properties: Dict[str, float]
    projected_delta: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    realization: float = 1.0
    uncertainty: float = 0.0
    provenance: str = "OQMD"

    def to_pybamm_delta(self) -> Dict[str, Any]:
        """Maps derived deltas to PyBaMM parameter names."""
        mapping = {}
        if self.category == "Cathode_Dopant":
            mapping["Positive electrode OCP [V]"] = ("additive", self.projected_delta.get("voltage_boost", 0.0))
            mapping["Positive particle diffusivity [m2.s-1]"] = ("multiplier", self.projected_delta.get("diffusivity_mult", 1.0))
        elif self.category == "Salt":
            mapping["Electrolyte conductivity [S.m-1]"] = ("multiplier", self.projected_delta.get("conductivity_mult", 1.0))
            mapping["Cation transference number"] = ("multiplier", self.projected_delta.get("ion_transference_mult", 1.0))
        elif self.category == "Functionalization":
            mapping["SEI reaction exchange current density [A.m-2]"] = ("multiplier", self.projected_delta.get("sei_growth_mult", 1.0))
            mapping["Initial concentration in negative electrode [mol.m-3]"] = ("multiplier", self.projected_delta.get("initial_loss_mult", 1.0))
            mapping["SEI resistivity [Ohm.m]"] = ("multiplier", self.projected_delta.get("resistance_drift_mult", 1.0))
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
    def cathode_perturbation(electronic_anchor: Dict[str, float], structural_anchor: Dict[str, float], base_params: Any, base_formula: str, proxy_formula: str) -> tuple[Dict[str, float], float]:
        realization = compute_chemical_realization(base_formula, proxy_formula, structural_anchor, electronic_anchor)

        base_ocp = base_params["Positive electrode OCP [V]"]
        base_v = PhysicsModels.safe_ocp(base_ocp)

        deltas = derive_coupled_deltas(structural_anchor, electronic_anchor, base_v, realization)
        return deltas, realization

    @staticmethod
    def salt_dissociation(props: Dict[str, float], base_props: Dict[str, float]) -> Dict[str, float]:
        # Refined salt model using dissociation energy proxies
        s_thermo, _ = PhysicsModels.stability_decomposition(props)
        s_base_thermo, _ = PhysicsModels.stability_decomposition(base_props)

        # Conductivity proxy: sigma ~ exp(-dEf/kT)
        # delta_Ef as first-order proxy for dissociation enhancement/inhibition
        ef_diff = props["formation_energy"] - base_props["formation_energy"]
        sigma_mult = math.exp(-ef_diff / (2 * KT))
        sigma_mult = min(max(sigma_mult, 0.1), 10.0)

        # Dissociation effect: sigmoid relative to NaPF6 stability
        delta_s = s_thermo - s_base_thermo
        dissociation = 1.0 / (1.0 + math.exp(25.0 * delta_s))

        return {
            "conductivity_mult": sigma_mult * dissociation,
            "ion_transference_mult": 1.0 + (0.15 * dissociation)
        }

    @staticmethod
    def anode_interface(props: Dict[str, float]) -> Dict[str, float]:
        s_thermo, _ = PhysicsModels.stability_decomposition(props)
        sei_growth = 0.5 + 0.5 * math.exp(-s_thermo * 8.0)
        r_sei = 1.0 + 0.4 * (1.0 - math.exp(-s_thermo))
        loss = 0.7 + 0.3 * (1.0 - math.exp(-s_thermo))

        return {"sei_growth_mult": sei_growth, "resistance_drift_mult": r_sei, "initial_loss_mult": loss}

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
        with open(CACHE_FILE, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _valid_props(self, p: Dict[str, Any]) -> bool:
        required = ["stability", "formation_energy", "band_gap", "volume_per_atom"]
        return all(k in p and isinstance(p[k], (int, float)) for k in required)

    def _resolve_material(self, formula: str, category_baseline: str) -> tuple[Optional[Dict[str, float]], float, str]:
        cache_key = f"RESOLVE:{formula}:{CACHE_VERSION}"
        if cache_key in self.cache:
            return self.cache[cache_key]["props"], self.cache[cache_key]["conf"], self.cache[cache_key].get("source", "UNKNOWN")

        props, conf, source = None, 0.0, "NONE"

        if self.session:
            try:
                params = {"composition": formula, "limit": 10, "fields": "delta_e,stability,band_gap,volume,natoms"}
                r = self.session.get(OQMD_URL, params=params, timeout=15)
                r.raise_for_status()
                data = r.json().get("data", [])
                if data:
                    data.sort(key=lambda x: float(x.get("stability", 1e9)))
                    best = data[0]
                    props = {
                        "stability": float(best.get("stability", 0.1)),
                        "formation_energy": float(best.get("delta_e", 0.0)),
                        "band_gap": float(best.get("band_gap", 0.0)),
                        "volume_per_atom": float(best.get("volume", 1.0)) / float(best.get("natoms", 1.0))
                    }
                    conf, source = 1.0, "OQMD"
            except Exception as e:
                logging.warning(f"OQMD query failed for {formula}: {e}")

        if not props and MPRester and self.mp_key:
            try:
                with MPRester(api_key=self.mp_key) as mpr:
                    docs = mpr.materials.summary.search(formula=formula, fields=['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites'])
                    if docs:
                        docs.sort(key=lambda d: d.energy_above_hull)
                        best = docs[0]
                        props = {
                            "stability": best.energy_above_hull,
                            "formation_energy": best.formation_energy_per_atom,
                            "band_gap": best.band_gap,
                            "volume_per_atom": best.volume / best.nsites if best.nsites else 15.0
                        }
                        conf, source = 0.9, "MATERIALS_PROJECT"
            except Exception as e:
                logging.warning(f"MP query failed for {formula}: {e}")

        if props and self._valid_props(props):
            self.cache[cache_key] = {"props": props, "conf": conf, "source": source}
            self._save_cache()

        return props, conf, source

    def run(self):
        print("Executing Decoupled Materials Mapping & Physics Engine...")
        system = {"Cathode_Dopant": [], "Salt": [], "Functionalization": []}
        physics = PhysicsModels()

        base_cathode, _, _ = self._resolve_material(BASE_CATHODE_FORMULA, "Cathode")
        base_salt, _, _ = self._resolve_material(BASE_SALT_FORMULA, "Salt")

        # Fallback to defaults if resolution fails for base materials
        if not base_cathode:
            base_cathode = {"stability": 0.05, "formation_energy": -2.1, "band_gap": 2.5, "volume_per_atom": 15.5}
        if not base_salt:
            base_salt = {"stability": 0.01, "formation_energy": -1.8, "band_gap": 5.0, "volume_per_atom": 25.0}

        dopant_proxies = {"Mn": "NaMnPO4", "Cr": "NaCrPO4", "Ni": "NaNiPO4"}
        for d, proxy_formula in dopant_proxies.items():
            electronic_anchor, conf, src = self._resolve_material(proxy_formula, "Cathode")
            if not electronic_anchor: continue
            deltas, realization = physics.cathode_perturbation(electronic_anchor, base_cathode, self.base_params, BASE_CATHODE_FORMULA, proxy_formula)
            system["Cathode_Dopant"].append(MaterialCandidate(
                name=d, category="Cathode_Dopant", composition=f"Doped-{d}-NFPP",
                properties=electronic_anchor, projected_delta=deltas, confidence=conf,
                realization=realization, uncertainty=(1.0 - realization)**2, provenance=src))

        for name, formula in ALLOWED_SALTS.items():
            props, conf, src = self._resolve_material(formula, "Salt")
            if not props: continue
            deltas = physics.salt_dissociation(props, base_salt)
            system["Salt"].append(MaterialCandidate(
                name=name, category="Salt", composition=formula, properties=props,
                projected_delta=deltas, confidence=conf, realization=1.0, uncertainty=0.0, provenance=src))

        for name, formula in ALLOWED_FUNCTIONALIZATION.items():
            props, conf, src = self._resolve_material(formula, "Anode")
            if not props: continue
            deltas = physics.anode_interface(props)
            system["Functionalization"].append(MaterialCandidate(
                name=name, category="Functionalization", composition=formula, properties=props,
                projected_delta=deltas, confidence=conf, realization=1.0, uncertainty=0.0, provenance=src))

        return system

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    res = engine.run()
    for cat, cands in res.items():
        print(f"\nCategory: {cat}")
        for c in cands:
            print(f"  - {c.name} (Conf: {c.confidence:.1f}, Source: {c.provenance}): {c.projected_delta}")
