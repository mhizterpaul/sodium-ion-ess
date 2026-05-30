import json
import os
import re
import math
from typing import List, Dict, Optional, Any
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

import pybamm
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

# --- CONSTRAINED CHEMICAL SPACE ---
ALLOWED_SALTS = {"NaBOB": "C4BNaO8", "NaTCP": "C5H3Cl3NNaO"}
ALLOWED_FUNCTIONALIZATION = {"MTMS": "C4H12O3Si"}
BASE_CATHODE_FORMULA = "Na4Fe3P4O15"
DOPANTS = ["Mn", "Cr", "Ni"]

# --- SCIENTIFIC CONSTANTS ---
CACHE_FILE = "material_cache.json"
KT = 0.0259 # eV at 300K

# Class Baselines (Strictly separated physical fallbacks)
CLASS_BASELINES = {
    "Cathode": {"stability": 0.15, "formation_energy": -2.25, "band_gap": 0.1, "volume_per_atom": 12.6},
    "Salt": {"stability": 0.0, "formation_energy": -3.0, "band_gap": 7.9, "volume_per_atom": 12.7},
    "Anode": {"stability": 0.05, "formation_energy": -0.12, "band_gap": 0.4, "volume_per_atom": 12.0}
}

@dataclass
class MaterialCandidate:
    name: str
    category: str
    composition: str
    properties: Dict[str, float]
    projected_delta: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0

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
    """Decoupled transformation layer for material properties to performance deltas."""

    @staticmethod
    def cathode_perturbation(proxy_props: Dict[str, float], base_props: Dict[str, float], base_params: Any) -> Dict[str, float]:
        # Bounded realization factor to preserve diversity
        realization = max(0.3, min(1.0, math.exp(-15.0 * proxy_props["stability"])))

        # OCP Sampling
        base_ocp = base_params["Positive electrode OCP [V]"]
        base_v = float(getattr(base_ocp(0.5), 'value', base_ocp(0.5))) if callable(base_ocp) else float(base_ocp)

        # Perturbation Operators
        # Voltage shift (10% doping)
        de_diff = proxy_props["formation_energy"] - base_props["formation_energy"]
        v_boost = -de_diff * 0.1 * (base_v / 3.2) * realization

        # Diffusion scaling (10% lattice perturbation proxy)
        vol_ratio = proxy_props["volume_per_atom"] / base_props["volume_per_atom"]
        d_mult = (1.0 + 0.4 * (vol_ratio - 1.0)) * realization

        return {
            "voltage_boost": v_boost,
            "diffusivity_mult": max(0.2, d_mult)
        }

    @staticmethod
    def salt_dissociation(props: Dict[str, float], base_props: Dict[str, float]) -> Dict[str, float]:
        # Conductivity proxy: sigma ~ exp(-Eg/2kT)
        gap_diff = base_props["band_gap"] - props["band_gap"]
        sigma_index = min(max(math.exp(gap_diff / (2 * KT)), 0.1), 10.0)

        # Dissociation factor
        dissociation = 1.0 / (1.0 + math.exp(25.0 * (props["stability"] - 0.05)))

        return {
            "conductivity_mult": sigma_index * dissociation,
            "ion_transference_mult": 1.0 + (0.15 * dissociation)
        }

    @staticmethod
    def anode_interface(props: Dict[str, float]) -> Dict[str, float]:
        # SEI kinetics based on proxy stability
        sei_growth = 0.5 + 0.5 * math.exp(-props["stability"] * 8.0)
        r_sei = 1.0 + 0.4 * (1.0 - math.exp(-props["stability"]))
        loss = 0.7 + 0.3 * (1.0 - math.exp(-props["stability"]))

        return {
            "sei_growth_mult": sei_growth,
            "resistance_drift_mult": r_sei,
            "initial_loss_mult": loss
        }

class MaterialMappingEngine:
    """Constrained mapping engine for OQMD and Materials Project resolution."""

    def __init__(self):
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

    def _resolve_material(self, formula: str, category_baseline: str) -> tuple[Dict[str, float], float]:
        """Prioritized Resolution with strict sorting and baseline separation."""
        cache_key = f"RESOLVE:{formula}"
        if cache_key in self.cache:
            return self.cache[cache_key]["props"], self.cache[cache_key]["conf"]

        props, conf = None, 0.0

        # 1. OQMD Exact with Stability Sorting
        if self.session:
            try:
                oqmd_url = "https://oqmd.org/oqmdapi/formationenergy"
                params = {"composition": formula, "limit": 10, "fields": "delta_e,stability,band_gap,volume,natoms"}
                r = self.session.get(oqmd_url, params=params, timeout=15)
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
                    conf = 1.0
            except Exception: pass

        # 2. Materials Project Exact with Stability Sorting
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
                        conf = 0.9
            except Exception: pass

        # 3. Strict Baseline Fallback
        if not props:
            if category_baseline == "Salt": props = CLASS_BASELINES["Salt"]
            elif category_baseline == "Cathode": props = CLASS_BASELINES["Cathode"]
            else: props = CLASS_BASELINES["Anode"]
            conf = 0.5

        self.cache[cache_key] = {"props": props, "conf": conf}
        self._save_cache()
        return props, conf

    def run(self):
        print("Executing Decoupled Materials Mapping & Physics Engine...")
        system = {"Cathode_Dopant": [], "Salt": [], "Functionalization": []}
        physics = PhysicsModels()

        # Resolving Immutable Baselines
        base_cathode, _ = self._resolve_material(BASE_CATHODE_FORMULA, "Cathode")
        base_salt, _ = self._resolve_material("NaPF6", "Salt")

        # 1. Cathode Channel (Perturbation Operators)
        # We query dopant-related phosphates (e.g. NaMnPO4) as proxy for lattice perturbation drivers
        dopant_proxies = {"Mn": "NaMnPO4", "Cr": "NaCrPO4", "Ni": "NaNiPO4"}
        for d, proxy_formula in dopant_proxies.items():
            proxy_props, conf = self._resolve_material(proxy_formula, "Cathode")
            deltas = physics.cathode_perturbation(proxy_props, base_cathode, self.base_params)
            system["Cathode_Dopant"].append(MaterialCandidate(
                d, "Cathode_Dopant", f"Doped-{d}-NFPP", proxy_props, deltas, conf
            ))

        # 2. Salt Channel
        for name, formula in ALLOWED_SALTS.items():
            props, conf = self._resolve_material(formula, "Salt")
            deltas = physics.salt_dissociation(props, base_salt)
            system["Salt"].append(MaterialCandidate(name, "Salt", formula, props, deltas, conf))

        # 3. Anode Channel
        for name, formula in ALLOWED_FUNCTIONALIZATION.items():
            props, conf = self._resolve_material(formula, "Anode")
            deltas = physics.anode_interface(props)
            system["Functionalization"].append(MaterialCandidate(name, "Functionalization", formula, props, deltas, conf))

        return system

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    res = engine.run()
    for cat, cands in res.items():
        print(f"\nCategory: {cat}")
        for c in cands:
            print(f"  - {c.name} (Conf: {c.confidence:.1f}): {c.projected_delta}")
