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
BASE_CATHODE = "Na4Fe3P4O15"
DOPANTS = ["Mn", "Cr", "Ni"]

# --- SCIENTIFIC CONSTANTS ---
CACHE_FILE = "material_cache.json"
KT = 0.0259 # eV at 300K

# Class Baselines (Fallback if API fails)
BASELINES = {
    "Cathode": {"stability": 0.1, "formation_energy": -2.2, "band_gap": 0.5, "volume_per_atom": 12.0},
    "Salt": {"stability": 0.05, "formation_energy": -1.5, "band_gap": 4.0, "volume_per_atom": 10.0},
    "Anode": {"stability": 0.02, "formation_energy": -0.1, "band_gap": 1.0, "volume_per_atom": 15.0}
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
        """Maps derived deltas to PyBaMM parameter names based on physics channel."""
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

class MaterialMappingEngine:
    """Constrained materials-to-parameter mapping engine."""

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
        """Resolution Flow: OQMD (Exact) -> MP (Exact) -> Class Baseline."""
        cache_key = f"RESOLVE:{formula}"
        if cache_key in self.cache:
            return self.cache[cache_key]["props"], self.cache[cache_key]["conf"]

        props, conf = None, 0.0

        # 1. OQMD Exact
        if self.session:
            try:
                oqmd_url = "https://oqmd.org/oqmdapi/formationenergy"
                params = {"composition": formula, "limit": 1, "fields": "delta_e,stability,band_gap,volume,natoms"}
                r = self.session.get(oqmd_url, params=params, timeout=15)
                r.raise_for_status()
                data = r.json().get("data", [])
                if data:
                    best = data[0]
                    props = {
                        "stability": float(best.get("stability", 0.1)),
                        "formation_energy": float(best.get("delta_e", 0.0)),
                        "band_gap": float(best.get("band_gap", 0.0)),
                        "volume_per_atom": float(best.get("volume", 1.0)) / float(best.get("natoms", 1.0))
                    }
                    conf = 1.0
            except Exception: pass

        # 2. MP Exact
        if not props and MPRester and self.mp_key:
            try:
                with MPRester(api_key=self.mp_key) as mpr:
                    docs = mpr.materials.summary.search(formula=formula, fields=['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'volume', 'nsites'])
                    if docs:
                        best = docs[0]
                        props = {
                            "stability": best.energy_above_hull,
                            "formation_energy": best.formation_energy_per_atom,
                            "band_gap": best.band_gap,
                            "volume_per_atom": best.volume / best.nsites if best.nsites else 15.0
                        }
                        conf = 0.9
            except Exception: pass

        # 3. Class Baseline
        if not props:
            props = BASELINES.get(category_baseline, BASELINES["Anode"])
            conf = 0.5

        # Update cache
        self.cache[cache_key] = {"props": props, "conf": conf}
        self._save_cache()
        return props, conf

    def derive_cathode_channel(self, dopant: str, base_props: Dict[str, float]) -> MaterialCandidate:
        """Dopant perturbation model for fixed NFPP framework."""
        formula = f"Na2{dopant}P2O7"
        props, conf = self._resolve_material(formula, "Cathode")

        f_dopant = 0.1
        alpha = 0.5
        beta = 10.0

        de_diff = props["formation_energy"] - base_props["formation_energy"]
        v_boost = -de_diff * f_dopant

        vol_ratio = props["volume_per_atom"] / base_props["volume_per_atom"]
        d_mult = 1.0 + alpha * (vol_ratio - 1.0)

        realization = math.exp(-beta * props["stability"])

        deltas = {
            "voltage_boost": v_boost * realization,
            "diffusivity_mult": max(0.1, d_mult * realization)
        }

        return MaterialCandidate(dopant, "Cathode_Dopant", formula, props, deltas, conf)

    def derive_salt_channel(self, name: str, formula: str, base_props: Dict[str, float]) -> MaterialCandidate:
        """Molecular dissociation proxy model."""
        props, conf = self._resolve_material(formula, "Salt")

        gamma = 20.0
        gap_diff = base_props["band_gap"] - props["band_gap"]
        sigma_index = math.exp(gap_diff / (2 * KT))
        sigma_index = min(max(sigma_index, 0.2), 5.0)

        dissociation = 1.0 / (1.0 + math.exp(gamma * (props["stability"] - 0.05)))

        deltas = {
            "conductivity_mult": sigma_index * dissociation,
            "ion_transference_mult": 1.0 + (0.1 * dissociation)
        }

        return MaterialCandidate(name, "Salt", formula, props, deltas, conf)

    def derive_anode_channel(self, name: str, formula: str) -> MaterialCandidate:
        """MTMS SEI kinetics model."""
        props, conf = self._resolve_material(formula, "Anode")

        kappa = 0.5
        sei_growth = 0.5 + 0.5 * math.exp(-props["stability"] * 5.0)
        r_sei = 1.0 + kappa * (1.0 - math.exp(-props["stability"]))
        loss = 0.7 + 0.3 * (1.0 - math.exp(-props["stability"]))

        deltas = {
            "sei_growth_mult": sei_growth,
            "resistance_drift_mult": r_sei,
            "initial_loss_mult": loss
        }

        return MaterialCandidate(name, "Functionalization", formula, props, deltas, conf)

    def run(self):
        print("Executing Constrained Materials-to-Parameter Mapping...")
        system = {"Cathode_Dopant": [], "Salt": [], "Functionalization": []}

        base_cathode, _ = self._resolve_material(BASE_CATHODE, "Cathode")
        base_salt, _ = self._resolve_material("NaPF6", "Salt")

        for d in DOPANTS:
            system["Cathode_Dopant"].append(self.derive_cathode_channel(d, base_cathode))

        for name, formula in ALLOWED_SALTS.items():
            system["Salt"].append(self.derive_salt_channel(name, formula, base_salt))

        for name, formula in ALLOWED_FUNCTIONALIZATION.items():
            system["Functionalization"].append(self.derive_anode_channel(name, formula))

        return system

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    res = engine.run()
    for cat, cands in res.items():
        print(f"\nCategory: {cat}")
        for c in cands:
            print(f"  - {c.name} (Conf: {c.confidence}): {c.projected_delta}")
