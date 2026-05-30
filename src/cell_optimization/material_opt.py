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

# Class Baselines (Physical fallbacks if API resolution fails)
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

class MaterialMappingEngine:
    """Constrained materials-to-parameter mapping engine using OQMD and Materials Project."""

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
        """Saves cache with deduplication logic."""
        clean_cache = {}
        # Keys are already strings like "RESOLVE:Formula"
        for key, val in sorted(self.cache.items()):
            clean_cache[key] = val
        with open(CACHE_FILE, "w") as f:
            json.dump(clean_cache, f, indent=2)

    def _resolve_material(self, formula: str, category_baseline: str) -> tuple[Dict[str, float], float]:
        """Resolution Flow: OQMD (Exact) -> MP (Exact) -> Class Baseline."""
        cache_key = f"RESOLVE:{formula}"
        if cache_key in self.cache:
            return self.cache[cache_key]["props"], self.cache[cache_key]["conf"]

        props, conf = None, 0.0

        # 1. OQMD Exact Lookup
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

        # 2. Materials Project Exact Lookup
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

        # 3. Class Baseline Fallback
        if not props:
            props = CLASS_BASELINES.get(category_baseline, CLASS_BASELINES["Anode"])
            conf = 0.5

        # Persist to cache
        self.cache[cache_key] = {"props": props, "conf": conf}
        self._save_cache()
        return props, conf

    def derive_cathode_channel(self, dopant: str, base_props: Dict[str, float]) -> MaterialCandidate:
        """Cathode Channel: Doping perturbations in NFPP framework."""
        # Primary search: Exact doped formula
        target_formula = f"Na4Fe2.9{dopant}0.1P4O15"
        props, conf = self._resolve_material(target_formula, "Cathode")

        # If exact fails, try singular variant as proxy for dopant influence
        if conf == 0.5:
            proxy_formula = f"Na2{dopant}P2O7"
            proxy_props, proxy_conf = self._resolve_material(proxy_formula, "Cathode")
            if proxy_conf > 0.5:
                props, conf = proxy_props, proxy_conf * 0.8 # Lower confidence for proxy

        # Physics parameters
        f_dopant = 0.1
        alpha = 0.4 # Volumetric diffusion sensitivity
        beta = 15.0 # Stability realization decay

        # Voltage Shift: ΔV = -(ΔEf_target - ΔEf_base) * f_dopant
        de_diff = props["formation_energy"] - base_props["formation_energy"]
        v_boost = -de_diff * f_dopant

        # Diffusion Modifier: 1 + alpha * (V_target/V_base - 1)
        vol_ratio = props["volume_per_atom"] / base_props["volume_per_atom"]
        d_mult = 1.0 + alpha * (vol_ratio - 1.0)

        # Stability Realization: exp(-beta * Ehull)
        realization = math.exp(-beta * props["stability"])

        deltas = {
            "voltage_boost": v_boost * realization,
            "diffusivity_mult": max(0.2, d_mult * realization)
        }

        return MaterialCandidate(dopant, "Cathode_Dopant", target_formula, props, deltas, conf)

    def derive_salt_channel(self, name: str, formula: str, base_props: Dict[str, float]) -> MaterialCandidate:
        """Salt Channel: Electrolyte dissociation and transport proxy."""
        props, conf = self._resolve_material(formula, "Salt")

        gamma = 25.0 # Dissociation penalty factor

        # Conductivity Index: exp(-Eg / 2kT)
        gap_diff = base_props["band_gap"] - props["band_gap"]
        sigma_index = math.exp(gap_diff / (2 * KT))
        sigma_index = min(max(sigma_index, 0.1), 10.0) # Clamping

        # Dissociation Effect: 1 / (1 + exp(gamma * stability))
        dissociation = 1.0 / (1.0 + math.exp(gamma * (props["stability"] - 0.05)))

        deltas = {
            "conductivity_mult": sigma_index * dissociation,
            "ion_transference_mult": 1.0 + (0.15 * dissociation)
        }

        return MaterialCandidate(name, "Salt", formula, props, deltas, conf)

    def derive_anode_channel(self, name: str, formula: str) -> MaterialCandidate:
        """Anode Channel: Interphase kinetics for MTMS functionalization."""
        props, conf = self._resolve_material(formula, "Anode")

        # MTMS kinetics parameters
        sei_suppression = 0.5 + 0.5 * math.exp(-props["stability"] * 8.0)
        resistance_drift = 1.0 + 0.4 * (1.0 - math.exp(-props["stability"]))
        initial_loss = 0.7 + 0.3 * (1.0 - math.exp(-props["stability"]))

        deltas = {
            "sei_growth_mult": sei_suppression,
            "resistance_drift_mult": resistance_drift,
            "initial_loss_mult": initial_loss
        }

        return MaterialCandidate(name, "Functionalization", formula, props, deltas, conf)

    def run(self):
        print("Executing Constrained Materials-to-Parameter Mapping Engine...")
        system = {"Cathode_Dopant": [], "Salt": [], "Functionalization": []}

        # Resolve Baselines
        base_cathode, _ = self._resolve_material(BASE_CATHODE_FORMULA, "Cathode")
        base_salt, _ = self._resolve_material("NaPF6", "Salt")

        # 1. Cathode Channel (Mn, Cr, Ni)
        for d in DOPANTS:
            system["Cathode_Dopant"].append(self.derive_cathode_channel(d, base_cathode))

        # 2. Salt Channel (NaBOB, NaTCP)
        for name, formula in ALLOWED_SALTS.items():
            system["Salt"].append(self.derive_salt_channel(name, formula, base_salt))

        # 3. Anode Channel (MTMS)
        for name, formula in ALLOWED_FUNCTIONALIZATION.items():
            system["Functionalization"].append(self.derive_anode_channel(name, formula))

        return system

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    results = engine.run()
    for category, candidates in results.items():
        print(f"\nCategory: {category}")
        for c in candidates:
            print(f"  - {c.name} [Confidence: {c.confidence:.1f}]: {c.projected_delta}")

    # Verify cache
    if os.path.exists(CACHE_FILE):
        print("\n--- material_cache.json (Cleaned & Deduplicated) ---")
        with open(CACHE_FILE, "r") as f:
            print(f.read())
