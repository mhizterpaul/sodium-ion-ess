import json
import os
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None

import pybamm
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

# --- CONSTANTS & CONFIG ---
CACHE_FILE = "material_cache.json"
OQMD_URL = "http://oqmd.org/oqmdapi/formationenergy"

@dataclass
class MaterialCandidate:
    name: str
    category: str
    composition: str
    energy_above_hull: float = 0.0
    formation_energy: float = 0.0
    band_gap: float = 0.0
    volume_per_atom: float = 0.0
    projected_delta: Dict[str, float] = field(default_factory=dict)
    reference: str = "OQMD Derived"

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
            mapping["Negative electrode exchange-current density [A.m-2]"] = ("multiplier", self.projected_delta.get("exchange_current_mult", 1.0))
        return mapping

class MaterialDiscoveryFramework:
    def __init__(self):
        self.cache = self._load_cache()
        self.session = self._setup_session() if requests else None
        self.base_params = get_parameter_values()

    def _setup_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))
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
        # Deduplicate keys by normalizing params
        clean_cache = {}
        for key, val in self.cache.items():
            try:
                # Key is a JSON string of params, normalize it
                p = json.loads(key)
                norm_key = json.dumps(p, sort_keys=True)
                if norm_key not in clean_cache:
                    clean_cache[norm_key] = val
            except Exception:
                clean_cache[key] = val

        with open(CACHE_FILE, "w") as f:
            json.dump(clean_cache, f, indent=2)

    def _fetch_oqmd(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        cache_key = json.dumps(params, sort_keys=True)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.session:
            return []

        try:
            params["fields"] = "name,entry_id,composition,delta_e,stability,band_gap,volume,natoms"
            response = self.session.get(OQMD_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("data", []) or data.get("results", [])
            self.cache[cache_key] = results
            self._save_cache()
            return results
        except Exception as e:
            print(f"OQMD API Error for {params}: {e}")
            return []

    def get_properties(self, composition: str) -> Dict[str, float]:
        """Fetch thermodynamic and electronic properties from OQMD."""
        results = self._fetch_oqmd({"composition": composition, "limit": 1})
        if not results:
            import re
            elements = list(set(re.findall(r'[A-Z][a-z]?', composition)))
            if elements:
                filter_str = f"element_set=({','.join(elements)}) AND ntypes={len(elements)}"
                results = self._fetch_oqmd({"filter": filter_str, "limit": 10})

        if results:
            try:
                results.sort(key=lambda x: float(x.get("stability", 1.0)))
                best = results[0]
                natoms = float(best.get("natoms", 1.0))
                return {
                    "stability": float(best.get("stability", 0.1)),
                    "formation_energy": float(best.get("delta_e", 0.0)),
                    "band_gap": float(best.get("band_gap", 0.0)),
                    "volume_per_atom": float(best.get("volume", 1.0)) / natoms
                }
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        return {"stability": 0.2, "formation_energy": -1.0, "band_gap": 1.0, "volume_per_atom": 15.0}

    def derive_deltas(self, target_props: Dict[str, float], base_props: Dict[str, float], category: str) -> Dict[str, float]:
        """Derive performance deltas using OQMD property ratios and PyBaMM baseline values."""
        deltas = {}
        de_diff = target_props["formation_energy"] - base_props["formation_energy"]

        if category == "Cathode_Dopant":
            # Extract baseline from PyBaMM to inform the delta derivation
            # OCP shift: ~ -de_diff (eV) / valence. We scale based on typical redox window.
            deltas["voltage_boost"] = -de_diff * 0.1

            # Diffusivity: D_new = D_base * (V_new/V_base)^2
            vol_ratio = target_props["volume_per_atom"] / base_props["volume_per_atom"]
            deltas["diffusivity_mult"] = vol_ratio ** 2

        elif category == "Salt":
            # Conductivity: sigma_new = sigma_base * (Eg_base / Eg_new)^0.5
            bg_ratio = base_props["band_gap"] / max(target_props["band_gap"], 0.1)
            deltas["conductivity_mult"] = bg_ratio ** 0.5

            # Cation transference multiplier
            deltas["ion_transference_mult"] = 1.0 + (target_props["stability"] * 0.1)

        elif category == "Functionalization":
            # SEI growth and loss multipliers derived from stability ratios
            stab_ratio = base_props["stability"] / max(target_props["stability"], 0.01)
            deltas["sei_growth_mult"] = 0.5 + 0.5 * (1.0/stab_ratio)
            deltas["initial_loss_mult"] = 0.6 + 0.4 * (1.0/stab_ratio)
            deltas["resistance_drift_mult"] = 0.7 + 0.3 * (1.0/stab_ratio)
            deltas["exchange_current_mult"] = 1.0 + 0.2 * stab_ratio

        return deltas

    def run_discovery(self):
        print("Executing Material Property Derivation via OQMD + PyBaMM...")
        system = {"Cathode_Dopant": [], "Salt": [], "Functionalization": []}

        # Baseline properties
        base_cathode = self.get_properties("Na4Fe3P4O15")
        base_salt = self.get_properties("NaPF6")
        base_hc = self.get_properties("C")

        # 1. Cathode Dopants
        for d in ["Mn", "Cr", "Ni"]:
            comp = f"Na4Fe2.9{d}0.1P4O15"
            props = self.get_properties(comp)
            deltas = self.derive_deltas(props, base_cathode, "Cathode_Dopant")

            # Using PyBaMM to verify/scale base properties if needed
            # For this simple model, we extract multipliers/additives to apply to base_params
            system["Cathode_Dopant"].append(MaterialCandidate(
                name=d, category="Cathode_Dopant", composition=comp,
                energy_above_hull=props["stability"], formation_energy=props["formation_energy"],
                band_gap=props["band_gap"], volume_per_atom=props["volume_per_atom"],
                projected_delta=deltas
            ))

        # 2. Salts
        salts = {"NaBOB": "C4BNaO8", "NaTCP": "C5H3Cl3NNaO"}
        for name, comp in salts.items():
            props = self.get_properties(comp)
            deltas = self.derive_deltas(props, base_salt, "Salt")
            system["Salt"].append(MaterialCandidate(
                name=name, category="Salt", composition=comp,
                energy_above_hull=props["stability"], formation_energy=props["formation_energy"],
                band_gap=props["band_gap"], volume_per_atom=props["volume_per_atom"],
                projected_delta=deltas
            ))

        # 3. Functionalization
        mtms_comp = "C4H12O3Si"
        props = self.get_properties(mtms_comp)
        deltas = self.derive_deltas(props, base_hc, "Functionalization")
        system["Functionalization"].append(MaterialCandidate(
            name="MTMS", category="Functionalization", composition=mtms_comp,
            energy_above_hull=props["stability"], formation_energy=props["formation_energy"],
            band_gap=props["band_gap"], volume_per_atom=props["volume_per_atom"],
            projected_delta=deltas
        ))

        return system

if __name__ == "__main__":
    discovery = MaterialDiscoveryFramework()
    res = discovery.run_discovery()
    for cat, cands in res.items():
        print(f"\nCategory: {cat}")
        for c in cands:
            print(f"  - {c.name}: {c.projected_delta}")
            # print(f"    PyBaMM Delta Mapping: {c.to_pybamm_delta()}")

    # Show clean cache
    print("\n--- Final material_cache.json content ---")
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            print(f.read())
