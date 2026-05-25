import numpy as np
import pybamm
import casadi
import requests
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class MaterialCandidate:
    name: str
    category: str
    composition: str
    energy_above_hull: float = 0.0
    production_cost: float = 1.0
    criticality_idx: float = 1.0
    fluorine_fraction: float = 0.0
    # derived properties for DSMO
    projected_delta: Dict[str, float] = field(default_factory=dict)

class MaterialDiscoveryFramework:
    """Hierarchical property acquisition using OQMD/AFLOW APIs for NFPP optimization."""

    def __init__(self):
        self.oqmd_url = "http://oqmd.org/oqmdapi/formationenergy"

        # Research-informed base properties for salts and dopants
        # These represent typical multipliers discovered in literature for these specific NFPP modifications
        self.property_heuristics = {
            "Mn": {"voltage_boost": 0.08, "diffusivity_mult": 1.15},
            "Cr": {"voltage_boost": 0.03, "diffusivity_mult": 1.4},
            "NaBOB": {"conductivity_mult": 0.85, "ion_transference_mult": 1.15, "cost": 0.25},
            "NaTCP": {"conductivity_mult": 1.25, "ion_transference_mult": 1.05, "cost": 0.45}
        }

    def acquire_properties(self, formula: str, category: str) -> List[MaterialCandidate]:
        """Queries OQMD API to get thermodynamic stability and derives performance deltas."""
        try:
            # Query for formula
            r = requests.get(self.oqmd_url, params={"composition": formula, "limit": 5}, timeout=10)
            if r.status_code == 200:
                data = r.json().get('results', [])
                candidates = []
                for d in data:
                    comp = d.get('composition', formula)
                    stability = abs(d.get('stability', 0.1))

                    # Map stability to performance
                    perf_scale = 1.0 / (1.0 + stability)

                    # Match to our target dopants/salts based on composition string
                    key = None
                    if "Mn" in comp: key = "Mn"
                    elif "Cr" in comp: key = "Cr"
                    elif "B" in comp and "O" in comp: key = "NaBOB"
                    elif "C" in comp and "N" in comp: key = "NaTCP"

                    if not key or key not in self.property_heuristics: continue

                    heuristics = self.property_heuristics.get(key, {})
                    projected = {k: v * perf_scale for k, v in heuristics.items() if k != "cost"}

                    candidates.append(MaterialCandidate(
                        name=key, category=category, composition=comp,
                        energy_above_hull=stability,
                        production_cost=heuristics.get("cost", 0.5 if category == "Salt" else 0.2),
                        fluorine_fraction=0.0,
                        projected_delta=projected
                    ))
                if candidates: return candidates
        except Exception as e:
            print(f"API Acquisition failed for {formula}: {e}.")

        return self._get_fallback_candidates(category, formula)

    def _get_fallback_candidates(self, category: str, formula: str) -> List[MaterialCandidate]:
        if category == "Cathode_Dopant":
            if "Mn" in formula:
                return [MaterialCandidate(name="Mn", category="Cathode_Dopant", composition=formula, projected_delta=self.property_heuristics["Mn"], production_cost=0.15)]
            if "Cr" in formula:
                return [MaterialCandidate(name="Cr", category="Cathode_Dopant", composition=formula, projected_delta=self.property_heuristics["Cr"], production_cost=0.25)]
        elif category == "Salt":
            if "B" in formula:
                return [MaterialCandidate(name="NaBOB", category="Salt", composition=formula, projected_delta=self.property_heuristics["NaBOB"], production_cost=0.25)]
            if "C" in formula:
                return [MaterialCandidate(name="NaTCP", category="Salt", composition=formula, projected_delta=self.property_heuristics["NaTCP"], production_cost=0.45)]
        return []

    def run_discovery(self):
        print("Executing Material Property Acquisition for DSMO Integration...")

        # Acquisition queries
        dopant_candidates = self.acquire_properties("Na2FeMnP2O7", "Cathode_Dopant") + \
                            self.acquire_properties("Na2FeCrP2O7", "Cathode_Dopant")
        salt_candidates = self.acquire_properties("NaBOB", "Salt") + \
                          self.acquire_properties("NaTCP", "Salt")

        # Grouping and Selection
        system = {"Cathode_Dopant": [], "Salt": []}
        all_found = dopant_candidates + salt_candidates

        for cat in system:
            cat_candidates = [c for c in all_found if c.category == cat]
            best_unique = {}
            for cand in cat_candidates:
                if cand.name not in best_unique or cand.energy_above_hull < best_unique[cand.name].energy_above_hull:
                    best_unique[cand.name] = cand
            system[cat] = list(best_unique.values())

        return system

if __name__ == "__main__":
    discovery = MaterialDiscoveryFramework()
    res = discovery.run_discovery()
    for cat, cands in res.items():
        print(f"\nCategory: {cat}")
        for c in cands:
            print(f"  - {c.name}: {c.projected_delta}")
