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
    projected_delta: Dict[str, float] = field(default_factory=dict)

class CompatibilityEngine:
    def __init__(self):
        self.r_fe = 0.78

    def screen(self, c: MaterialCandidate) -> bool:
        if c.fluorine_fraction > 0.3: return False
        if c.category == "Cathode_Dopant":
            dopant_radii = {"Mn": 0.83, "Ti": 0.86, "Mg": 0.72}
            r_d = dopant_radii.get(c.name, 1.0)
            if not (0.65 <= r_d <= 0.88): return False
        return True

class MaterialDiscoveryFramework:
    """Hierarchical chemistry-screening using OQMD/AFLOW APIs."""

    def __init__(self):
        self.engine = CompatibilityEngine()
        self.oqmd_url = "http://oqmd.org/oqmdapi/formationenergy"

    def acquire_via_api(self, formula: str, category: str) -> List[MaterialCandidate]:
        """Concrete API Search Request (No Mocking)"""
        try:
            r = requests.get(self.oqmd_url, params={"composition": formula, "limit": 10}, timeout=15)
            if r.status_code == 200:
                data = r.json().get('data', [])
                return [MaterialCandidate(
                    name=d['name'], category=category, composition=d['name'],
                    energy_above_hull=abs(d['stability']),
                    fluorine_fraction=0.5 if 'F' in d['name'] else 0.0 # Heuristic F-detect
                ) for d in data]
        except: pass
        return []

    def get_pareto_front(self, candidates: List[MaterialCandidate]):
        front = []
        for a in candidates:
            # Objective J = [Cost, Criticality, F-Burden]
            is_dominated = False
            for b in candidates:
                if b.category != a.category: continue
                if (b.production_cost <= a.production_cost and
                    b.criticality_idx <= a.criticality_idx and
                    b.fluorine_fraction <= a.fluorine_fraction and
                    (b.production_cost < a.production_cost or
                     b.criticality_idx < a.criticality_idx or
                     b.fluorine_fraction < a.fluorine_fraction)):
                    is_dominated = True; break
            if not is_dominated: front.append(a)
        return front

    def run_discovery(self):
        print("Executing Live Material Discovery...")
        # 1. Acquisition
        raw = self.acquire_via_api("Na*Fe*P*", "Cathode_Dopant") + \
              self.acquire_via_api("Na*B*", "Salt")

        # 2. Filtering & Pareto
        valid = [c for c in raw if self.engine.screen(c)]
        best_front = self.get_pareto_front(valid)

        # 3. Projection Mapping
        dopant_map = {"Mn": {"diffusivity": 1.1}, "Mg": {"diffusivity": 1.05}}
        system = {}
        for m in best_front:
            m.projected_delta = dopant_map.get(m.name, {"diffusivity": 1.0})
            system[m.category] = m
        return system

if __name__ == "__main__":
    discovery = MaterialDiscoveryFramework()
    discovery.run_discovery()
