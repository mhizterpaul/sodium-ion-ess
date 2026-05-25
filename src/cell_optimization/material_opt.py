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
            # Mn: booster, Cr: stabilizer
            dopant_radii = {"Mn": 0.83, "Cr": 0.755}
            r_d = dopant_radii.get(c.name, 1.0)
            if not (0.65 <= r_d <= 0.88): return False
        return True

class MaterialDiscoveryFramework:
    """Hierarchical chemistry-screening using OQMD/AFLOW APIs."""

    def __init__(self):
        self.engine = CompatibilityEngine()
        self.oqmd_url = "http://oqmd.org/oqmdapi/formationenergy"

    def run_discovery(self):
        print("Executing Material Selection for DSMO...")

        # Specified materials
        candidates = [
            MaterialCandidate(name="Mn", category="Cathode_Dopant", composition="Na2Fe1-xMnxP2O7", production_cost=0.15, projected_delta={"voltage_boost": 0.1, "diffusivity_mult": 1.1}),
            MaterialCandidate(name="Cr", category="Cathode_Dopant", composition="Na2Fe1-xCrxP2O7", production_cost=0.25, projected_delta={"voltage_boost": 0.02, "diffusivity_mult": 1.3}),
            MaterialCandidate(name="NaBOB", category="Salt", composition="NaB(C2O4)2", production_cost=0.3, projected_delta={"conductivity_mult": 0.9, "ion_transference_mult": 1.1}),
            MaterialCandidate(name="NaTCP", category="Salt", composition="NaC4N3", production_cost=0.4, projected_delta={"conductivity_mult": 1.2, "ion_transference_mult": 1.05})
        ]

        # Filtering
        valid = [c for c in candidates if self.engine.screen(c)]

        # Group by category for DSMO
        system = {"Cathode_Dopant": [], "Salt": []}
        for c in valid:
            system[c.category].append(c)

        return system

if __name__ == "__main__":
    discovery = MaterialDiscoveryFramework()
    res = discovery.run_discovery()
    print(res)
