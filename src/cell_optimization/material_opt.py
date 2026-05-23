import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json

@dataclass
class MaterialCandidate:
    name: str
    category: str # "Cathode_Dopant", "Anode_Dopant", "Salt", "Solvent"
    composition: str

    # Stage C: Thermodynamics
    formation_energy: float = 0.0
    energy_above_hull: float = 0.0 # meV/atom

    # Stage B: Electrochemical/Structural
    ionic_conductivity: float = 0.0
    diffusion_barrier: float = 0.0
    voltage_shift: float = 0.0

    # Stage E: Multi-Objective Vector J = [Ed, Lc, C, Rc, F, S]
    energy_density_gain: float = 0.0 # Ed
    cycle_life_impact: float = 1.0   # Lc
    production_cost: float = 1.0      # C (USGS based)
    criticality_idx: float = 1.0      # Rc (IEA based)
    fluorine_burden: float = 0.0      # F
    safety_score: float = 1.0         # S

    # Stage D: Projection Delta for DFN
    projected_delta: Dict[str, float] = field(default_factory=dict)

class CompatibilityEngine:
    """Stage B: NFPP Architectural Constraints."""
    def __init__(self):
        self.r_fe = 0.78
        self.delta_r_max = 0.15

    def screen(self, c: MaterialCandidate) -> bool:
        if c.category == "Cathode_Dopant":
            # 1. Structural: Ionic radius check
            dopant_radii = {"Mn": 0.83, "Ti": 0.86, "Mg": 0.72, "V": 0.79}
            r_d = dopant_radii.get(c.name, 1.0)
            if abs(r_d - self.r_fe) > self.delta_r_max: return False
            # 2. Voltage: |dV| < 0.15V
            if abs(c.voltage_shift) > 0.15: return False
        elif c.category == "Salt":
            # Conductivity check
            if c.ionic_conductivity < 0.1: return False
        return True

class MaterialDiscoveryFramework:
    """Hierarchical chemistry-screening and compatibility-ranking pipeline."""

    def __init__(self):
        self.engine = CompatibilityEngine()
        self.oqmd_url = "http://oqmd.org/oqmdapi/formationenergy"
        # IEA/USGS Reference Indices
        self.market_data = {
            "Na": {"price": 1.1, "crit": 1.1},
            "Fe": {"price": 1.0, "crit": 1.0},
            "Mn": {"price": 2.2, "crit": 1.5},
            "F":  {"price": 5.0, "crit": 2.5},
            "Li": {"price": 50.0, "crit": 5.0}
        }

    def stage_a_acquisition(self) -> List[MaterialCandidate]:
        print("Stage A: Candidate Acquisition (OQMD/AFLOW)...")
        # Logic to fetch and create MaterialCandidate objects
        return [
            MaterialCandidate("Mn", "Cathode_Dopant", "Na2Fe0.9Mn0.1P2O7", energy_above_hull=10, production_cost=2.0, criticality_idx=1.5, projected_delta={"diffusivity": 1.1}),
            MaterialCandidate("NaDFOB", "Salt", "NaB(C2O4)F2", energy_above_hull=15, ionic_conductivity=0.8, fluorine_fraction=0.1, production_cost=5.0, criticality_idx=1.2, projected_delta={"conductivity": 0.9}),
            MaterialCandidate("NaPF6", "Salt", "NaPF6", energy_above_hull=5, ionic_conductivity=1.0, fluorine_fraction=0.6, production_cost=8.0, criticality_idx=2.5, projected_delta={"conductivity": 1.0}),
        ]

    def stage_c_thermodynamic_screening(self, candidates: List[MaterialCandidate]):
        # E_hull < 50 meV/atom
        return [c for c in candidates if c.energy_above_hull < 50]

    def stage_b_compatibility_screening(self, candidates: List[MaterialCandidate]):
        return [c for c in candidates if self.engine.screen(c)]

    def stage_e_pareto_ranking(self, candidates: List[MaterialCandidate]):
        """Rank using Pareto Dominance on Objective Vector J."""
        # For simplicity in this implementation, we use a weighted score
        # that reflects the Pareto objectives: Max(Life, Ed, Safety), Min(Cost, Crit, F)
        for c in candidates:
            # Objective: Minimize J_score
            cost_factor = c.production_cost * c.criticality_idx
            burden_factor = 1.0 + 5.0 * c.fluorine_fraction
            stability_factor = 1.0 + 0.1 * c.energy_above_hull
            c.rank_score = cost_factor * burden_factor * stability_factor

        return sorted(candidates, key=lambda x: x.rank_score)

    def run_discovery(self):
        raw = self.stage_a_acquisition()
        stable = self.stage_c_thermodynamic_screening(raw)
        compatible = self.stage_b_compatibility_screening(stable)
        ranked = self.stage_e_pareto_ranking(compatible)

        # Stage D: Projection
        best_system = {}
        for cat in ["Cathode_Dopant", "Anode_Dopant", "Salt", "Solvent"]:
            matches = [c for c in ranked if c.category == cat]
            if matches:
                best_system[cat] = matches[0]
                print(f"  Selected {cat}: {matches[0].name} (Score: {matches[0].rank_score:.2f})")

        return best_system

if __name__ == "__main__":
    framework = MaterialDiscoveryFramework()
    framework.run_discovery()
