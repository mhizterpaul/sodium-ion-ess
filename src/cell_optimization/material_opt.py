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

    # Stage C: Thermodynamics (DFT)
    formation_energy: float = 0.0
    energy_above_hull: float = 0.0 # meV/atom

    # Stage B: Electrochemical/Structural
    ionic_radius: float = 0.0 # Angstrom
    charge: int = 2
    ionic_conductivity: float = 0.0
    diffusion_barrier: float = 0.0
    voltage_shift: float = 0.0

    # Stage E: Multi-Objective Metrics
    production_cost: float = 1.0      # USGS based
    criticality_idx: float = 1.0      # IEA based
    fluorine_fraction: float = 0.0    # F burden
    safety_score: float = 1.0         # S

    # Stage D: Projection Delta for DFN
    projected_delta: Dict[str, float] = field(default_factory=dict)
    rank_score: float = 0.0

class CompatibilityEngine:
    """Stage B: Hard NFPP Architectural Constraints."""
    def __init__(self):
        self.r_fe = 0.78

    def screen(self, c: MaterialCandidate) -> bool:
        # 1. Fluorine constraint (Nonlinear threshold)
        if c.fluorine_fraction > 0.3:
            return False # Unstable SEI regime

        if c.category == "Cathode_Dopant":
            # 2. Ionic radius compatibility (0.65 - 0.88 A)
            if not (0.65 <= c.ionic_radius <= 0.88):
                return False
            # 3. Charge balance: Dopant must preserve lattice neutrality
            # (Heuristic: 2+ or 3+ are OK for Fe site)
            if c.charge > 3:
                return False # Needs Na vacancy compensation (Complex)
            # 4. Voltage preservation: |dV| < 0.15V
            if abs(c.voltage_shift) > 0.15:
                return False

        elif c.category == "Salt":
            if c.ionic_conductivity < 0.5: # mS/cm approx
                return False

        return True

class MaterialDiscoveryFramework:
    """Hierarchical co-optimization pipeline (A-F)."""

    def __init__(self):
        self.engine = CompatibilityEngine()

        # Mapping Tables: Material -> DFN Delta
        self.dopant_map = {
            "Mn": {"diffusivity": 1.1, "exchange_current": 1.05},
            "Ti": {"diffusivity": 0.95, "cycle_life": 1.3},
            "Mg": {"strain": 0.9, "stability": 1.2}
        }
        self.salt_map = {
            "NaDFOB": {"conductivity": 0.9, "SEI_stability": 1.3},
            "NaFSI":  {"conductivity": 1.2, "thermal_stability": 1.4},
            "NaClO4": {"conductivity": 0.85, "cost": 0.5}
        }
        self.solvent_map = {
            "PC":     {"diffusion_electrolyte": 1.0},
            "EC_PC":  {"SEI_quality": 1.2},
            "PC_EMC": {"mobility": 1.3}
        }

    def stage_a_acquisition(self) -> List[MaterialCandidate]:
        """Fetch concrete realistic materials."""
        # This replaces mock discovery with the requested compatible set
        return [
            # Dopants
            MaterialCandidate("Mn", "Cathode_Dopant", "Na2Fe0.9Mn0.1P2O7", ionic_radius=0.83, charge=2, energy_above_hull=10, production_cost=1.5, criticality_idx=1.2, projected_delta=self.dopant_map["Mn"]),
            MaterialCandidate("Ti", "Cathode_Dopant", "Na2Fe0.95Ti0.05P2O7", ionic_radius=0.86, charge=4, energy_above_hull=5, voltage_shift=0.2, projected_delta=self.dopant_map["Ti"]), # Charge/Volt Fail
            MaterialCandidate("Mg", "Cathode_Dopant", "Na2Fe0.92Mg0.08P2O7", ionic_radius=0.72, charge=2, energy_above_hull=8, production_cost=1.1, criticality_idx=1.0, projected_delta=self.dopant_map["Mg"]),

            # Salts
            MaterialCandidate("NaDFOB", "Salt", "NaB(C2O4)F2", ionic_conductivity=0.8, fluorine_fraction=0.1, production_cost=5.0, criticality_idx=1.2, projected_delta=self.salt_map["NaDFOB"]),
            MaterialCandidate("NaFSI", "Salt", "NaFSI", ionic_conductivity=1.2, fluorine_fraction=0.25, production_cost=6.0, criticality_idx=1.8, projected_delta=self.salt_map["NaFSI"]),
            MaterialCandidate("NaPF6", "Salt", "NaPF6", ionic_conductivity=1.0, fluorine_fraction=0.6, production_cost=8.0, criticality_idx=2.5, projected_delta=self.salt_map.get("NaPF6", {})), # Fluorine Fail
            MaterialCandidate("NaClO4", "Salt", "NaClO4", ionic_conductivity=0.7, fluorine_fraction=0.0, production_cost=2.0, criticality_idx=1.1, projected_delta=self.salt_map["NaClO4"]),

            # Solvents
            MaterialCandidate("PC", "Solvent", "Propylene Carbonate", energy_above_hull=2, production_cost=1.0, fluorine_fraction=0.0, projected_delta=self.solvent_map["PC"]),
            MaterialCandidate("EC_PC", "Solvent", "EC+PC 1:1", energy_above_hull=1, production_cost=1.2, fluorine_fraction=0.0, projected_delta=self.solvent_map["EC_PC"]),
            MaterialCandidate("PC_EMC", "Solvent", "PC+EMC", energy_above_hull=3, production_cost=2.0, fluorine_fraction=0.0, projected_delta=self.solvent_map["PC_EMC"]),
        ]

    def dominates(self, a: MaterialCandidate, b: MaterialCandidate):
        """Pareto dominance: a dominates b if no objective is worse and at least one is better."""
        return (
            a.production_cost <= b.production_cost and
            a.criticality_idx <= b.criticality_idx and
            a.fluorine_fraction <= b.fluorine_fraction and
            (
                a.production_cost < b.production_cost or
                a.criticality_idx < b.criticality_idx or
                a.fluorine_fraction < b.fluorine_fraction
            )
        )

    def get_pareto_front(self, candidates: List[MaterialCandidate]) -> List[MaterialCandidate]:
        """Stage E: Pareto Front Selection."""
        front = []
        for a in candidates:
            if not any(self.dominates(b, a) for b in candidates if b.category == a.category):
                front.append(a)
        return front

    def run_discovery(self):
        print("Material Discovery Loop Started...")
        raw = self.stage_a_acquisition()

        # Stage C & B: Filtering
        valid = [c for c in raw if c.energy_above_hull < 50 and self.engine.screen(c)]

        # Stage E: Pareto Selection
        best_materials = self.get_pareto_front(valid)

        system = {}
        for cat in ["Cathode_Dopant", "Salt", "Solvent"]:
            matches = [m for m in best_materials if m.category == cat]
            if matches:
                # If multiple on front, pick one with lowest E_hull
                system[cat] = sorted(matches, key=lambda x: x.energy_above_hull)[0]
                print(f"  Selected {cat}: {system[cat].name}")

        return system

if __name__ == "__main__":
    framework = MaterialDiscoveryFramework()
    framework.run_discovery()
