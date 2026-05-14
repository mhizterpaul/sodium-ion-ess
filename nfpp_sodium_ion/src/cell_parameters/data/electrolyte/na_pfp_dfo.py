from dataclasses import dataclass
from typing import Dict


@dataclass
class NaPfpDfoParameters:
    salt_primary: str = "NaPF6"
    salt_secondary: str = "NaDFOB"
    concentration_primary_mol_per_l: float = 1.0
    concentration_secondary_mol_per_l: float = 0.2
    solvent_system: str = "EC:PC 1:1"
    ionic_conductivity_mS_cm: float = 10.0
    additives: Dict[str, float] = None

    def __post_init__(self):
        if self.additives is None:
            self.additives = {"FEC": 0.03, "VC": 0.02}

    def as_dict(self) -> dict:
        return {
            "salt_primary": self.salt_primary,
            "salt_secondary": self.salt_secondary,
            "concentration_primary_mol_per_l": self.concentration_primary_mol_per_l,
            "concentration_secondary_mol_per_l": self.concentration_secondary_mol_per_l,
            "solvent_system": self.solvent_system,
            "ionic_conductivity_mS_cm": self.ionic_conductivity_mS_cm,
            "additives": self.additives,
        }
