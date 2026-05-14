from dataclasses import dataclass


@dataclass
class ChemistryParameters:
    active_material: str = "Na2FePO4P2O7"
    anode_material: str = "hard carbon"
    electrolyte_salt_primary: str = "NaPF6"
    electrolyte_salt_secondary: str = "NaDFOB"
    solvent_system: str = "EC:PC 1:1"
    fem_additives: dict = None

    def __post_init__(self):
        if self.fem_additives is None:
            self.fem_additives = {"FEC": 0.03, "VC": 0.02}

    def as_dict(self) -> dict:
        return {
            "active_material": self.active_material,
            "anode_material": self.anode_material,
            "electrolyte_salt_primary": self.electrolyte_salt_primary,
            "electrolyte_salt_secondary": self.electrolyte_salt_secondary,
            "solvent_system": self.solvent_system,
            "electrolyte_additives": self.fem_additives,
        }
