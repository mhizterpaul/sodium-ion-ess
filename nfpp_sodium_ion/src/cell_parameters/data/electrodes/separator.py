from dataclasses import dataclass


@dataclass
class SeparatorParameters:
    material: str = "polyolefin trilayer"
    thickness_um: float = 20.0
    porosity: float = 0.42
    ionic_conductivity_S_cm: float = 1e-4

    def as_dict(self) -> dict:
        return {
            "material": self.material,
            "thickness_um": self.thickness_um,
            "porosity": self.porosity,
            "ionic_conductivity_S_cm": self.ionic_conductivity_S_cm,
        }
