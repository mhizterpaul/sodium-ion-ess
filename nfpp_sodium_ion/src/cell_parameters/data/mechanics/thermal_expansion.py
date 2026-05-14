from dataclasses import dataclass


@dataclass
class ThermalExpansionModel:
    alpha_ref: float = 1e-5

    def expansion_coefficient(self, temperature_k: float) -> float:
        return self.alpha_ref * (1 + 0.0002 * (temperature_k - 298.15))

    def as_dict(self) -> dict:
        return {"alpha_ref": self.alpha_ref}
