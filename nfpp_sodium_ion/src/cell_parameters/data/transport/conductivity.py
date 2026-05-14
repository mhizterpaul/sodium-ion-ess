from dataclasses import dataclass


@dataclass
class ConductivityModel:
    reference_conductivity_S_m: float = 1.0
    temperature_coefficient: float = 0.02

    def effective_conductivity(self, temperature_k: float, phase: str = "electrolyte") -> float:
        return self.reference_conductivity_S_m * (1 + self.temperature_coefficient * (temperature_k - 298.15) / 298.15)

    def as_dict(self) -> dict:
        return {
            "reference_conductivity_S_m": self.reference_conductivity_S_m,
            "temperature_coefficient": self.temperature_coefficient,
        }
