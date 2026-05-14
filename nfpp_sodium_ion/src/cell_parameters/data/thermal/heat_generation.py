from dataclasses import dataclass


@dataclass
class HeatGenerationModel:
    reaction_heat_factor: float = 0.5
    ohmic_heat_factor: float = 0.3
    polarization_heat_factor: float = 0.2

    def total_heat(self, reaction_heat: float, ohmic_heat: float, polarization_heat: float) -> float:
        return reaction_heat * self.reaction_heat_factor + ohmic_heat * self.ohmic_heat_factor + polarization_heat * self.polarization_heat_factor

    def as_dict(self) -> dict:
        return {
            "reaction_heat_factor": self.reaction_heat_factor,
            "ohmic_heat_factor": self.ohmic_heat_factor,
            "polarization_heat_factor": self.polarization_heat_factor,
        }
