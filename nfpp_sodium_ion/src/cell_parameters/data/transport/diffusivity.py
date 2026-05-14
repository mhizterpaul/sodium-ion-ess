from dataclasses import dataclass


@dataclass
class DiffusivityModel:
    reference_diffusivity_m2_s: float = 1e-14
    activation_energy_j_mol: float = 30000.0

    def effective_diffusivity(self, temperature_k: float, porosity: float) -> float:
        return self.reference_diffusivity_m2_s * porosity * (temperature_k / 298.15) ** 1.5

    def as_dict(self) -> dict:
        return {
            "reference_diffusivity_m2_s": self.reference_diffusivity_m2_s,
            "activation_energy_j_mol": self.activation_energy_j_mol,
        }
