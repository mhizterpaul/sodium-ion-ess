from dataclasses import dataclass
import numpy as np

@dataclass
class DiffusivityModel:
    # Ref: Typical polyanionic transport data
    reference_diffusivity_m2_s: float = 1e-14
    activation_energy_j_mol: float = 30000.0

    def effective_diffusivity(self, temperature_k: float, porosity: float) -> float:
        R = 8.314
        arrhenius = np.exp(self.activation_energy_j_mol / R * (1 / 298.15 - 1 / temperature_k))
        return self.reference_diffusivity_m2_s * arrhenius * porosity**1.5

    def as_dict(self) -> dict:
        return {
            "reference_diffusivity_m2_s": self.reference_diffusivity_m2_s,
            "activation_energy_j_mol": self.activation_energy_j_mol,
        }
