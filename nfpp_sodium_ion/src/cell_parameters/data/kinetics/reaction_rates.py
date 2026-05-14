from dataclasses import dataclass


@dataclass
class ReactionRateModel:
    k0: float = 1e-11
    activation_energy_j_mol: float = 25000.0

    def rate_constant(self, temperature_k: float) -> float:
        return self.k0 * (temperature_k / 298.15) ** 0.5

    def as_dict(self) -> dict:
        return {"k0": self.k0, "activation_energy_j_mol": self.activation_energy_j_mol}
