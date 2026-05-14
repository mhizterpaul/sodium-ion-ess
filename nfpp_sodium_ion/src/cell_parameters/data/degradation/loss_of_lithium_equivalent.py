from dataclasses import dataclass


@dataclass
class LossOfSodiumEquivalentModel:
    loss_rate_fraction_per_cycle: float = 1e-4

    def loss_per_cycle(self, cycles: int) -> float:
        return self.loss_rate_fraction_per_cycle * cycles

    def as_dict(self) -> dict:
        return {"loss_rate_fraction_per_cycle": self.loss_rate_fraction_per_cycle}
