from dataclasses import dataclass


@dataclass
class ExchangeCurrentDensityModel:
    j0_ref_a_m2: float = 1.0

    def exchange_current_density(self, temperature_k: float, soc: float) -> float:
        return self.j0_ref_a_m2 * soc * (temperature_k / 298.15) ** 0.5

    def as_dict(self) -> dict:
        return {"j0_ref_a_m2": self.j0_ref_a_m2}
