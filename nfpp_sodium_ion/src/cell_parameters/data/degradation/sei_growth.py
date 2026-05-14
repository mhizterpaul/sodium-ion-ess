from dataclasses import dataclass


@dataclass
class SeiGrowthModel:
    rate_constant: float = 1e-10

    def growth_rate(self, current_density_a_m2: float, time_s: float) -> float:
        return self.rate_constant * abs(current_density_a_m2) * time_s

    def as_dict(self) -> dict:
        return {"rate_constant": self.rate_constant}
