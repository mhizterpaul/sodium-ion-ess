from dataclasses import dataclass


@dataclass
class SwellingCoefficientModel:
    swelling_coefficient: float = 5e-5

    def strain_from_soc(self, soc: float) -> float:
        return self.swelling_coefficient * soc

    def as_dict(self) -> dict:
        return {"swelling_coefficient": self.swelling_coefficient}
