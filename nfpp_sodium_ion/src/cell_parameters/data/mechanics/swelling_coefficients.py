from dataclasses import dataclass

@dataclass
class SwellingCoefficientModel:
    negative_electrode_swelling_coefficient: float = 0.1
    positive_electrode_swelling_coefficient: float = 0.05

    def as_dict(self) -> dict:
        return {
            "negative_electrode_swelling_coefficient": self.negative_electrode_swelling_coefficient,
            "positive_electrode_swelling_coefficient": self.positive_electrode_swelling_coefficient,
        }
