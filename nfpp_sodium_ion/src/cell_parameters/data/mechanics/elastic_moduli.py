from dataclasses import dataclass


@dataclass
class ElasticModuliModel:
    youngs_modulus_pa: float = 2.0e9
    poisson_ratio: float = 0.3

    def as_dict(self) -> dict:
        return {
            "youngs_modulus_pa": self.youngs_modulus_pa,
            "poisson_ratio": self.poisson_ratio,
        }
