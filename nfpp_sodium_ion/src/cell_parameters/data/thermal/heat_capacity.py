from dataclasses import dataclass


@dataclass
class HeatCapacityModel:
    reference_cp_j_kg_k: float = 900.0

    def specific_heat(self, temperature_k: float) -> float:
        return self.reference_cp_j_kg_k * (1 + 0.001 * (temperature_k - 298.15))

    def as_dict(self) -> dict:
        return {"reference_cp_j_kg_k": self.reference_cp_j_kg_k}
