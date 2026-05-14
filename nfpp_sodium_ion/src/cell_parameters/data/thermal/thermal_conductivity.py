from dataclasses import dataclass


@dataclass
class ThermalConductivityModel:
    reference_k_w_m_k: float = 0.2

    def conductivity(self, temperature_k: float) -> float:
        return self.reference_k_w_m_k * (1 + 0.001 * (temperature_k - 298.15))

    def as_dict(self) -> dict:
        return {"reference_k_w_m_k": self.reference_k_w_m_k}
