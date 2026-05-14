from dataclasses import dataclass


@dataclass
class HardCarbonAnodeParameters:
    active_material_fraction: float = 0.88
    conductive_carbon_fraction: float = 0.06
    binder_fraction: float = 0.06
    practical_capacity_mAh_g: float = 275.0
    current_collector: str = "Copper"

    def composition(self) -> dict:
        return {
            "active_material": self.active_material_fraction,
            "conductive_carbon": self.conductive_carbon_fraction,
            "binder": self.binder_fraction,
        }

    def as_dict(self) -> dict:
        return {
            "composition": self.composition(),
            "practical_capacity_mAh_g": self.practical_capacity_mAh_g,
            "current_collector": self.current_collector,
        }
