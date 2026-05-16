from dataclasses import dataclass

@dataclass
class CellParameters:
    # Ref: paper.md
    name: str = "NFPP Sodium-ion Pouch Cell"
    nominal_voltage: float = 3.1
    capacity_ah: float = 10.0
    form_factor: str = "stacked pouch"
    cathode_collector_thickness_um: float = 15.0
    anode_collector_thickness_um: float = 10.0
    separator_thickness_um: float = 20.0
    casing_thickness_um: float = 40.0
    number_of_layers: int = 42

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "nominal_voltage": self.nominal_voltage,
            "capacity_ah": self.capacity_ah,
            "number_of_layers": self.number_of_layers,
        }
