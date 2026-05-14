from dataclasses import dataclass


@dataclass
class Constants:
    R: float = 8.3145
    F: float = 96485.3329
    T_ref: float = 298.15
    sigma_al: float = 3.5e7
    sigma_cu: float = 5.96e7
    epsilon_0: float = 8.854e-12

    def as_dict(self) -> dict:
        return {
            "R": self.R,
            "F": self.F,
            "T_ref": self.T_ref,
            "sigma_al": self.sigma_al,
            "sigma_cu": self.sigma_cu,
            "epsilon_0": self.epsilon_0,
        }
