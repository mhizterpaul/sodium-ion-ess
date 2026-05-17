import pybamm
import numpy as np
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

def verify():
    params_dict = get_parameter_values()
    params = pybamm.ParameterValues(params_dict)

    try:
        model = pybamm.sodium_ion.DFN()
    except AttributeError:
        model = pybamm.lithium_ion.DFN()

    # 0.1C Discharge
    params["Current function [A]"] = 1.0

    sim = pybamm.Simulation(model, parameter_values=params)
    sol = sim.solve([0, 3600*12])

    print(f"Initial Voltage: {sol['Terminal voltage [V]'].data[0]:.3f} V")
    print(f"Discharge Capacity (0.1C): {sol['Discharge capacity [A.h]'].data[-1]:.3f} Ah")

if __name__ == "__main__":
    verify()
