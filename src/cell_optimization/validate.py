import pybamm
import numpy as np
import scipy.io as sio
import os
import math
from typing import Dict, Any, List, Tuple
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCandidate
from src.cell_optimization.chem_regularization import regularize_material

class OptimizationValidator:
    """
    Independent Multi-physics Validator.
    Orchestrates the 3-layer flow for validation.
    """

    def __init__(self, optimized_design: Dict[str, float], material_info: List[Tuple[MaterialCandidate, Dict[str, Any]]]):
        self.design = optimized_design
        self.materials_reg = material_info

    def get_final_parameters(self) -> pybamm.ParameterValues:
        base_params = get_parameter_values()
        # DSMO mapping logic
        from src.cell_optimization.parameter_opt import ParamTransform
        transform = ParamTransform(base_params)

        # 1. Apply design parameters
        for k, v in self.design.items():
            if k in transform.base:
                transform.base[k] = v

        # 2. Handle conductive carbon percolation
        if "Positive electrode conductive carbon fraction" in self.design:
            phi = self.design["Positive electrode conductive carbon fraction"]
            phi_c = 0.03
            cond_mult = max(((phi - phi_c) / (1 - phi_c + 1e-9)), 0.01)**1.8
            transform.add_multiplier("Positive electrode conductivity [S.m-1]", cond_mult / (0.08 / (1-phi_c))**1.8)

        # 3. Apply regularized material deltas
        for cand, reg in self.materials_reg:
            transform.apply_physics(reg, cand.category)

        params = transform.evaluate()
        if "Cell volume [m3]" not in params:
            params["Cell volume [m3]"] = 0.130 * 0.070 * 0.0003

        return params

    def run_validation(self):
        print("Running final Digital Twin validation (DFN Multi-physics)...")
        params = self.get_final_parameters()

        model = pybamm.lithium_ion.DFN({"thermal": "lumped"})
        solver = pybamm.CasadiSolver(mode="safe")
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)

        try:
            sol = sim.solve([0, 3600], inputs={"Current [A]": params["Nominal cell capacity [A.h]"]})
            v = sol["Terminal voltage [V]"].entries
            cap = sol["Discharge capacity [A.h]"].entries[-1]
            temp = sol["Cell temperature [K]"].entries
            trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
            energy = trapezoid(v, sol["Discharge capacity [A.h]"].entries)

            attributes = {
                "Energy_Wh": float(energy),
                "Capacity_Ah": float(cap),
                "Nominal_Voltage_V": float(np.mean(v)),
                "Max_Temp_K": float(np.max(temp)),
                "Energy_Density_Wh_kg": float(energy / 0.5)
            }

            print("Final Validated Attributes:")
            for k, v in attributes.items():
                print(f"  {k}: {v}")
            return attributes
        except Exception as e:
            print(f"Validation failed: {e}")
            return None

    def export_results(self, attributes, output_path="src/bms_design/cell_attributes.mat"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sio.savemat(output_path, {"cell_attributes": attributes})
        print(f"Cell attributes exported to {output_path}")

if __name__ == "__main__":
    engine = MaterialMappingEngine()
    db, bases = engine.run()

    # Example: Mn, NaBOB and MTMS
    candidates = [
        [m for m in db["Cathode_Dopant"] if m.name == "Mn"][0],
        [m for m in db["Salt"] if m.name == "NaBOB"][0],
        db["Functionalization"][0]
    ]

    # Regularization exclusively via physics layer
    materials_reg = []
    for c in candidates:
        if "Dopant" in c.category: b = bases["cathode"]
        elif "Salt" in c.category: b = bases["salt"]
        else: b = bases["interface"]
        materials_reg.append((c, regularize_material(c, b)))

    optimized_design = {
        "Positive electrode thickness [m]": 0.00015,
        "Negative electrode thickness [m]": 0.00015,
        "Positive electrode porosity": 0.3,
        "Negative electrode porosity": 0.3,
        "Separator porosity": 0.5,
        "Positive electrode active material volume fraction": 0.65,
        "Positive particle radius [m]": 1e-6,
        "Negative particle radius [m]": 5e-6,
        "Typical electrolyte concentration [mol.m-3]": 1000.0,
        "Positive electrode conductive carbon fraction": 0.08
    }

    validator = OptimizationValidator(optimized_design, materials_reg)
    res = validator.run_validation()
    if res:
        validator.export_results(res)
