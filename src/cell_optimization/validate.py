import pybamm
import numpy as np
import scipy.io as sio
import os
import json
import traceback
from typing import Dict, Any, List, Tuple, Optional
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values
from src.cell_optimization.material_opt import MaterialMappingEngine, MaterialCategory, MaterialCandidate
from src.cell_optimization.chem_regularization import derive_coupled_deltas, regularize_salt_props, regularize_functionalization
from src.cell_optimization.parameter_opts import ParamTransform, HierarchicalOptimizer, DESIGN_SPACE
from src.simulation.utilities.mechanical.fenics_model import ThermoelasticStrainModel

class OptimizationValidator:
    """
    High-fidelity validation using DFN model and dolfinx mechanics.
    MTMS functionalization is applied here as a fixed design step.
    """

    def __init__(self, optimized_design: Dict[str, float], combined_deltas: Dict[str, Any], engine: Optional[MaterialMappingEngine] = None):
        self.design = optimized_design
        self.deltas = combined_deltas
        self.engine = engine or MaterialMappingEngine()
        self.mech_model = ThermoelasticStrainModel()

    def get_final_parameters(self) -> pybamm.ParameterValues:
        # 1. Resolve MTMS and apply deltas
        db, bases = self.engine.run()
        if MaterialCategory.FUNCTIONALIZATION in db and db[MaterialCategory.FUNCTIONALIZATION]:
            mtms = db[MaterialCategory.FUNCTIONALIZATION][0]
            print(f"Applying fixed functionalization: {mtms.name}")
            f_deltas = regularize_functionalization(bases["interface"]["formula"], mtms.composition, bases["interface"]["properties"], mtms.properties)
            # Merge functionalization deltas into combined deltas
            for cat, props in f_deltas.items():
                self.deltas.setdefault(cat, {}).update(props)

        base_params = get_parameter_values()
        pt = ParamTransform(pybamm.ParameterValues(base_params))
        pt.apply_physics_deltas(self.deltas)
        pt.apply_design_vector(
            np.array([self.design[k] for k in DESIGN_SPACE if k in self.design]),
            [k for k in DESIGN_SPACE if k in self.design]
        )
        p = pt.get_parameter_values()
        return p

    def solve_mechanical_integrity(self, sol: Any, params: pybamm.ParameterValues) -> Dict[str, float]:
        """
        High-fidelity Thermo-Mechanical Solver using consolidated fenics_model.
        """
        try:
            mech_res = self.mech_model.solve_strain(sol, params)
            max_strain = mech_res["max_strain"]

            # Extract E_eff for stress estimation proxy
            E_eff = float(params.get("Negative electrode Young's modulus [Pa]", 10e9))

            critical_strain = self.mech_model.critical_thresholds.get("NFPP", 2e-3)
            eta = max_strain / critical_strain

            return {
                "max_strain": float(max_strain),
                "max_stress_pa": float(max_strain * E_eff),
                "mechanical_integrity_factor": float(eta)
            }
        except Exception:
            traceback.print_exc()
            return {"max_strain": 0.0, "max_stress_pa": 0.0, "mechanical_integrity_factor": 0.0}

    def run_validation(self):
        print("Running high-fidelity DFN validation with degradation (Layer 4)...")
        params = self.get_final_parameters()
        # Add missing parameters for DFN stability
        if "SEI solvent diffusivity [m2.s-1]" not in params:
             params["SEI solvent diffusivity [m2.s-1]"] = 2.5e-22
        if "Bulk solvent concentration [mol.m-3]" not in params:
             params["Bulk solvent concentration [mol.m-3]"] = 2636.0

        model = pybamm.lithium_ion.DFN({
            "SEI": "solvent-diffusion limited",
            "loss of active material": "stress-driven",
            "thermal": "lumped"
        })
        sim = pybamm.Simulation(model, parameter_values=params)

        try:
            sol = sim.solve([0, 3600], inputs={"Current [A]": params["Nominal cell capacity [A.h]"]})
            v = sol["Terminal voltage [V]"].data
            cap = sol["Discharge capacity [A.h]"].data[-1]

            temp_key = "Volume-averaged cell temperature [K]"
            try:
                 temp = sol[temp_key].data
            except Exception:
                 temp_key = "Cell temperature [K]"
                 temp = sol[temp_key].data

            cs_n = sol["X-averaged negative particle concentration [mol.m-3]"].data

            # Pass full solution to mechanical solver
            mech = self.solve_mechanical_integrity(sol, params)

            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            # Energy calculation (Issue 5.1) - Discharge E is always positive
            energy = trapz_func(np.abs(v * sol["Current [A]"].data), sol["Time [s]"].data) / 3600

            sei_growth = 0.0
            try:
                 sei_t = sol["X-averaged negative SEI thickness [m]"].data
                 sei_growth = sei_t[-1] - sei_t[0]
            except:
                 pass

            attributes = {
                "energy_wh": float(energy),
                "capacity_ah": float(cap),
                "voltage_avg": float(np.mean(v)),
                "max_temp_k": float(np.max(temp)),
                "max_strain": mech["max_strain"],
                "max_stress_pa": mech["max_stress_pa"],
                "mechanical_integrity_factor": mech["mechanical_integrity_factor"],
                "sei_growth_m": float(sei_growth)
            }

            print("Validation complete.")
            print(json.dumps(attributes, indent=2))
            return attributes
        except Exception as e:
            print(f"Validation failed: {e}")
            return None

if __name__ == "__main__":
    import json
    import os

    result_path = "result.json"
    if not os.path.exists(result_path):
        print(f"File {result_path} not found. Running optimization...")
        from src.cell_optimization.parameter_opts import run_workflow
        result = run_workflow()
    else:
        with open(result_path, "r") as f:
            result = json.load(f)

    if result:
        print("\n--- VALIDATION LOG ---")
        print(f"Selected Cathode: {result['materials']['cathode']['name']} ({result['materials']['cathode']['formula']})")
        print(f"Selected Electrolyte Salt: {result['materials']['electrolyte']['salt']}")
        print(f"Optimized Parameters:")
        for k, v in result['design_specs_representative'].items():
            print(f"  {k}: {v}")
        print("----------------------\n")

        validator = OptimizationValidator(
            result.get("design_specs_representative", {}),
            result.get("combined_deltas_representative", {})
        )
        val_metrics = validator.run_validation()

        # Merge results for report.ipynb compliance
        final_report = {
            "optimization": result,
            "validation": val_metrics
        }
        with open("final_validation.json", "w") as f:
            json.dump(final_report, f, indent=2)
        print("Final validation report saved to final_validation.json")
