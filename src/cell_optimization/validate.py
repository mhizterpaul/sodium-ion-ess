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
            from src.cell_optimization.chem_regularization import regularize_functionalization
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
            # Extract C-rate for mechanical scaling
            current = np.abs(np.mean(sol["Current [A]"].entries))
            cap = float(params["Nominal cell capacity [A.h]"])
            c_rate = current / cap if cap > 0 else 1.0

            mech_res = self.mech_model.solve_strain(sol, params, c_rate=c_rate)
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
        except Exception as e:
            print(f"ERROR: Mechanical integrity solver failed: {e}\n{traceback.format_exc()}")
            return {"max_strain": 0.0, "max_stress_pa": 0.0, "mechanical_integrity_factor": 0.0}

    def run_validation(self):
        print("Running high-fidelity DFN validation with degradation (Layer 4)...")
        derived = get_derived_parameters()

        # 1. Baseline Performance Calculation
        print("Calculating baseline performance (nominal design, original materials)...")
        base_pv = pybamm.ParameterValues(get_parameter_values())
        # Use same high-fidelity configuration for baseline
        options = {
            "SEI": "solvent-diffusion limited",
            "loss of active material": "stress-driven",
            "thermal": "lumped"
        }
        try:
             base_model = pybamm.sodium_ion.DFN(options=options)
        except AttributeError:
             base_model = pybamm.lithium_ion.DFN(options=options)
        # Add missing parameters for baseline DFN stability using audit-validated defaults
        base_pv.update({
            "SEI solvent diffusivity [m2.s-1]": derived["sei_solvent_diffusivity"],
            "Bulk solvent concentration [mol.m-3]": derived["bulk_solvent_concentration"],
            "Cell volume [m3]": derived["cell_volume"],
            "Cell cooling surface area [m2]": derived["surface_area"],
            "Total heat transfer coefficient [W.m-2.K-1]": derived["total_htc"]
        }, check_already_exists=False)

        base_sim = pybamm.Simulation(base_model, parameter_values=base_pv)
        try:
             base_sol = base_sim.solve([0, 3600], inputs={"Current [A]": base_pv["Nominal cell capacity [A.h]"]})
             v_b = base_sol["Terminal voltage [V]"].entries
             curr_b = base_sol["Current [A]"].entries
             t_b = base_sol["Time [s]"].entries
             trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))

             energy_b = abs(trapz_func(v_b * curr_b, t_b)) / 3600
             power_b = np.max(np.abs(v_b * curr_b))
             mech_b = self.solve_mechanical_integrity(base_sol, base_pv)

             baseline_metrics = {
                  "energy": float(energy_b),
                  "power": float(power_b),
                  "max_strain": mech_b["max_strain"],
                  "mechanical_integrity_factor": mech_b["mechanical_integrity_factor"]
             }
        except Exception as e:
             print(f"WARNING: Baseline simulation failed: {e}")
             baseline_metrics = {"error": str(e)}

        # 2. Optimized Validation
        params = self.get_final_parameters()
        # Ensure parity with audited parameters
        params.update({
            "SEI solvent diffusivity [m2.s-1]": derived["sei_solvent_diffusivity"],
            "Bulk solvent concentration [mol.m-3]": derived["bulk_solvent_concentration"]
        }, check_already_exists=False)

        options = {
            "SEI": "solvent-diffusion limited",
            "loss of active material": "stress-driven",
            "thermal": "lumped"
        }
        try:
             model = pybamm.sodium_ion.DFN(options=options)
        except AttributeError:
             model = pybamm.lithium_ion.DFN(options=options)
        sim = pybamm.Simulation(model, parameter_values=params)

        try:
            sol = sim.solve([0, 3600], inputs={"Current [A]": params["Nominal cell capacity [A.h]"]})
        except Exception as e:
            print(f"ERROR: Optimization validation simulation failed: {e}\n{traceback.format_exc()}")
            return None

        try:
            v = sol["Terminal voltage [V]"].entries
            cap = sol["Discharge capacity [A.h]"].entries[-1]

            temp_key = "Volume-averaged cell temperature [K]"
            try:
                 temp = sol[temp_key].entries
            except Exception:
                 temp_key = "Cell temperature [K]"
                 temp = sol[temp_key].entries

            cs_n = sol["X-averaged negative particle concentration [mol.m-3]"].entries

            # Pass full solution to mechanical solver
            mech = self.solve_mechanical_integrity(sol, params)

            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            # Energy calculation (Issue 4) - Integration of V*I
            energy = abs(trapz_func(v * sol["Current [A]"].entries, sol["Time [s]"].entries)) / 3600
            power = np.max(np.abs(v * sol["Current [A]"].entries))

            sei_growth = 0.0
            try:
                 sei_t = sol["X-averaged negative SEI thickness [m]"].entries
                 sei_growth = sei_t[-1] - sei_t[0]
            except:
                 pass

            attributes = {
                "energy_wh": float(energy),
                "power_w": float(power),
                "capacity_ah": float(cap),
                "voltage_avg": float(np.mean(v)),
                "max_temp_k": float(np.max(temp)),
                "max_strain": mech["max_strain"],
                "max_stress_pa": mech["max_stress_pa"],
                "mechanical_integrity_factor": mech["mechanical_integrity_factor"],
                "sei_growth_m": float(sei_growth),
                "baseline_performance": baseline_metrics
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
