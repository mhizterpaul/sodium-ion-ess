# NFPP Sodium-Ion Energy Storage System

Complete scaffold and boilerplate for NFPP (sodium iron pyrophosphate) sodium-ion battery parameter package, validation models, and FEniCSx mechanics coupling.

## Module Overview

### 1. PyBaMM Parameter Package (`src/nfpp_sodium_ion/`)

Provides a **PyBaMM-registered parameter set** for NFPP sodium-ion cells.

**Installation & Registration:**
```bash
cd src/nfpp_sodium_ion
pip install -e .
```

This registers the entry point:
```
[project.entry-points.pybamm_parameter_sets]
nfpp_sodium_ion = "nfpp_sodium_ion:get_parameter_values"
```

**Usage in PyBaMM:**
```python
import pybamm

# Parameter set auto-discovered via entry point
param = pybamm.ParameterValues("nfpp_sodium_ion")
model = pybamm.lithium_ion.DFN()
# ... standard PyBaMM workflow
```

**Parameter Modules:**
- **base**: Cell geometry, chemistry identity, fundamental constants
- **electrodes**: Cathode (NFPP 85%), anode (hard carbon 88%), separator specs
- **electrolyte**: NaPF6/NaDFOB blend with EC:PC solvent and FEC/VC additives
- **transport**: Diffusivity and conductivity models with temperature dependence
- **thermal**: Heat generation sources, capacity, and conductivity
- **kinetics**: Reaction rates and exchange current density
- **degradation**: SEI/CEI growth and sodium-equivalent loss models

### 2. Validation Models (`src/evaluation/`)

Three coupled evaluation/validation models aligned with models.md requirements:

#### Model 1: ElectrochemicalThermalDriverModel
- **Framework**: PyBaMM DFN (Doyle-Fuller-Newman)
- **Inputs**: Current profile, ambient conditions
- **Outputs**: SOC trajectory, SOH degradation, heat generation rate Q(t)
- **Purpose**: Resolves electrochemical behavior and provides thermal forcing function

#### Model 2: ThermalFieldModel
- **Framework**: PyBaMM thermal modules or custom FEniCSx
- **Modes**: 
  - Resolved (spatial PDE): T(x,t) = temperature field through cell
  - Lumped (ODE): T(t) = single temperature for reduced-order model
- **Inputs**: Q(t) from driver, boundary conditions (ambient, convection)
- **Outputs**: Temperature distribution and evolution
- **Purpose**: Propagates electrochemical heat to mechanical domain

#### Model 3: ThermoelasticStrainModel
- **Framework**: FEniCSx (dolfinx) via mechanics module
- **Coupling**: T(x,t) → thermal expansion strain, SOC → swelling strain
- **Outputs**: 
  - Strain intensity evolution ε_int(t)
  - Critical strain envelope (failure criterion)
  - Endurance metrics: n_crit (cycles to failure), t_crit (time to failure)
- **Purpose**: Evaluates structural integrity under coupled loading

### 3. FEniCSx Mechanics Module (`src/mechanics/`)

**Purpose**: Bridge between parameter objects and dolfinx.fem.FunctionSpace

**Classes:**
- `ElasticModuliSpace`: Young's modulus E, Poisson ratio ν → Lamé parameters λ, μ
- `ThermalExpansionSpace`: Thermal expansion α → thermal strain ε_α(T)
- `SwellingCoefficientModel`: SOC swelling coefficient → ε_swelling(SOC)
- `ThermoelasticProblem`: Assembles weak form and solves coupled elasticity problem
- `ParameterCompatibleMechanicsInterface`: Maps nfpp_sodium_ion parameters directly to FEniCSx

**Usage:**
```python
from nfpp_sodium_ion import get_parameter_values
from mechanics import ParameterCompatibleMechanicsInterface

param_set = get_parameter_values()
problem = ParameterCompatibleMechanicsInterface.from_parameter_set(param_set)
# problem is a fully configured ThermoelasticProblem ready to solve in FEniCSx
```

## Installation & Setup

### Root Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install PyBaMM parameter package (registers entry point)
pip install -e src/nfpp_sodium_ion/
```

### Quick Test
```python
# Test parameter loading
from nfpp_sodium_ion import get_parameter_values

params = get_parameter_values()
print("Cathode capacity:", params["cathode"].theoretical_capacity_mAh_g)
print("Anode capacity:", params["anode"].practical_capacity_mAh_g)

# Test evaluation models
from evaluation import ElectrochemicalThermalDriverModel, ThermalFieldModel, ThermoelasticStrainModel

echem_model = ElectrochemicalThermalDriverModel()
thermal_model = ThermalFieldModel()
mech_model = ThermoelasticStrainModel()
```

## System Requirements (from models.md)

### Electrochemical Core
- **Cathode**: NFPP with reversible Na intercalation (~95–100 mAh/g)
- **Anode**: Hard carbon with adsorption/intercalation (250–300 mAh/g)
- **Electrolyte**: NaPF6 + NaDFOB in EC:PC with FEC/VC additives
- **Cell Format**: 10 Ah stacked pouch, 3.0–3.2 V nominal

### Model Validation Framework
1. **Electrochemical-Thermal**: Coupled SOC, SOH, Q(t) evolution
2. **Thermal Transport**: Spatial-temporal temperature field T(x,t) (or lumped T(t))
3. **Structural Integrity**: Thermoelastic strain under coupled loading
   - Strain drivers: thermal expansion, SOC swelling, SOH stiffness degradation
   - Failure criterion: ε_crit = min(strain at damage initiation)
   - Endurance: (n_crit, t_crit) = f(ε_int)

## Key Boilerplate Features

✅ **Dataclass-based parameter objects** with `as_dict()` serialization  
✅ **PyBaMM entry point registration** via pyproject.toml  
✅ **FEniCSx function space compatibility** for direct parameter-to-FEM mapping  
✅ **Three validation model scaffolds** with detailed docstrings  
✅ **Temperature-dependent transport** (diffusivity, conductivity)  
✅ **Coupled thermal-mechanical forcing** (T → ε_thermal, SOC → ε_swelling)  
✅ **Failure criterion and endurance metrics** for structural assessment  

## Future Extensions

- Implement full DFN solver in electrochemical_thermal_driver.py
- Add resolved thermal PDE solver in ThermalFieldModel
- Implement FEniCSx weak-form assembly and timestepper in ThermoelasticProblem
- Add parameter sensitivity analysis tools
- Experimental data validation workflows
- Post-processing and visualization utilities

## References

- models.md: System architecture and validation framework specification
- PyBaMM parameter set conventions: https://pybamm.readthedocs.io/
- FEniCSx mechanics tutorial: https://docs.fenicsproject.org/

---

**Status**: Scaffold & boilerplate complete. Ready for implementation of solvers and coupling strategies.
