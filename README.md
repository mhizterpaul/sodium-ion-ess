# DFN-Based NFPP Sodium-Ion Cell Optimization and Model-Based Battery Management System Design

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhizterpaul/sodium-ion-ess/blob/main/src/report.ipynb)

This repository implements a high-fidelity digital twin and optimization framework for Sodium Iron Pyrophosphate (NFPP) battery systems. The research follows a **Clean Decomposition** strategy, separating a fixed physical plant from a variable, high-performance algorithmic control layer.

## Research Scope

### 1. Fixed Plant Model (Digital Twin)
The plant environment represents the physical hardware and electrochemical dynamics, treated as a static baseline for control development:
*   **Electrochemical Core**: Standalone 16S1P NFPP pouch-cell pack (10Ah, 48V) modeled via the Doyle-Fuller-Newman (DFN) framework.
*   **Thermal Dynamics**: Distributed core-casing thermal nodes with natural convection and Arrhenius-based aging kinetics.
*   **Power Conversion**: A full conversion and conditioning system (STS, PQC, isolated DC/DC) with SRF-PLL grid monitoring.

### 2. Variable BMS Layer (Core Contribution)
The primary research focus is the design and validation of a model-based Battery Management System:
*   **State Estimation**: Extended Kalman Filter (EKF) for SOC tracking and Recursive Least Squares (RLS) for SOH/Impedance inference.
*   **Protection Logic**: Diagnostic hooks for voltage (OV/UV), temperature (OT), and abnormal impedance rise.
*   **Safety Enforcement**: Multi-objective current arbitration with thermal and SOC-boundary derating.

### 3. Hierarchical Optimization (DSMO)
A multi-stage framework for cell design enhancement:
*   **Material Discovery**: Property acquisition for eco-friendly, **non-fluorinated salts** (NaTCP, NaBOB) and Fe-site doping (**Cr**, **Mn**) using OQMD/AFLOW APIs.
*   **Parameter Optimization**: Differentiable Sensitivity Manifold Optimization (DSMO) fine-tuning a coupled design space:
    - **Structural ($\theta_s$):** Thickness, porosity, tortuosity, loading, and particle size.
    - **Material ($\theta_m$):** NFPP/carbon fractions and electrolyte composition.
    - **Engine:** PyBaMM/CasADi symbolic sensitivities and FEniCSx mechanical adjoints.

## Repository Structure

- `src/cell_optimization/`: Material discovery engines and structural DSMO scripts.
- `src/bms_design/`: BMS ECU logic (`ecu.m`), Simscape digital twin components (`.ssc`), and validation scripts.
- `nfpp_sodium_ion/`: Registered PyBaMM parameter set for NFPP/Hard-Carbon chemistry.
- `src/report.ipynb`: Orchestration notebook for the complete research pipeline.

## Getting Started

### Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Install PyBaMM parameter package
pip install -e nfpp_sodium_ion/
```

### Execution
Run the complete research pipeline (Verification -> Optimization -> Validation -> BMS Report) via the Jupyter notebook:
```bash
jupyter notebook src/report.ipynb
```

## References

- **Paper Title**: Constrained DFN-Based NFPP Sodium-Ion Cell Optimization and Model-Based Battery Management System Design
- **Core Chemistry**: Sodium Iron Pyrophosphate (NFPP) vs. Hard Carbon
- **Modeling Framework**: PyBaMM (Electrochemical), FEniCSx (Mechanical), Simscape (Control/Plant)
