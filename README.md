# NFPP Sodium-Ion BESS Performance Benchmarking and Latent Distribution Network State Estimation Using Network Realization Signatures

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhizterpaul/sodium-ion-ess/blob/main/src/report.ipynb)

## Research Summary & Scope

DFN-Based NFPP Cell Optimization
A hierarchical multi-stage framework for cell design enhancement:
*   **Layered Material Mapping**: Decoupled architecture for eco-friendly salts (NaTCP, NaBOB), cathode dopants (Cr, Mn, Ni), and MTMS functionalization.
*   **Parameter Optimization**: Hierarchical search for structural ($\theta_s$) and material ($\theta_m$) parameters using sensitivity-based Jacobian screening and a Sensitivity-Guided Cross-Entropy Method (SG-CEM).
This repository implements an integrated plant–network digital twin framework for high-fidelity performance benchmarking of Sodium Iron Pyrophosphate (NFPP) battery energy storage systems (BESS) and latent distribution network state estimation.

### 2. Latent Distribution Network State Estimation & Feature Extraction (Core Contribution)
The primary research focus is the realization of latent network states in a multi-feeder distribution network using boundary measurements and sub-cycle transient realization signatures:
*   **Fixed Upstream Plant**: OpenDSS model incorporating utility swing bus, substation step-down transformer, main bus, generator, power conditioning unit (PCU), and 3 feeders with a fixed set of distribution transformers acting as measurement boundaries.
*   **Scenario Generator**: Systematic perturbation of the unknown downstream networks connected to the feeders, featuring linear and non-linear loads, varying live loads, changing line lengths (electrical distance), switching events, and topology reconfigurations (radial vs ring/loop).
*   **ATP-EMTP Transient Coupling**: Real-time coupling of sub-cycle transients (such as transformer inrush, capacitor switching, motor starting, temporary faults, and non-linear switching  to extract high-frequency spectral and waveform features.
*   **Stride-Slicing Boundary Measurement**: Programmatic voltage magnitude and phase angle extraction from OpenDSS `Bus.VMagAngle()` using correct stride slicing: `[0:6:2]` for magnitudes and `[1:6:2]` for phase angles.
*   **Feature Tabulation & Rendering**: Export of all steady-state and dynamic parameters directly to CSV, rendering clean HTML tabulations of transformer transient parameters and feeder parameters in `report.ipynb`.

## Repository Structure

- `src/cell_optimization/`: Material discovery engines, chemical regularization, and structural optimization scripts.
- `src/power_plant/`: OpenDSS fixed plant model, stride-slicing measurement extraction, and ATP-EMTP dynamic transient emulator.
- `src/simulation/`: Scenario generator, perturbed downstream line, load, and switching event modeling.
- `nfpp_sodium_ion/`: Ready-to-be-published PyBaMM parameter set for NFPP/Hard-Carbon chemistry.
- `src/report.ipynb`: Orchestration notebook for the complete research pipeline.

## Getting Started

### Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Install PyBaMM parameter package in editable mode
pip install -e nfpp_sodium_ion/
```

### Execution
Run the complete research pipeline via the Jupyter notebook:
```bash
jupyter notebook src/report.ipynb
```

## References

- **Paper Title**: NFPP Sodium-Ion BESS Performance Benchmarking and Latent Distribution Network State Estimation Using Network Realization Signatures
- **Core Chemistry**: Sodium Iron Pyrophosphate (NFPP) vs. Hard Carbon
- **Modeling Framework**: PyBaMM (Electrochemical), FEniCSx (Mechanical), OpenDSS (Distribution Power Flow)
