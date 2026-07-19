
This repository implements a high-fidelity digital twin and optimization framework for Sodium Iron Pyrophosphate (NFPP) battery systems within an integrated plant–network digital twin framework for solar–BESS microgrids.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhizterpaul/sodium-ion-ess/blob/main/src/report.ipynb)

## Research Scope

### 1. DFN-Based NFPP Cell Optimization
A hierarchical multi-stage framework for cell design enhancement:
*   **Layered Material Mapping**: Decoupled architecture for eco-friendly salts (NaTCP, NaBOB), cathode dopants (Cr, Mn, Ni), and MTMS functionalization.
*   **Parameter Optimization**: Hierarchical search for structural ($\theta_s$) and material ($\theta_m$) parameters using sensitivity-based Jacobian screening and a Sensitivity-Guided Cross-Entropy Method (SG-CEM).

### 2. Multi-feeder solar–BESS network state realization and anomaly detection using phase dynamics (Core Contribution)
The primary research focus is the realization of network states and anomaly detection in a multi-feeder microgrid coupled by shared solar and BESS sources.

#### Phase-Based Diagnostics
*   **Shared Source Coupling**: Modeling $P_{source} = P_{solar} + P_{BESS} = \sum P_{F_i} + P_{loss}$.
*   **Network Realization State**: Tracking $X_R = [\Delta \theta_{F1}, \dots, \Delta \theta_{Fn}]$ for phase-based anomaly detection.
*   **Propagation Analysis**: Analyzing how disturbances in one feeder propagate through the shared source to affect the wider network.
*   **Anomaly Localization**: Identifying feeder-level faults when $\Delta \theta_{Fi}$ deviates from the expected stability envelope.

### 3. Physical Power Plant Model (Digital Twin)
The plant environment represents the physical microgrid hardware:
*   **Microgrid Assets**: 100kWp Solar PV, 50kW Primary Generation, and 100kWh BESS (208 modules).
*   **Multi-Feeder Topology**: Feeders coupled to a shared solar-BESS source via utility-scale power conditioning.
*   **Architecture**: Multi-string Central Inverter → LV/MV Step-up Transformer → MV Switchgear → Utility Grid.

## Repository Structure

- `src/cell_optimization/`: Material discovery engines and structural optimization scripts.
- `src/power_plant/`: Utility-scale power plant control logic, digital twin components, and energy dispatch validation.
- `src/simulation/`: Multi-feeder network simulator, cell simulation utilities and phase dynamics analysis.
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
Run the complete research pipeline via the Jupyter notebook:
```bash
jupyter notebook src/report.ipynb
```

## References

- **Paper Title**: DFN-Based Optimization of NFPP Sodium-Ion Cells within an Integrated Plant–Network Digital Twin Framework for Solar–BESS Microgrids
- **Core Chemistry**: Sodium Iron Pyrophosphate (NFPP) vs. Hard Carbon
- **Modeling Framework**: PyBaMM (Electrochemical), FEniCSx (Mechanical), Simscape, Matlab (Power Systems)
