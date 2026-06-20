# DFN-Based Optimization of NFPP Sodium-Ion Cells for Integrated Plant-Level Control in Solar–BESS Microgrids

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhizterpaul/sodium-ion-ess/blob/main/src/report.ipynb)

This repository implements a high-fidelity digital twin and optimization framework for Sodium Iron Pyrophosphate (NFPP) battery systems integrated with utility-scale solar-BESS microgrid control.

## Research Scope

### 1. Fixed Power Plant Model (Digital Twin)
The plant environment represents the physical microgrid hardware and electrochemical dynamics:
*   **Microgrid Assets**:
    *   **Solar PV**: 100kWp mono-crystalline silicon array.
    *   **Primary Generation Array**: 50kW dispatchable power asset.
    *   **BESS**: 100kWh / 50kW AC-coupled sodium-ion storage system (208 modules).
*   **Electrochemical Core**: 16S1P NFPP pouch-cell pack modules modeled via the Doyle-Fuller-Newman (DFN) framework.
*   **Thermal Dynamics**: Distributed core-casing thermal nodes with natural convection and aging kinetics.
*   **Power Conditioning**: Utility-scale PCUs with step-up transformers and MV switchgear for grid interconnection.

### 2. Plant Fault Detection, Network State Estimation & Sustainability (Core Contribution)
The primary research focus is an analytical framework that performs real-time network state estimation (distribution lines and load network) and fault detection to ensure plant integrity.

**Monitoring Objective**: Ensure system visibility and integrity under noise and faults while identifying opportunities for real power improvement.
**Optimal Sustainability**: Determine the system parameters where plant operation becomes sustainable through MST derivation.

#### Fundamental Energy Decomposition
Controlling the partition:
$P_{solar}(t) + P_{array}(t) = P_{load}(t) + P_{bat}(t) + P_{reactive}(t) + P_{harmonic}(t) + P_{dump}(t) + P_{loss}(t)$

*   **$P_{load}$ (Useful Real Power)**: Maximize energy consumed by the system load.
*   **$P_{bat}$ (Electrochemical Buffering)**: State transition constraint actuator limited by SOC, SOH, and thermal states.
*   **$P_{reactive}$ (Grid-Forming Stability)**: Electromagnetic field support for voltage stability ($Q(t) \neq 0$).
*   **$P_{harmonic}$ (Unwanted Spectral Energy)**: Penalty state representing inverter switching distortion and nonlinear coupling.
*   **$P_{dump}$ (Safety Dissipation)**: Controlled failure absorption channel (resistive dump loads) when sinks are saturated.
*   **$P_{loss}$ (Physical Inefficiency)**: Unavoidable conduction and switching losses.

#### Optimization Objectives
The primary goal is to **Maximize Plant Utilization**:
$U(t) = P_{load}(t) + P_{battery\_use}(t)$

*   **Sustainability Constraint (MST)**: $U(t) \ge MST(t) = \frac{C_{opex}(t)}{p(t)}$
*   **System Availability**: $\mathbb{P}(\text{instability}) \le \epsilon$
*   **Degradation Control**: $\min \Delta SOH(t) + \Delta R_{PCU}(t)$
*   **Energy Utilization Efficiency**: $\eta = \frac{\int P_{load}(t) dt}{\int P_{solar}(t) dt}$

### 3. Hierarchical Optimization
A multi-stage framework for cell design enhancement:
*   **Layered Material Mapping**: Decoupled architecture for eco-friendly salts (NaTCP, NaBOB), cathode dopants (Cr, Mn, Ni), and MTMS functionalization.
*   **Parameter Optimization**: Hierarchical search for structural ($\theta_s$) and material ($\theta_m$) parameters using sensitivity-based Jacobian screening and Genetic Algorithms.

## Repository Structure

- `src/cell_optimization/`: Material discovery engines and structural optimization scripts.
- `src/power_plant/`: Utility-scale power plant control logic, digital twin components, and energy dispatch validation.
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

- **Paper Title**: DFN-Based Optimization of NFPP Sodium-Ion Cells for Integrated Plant-Level Control in Solar–BESS Microgrids
- **Core Chemistry**: Sodium Iron Pyrophosphate (NFPP) vs. Hard Carbon
- **Modeling Framework**: PyBaMM (Electrochemical), FEniCSx (Mechanical), Simulink (Control)
