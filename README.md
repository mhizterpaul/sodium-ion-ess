
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhizterpaul/sodium-ion-ess/blob/main/src/report.ipynb)

This repository implements a high-fidelity digital twin and co-optimization framework for Sodium Iron Pyrophosphate (NFPP) battery systems and model-informed energy dispatch policies in hybrid solar–BESS systems.

## Research Scope

### 1. Model-Informed Energy Dispatch (Core Contribution)
The primary research focus is the real-time partitioning of stochastic solar power into physically constrained sinks while maintaining a stability manifold and minimizing lifetime degradation.

#### Fundamental Energy Decomposition
The system controls the partition:
$P_{solar}(t) = P_{load}(t) + P_{bat}(t) + P_{reactive}(t) + P_{harmonic}(t) + P_{dump}(t) + P_{loss}(t)$

*   **$P_{load}$ (Useful Real Power)**: Energy consumed by system load (Primary objective: maximize).
*   **$P_{bat}$ (Electrochemical Buffering)**: State transition constraint actuator (limited by SOC, SOH, thermal state).
*   **$P_{reactive}$ (Grid-Forming Stability)**: Electromagnetic field support for voltage stability ($Q(t) \neq 0$).
*   **$P_{harmonic}$ (Unwanted Spectral Energy)**: Minimized penalty state representing switching distortion and nonlinear coupling.
*   **$P_{dump}$ (Safety Dissipation Sink)**: Controlled failure absorption channel (resistive dump loads) for saturation events.
*   **$P_{loss}$ (Physical Inefficiency)**: Unavoidable conduction and switching losses.

#### Optimization Objectives
The flow partition policy $\pi$ is optimized to:
*   **Maximize Useful Energy Delivery**: $\max \mathbb{E}[P_{load}(t)]$
*   **System Availability**: $\mathbb{P}(\text{instability}) \le \epsilon$
*   **Operational Life Maximization**: $\min \Delta SOH(t) + \Delta R_{inverter}(t)$
*   **Energy Utilization Efficiency**: $\eta = \frac{\int P_{load}(t) dt}{\int P_{solar}(t) dt}$

### 2. DFN-Based NFPP Cell Optimization
A hierarchical multi-stage framework for cell design enhancement:
*   **Layered Material Mapping**: Decoupled architecture for eco-friendly salts (NaTCP, NaBOB), cathode dopants (Cr, Mn, Ni), and MTMS functionalization.
*   **Parameter Optimization**: Hierarchical search for structural ($\theta_s$) and material ($\theta_m$) parameters using sensitivity-based Jacobian screening and Genetic Algorithms.

### 3. Physical Power Plant Model (Digital Twin)
The plant environment represents the physical microgrid hardware:
*   **Microgrid Assets**: 100kWp Solar PV, 50kW Primary Generation, and 100kWh BESS (208 modules).
*   **Infrastructure**: Utility-scale power conditioning (150kVA PCU, Step-up transformer, MV Switchgear).
*   **Service Main Interface**: Balanced 3-phase interface for real-time energy flow partitioning.

## Repository Structure

- `src/cell_optimization/`: Material discovery engines and structural optimization scripts.
- `src/power_plant/`: Utility-scale power plant components and energy dispatch logic.
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
