# DFN-Based Co-Optimization of NFPP Sodium-Ion Cells and Model-Informed Energy Dispatch in Hybrid Solar–Battery Energy Storage Systems

## Methodology

### 1. Base Cell Model (Literature-Aligned NFPP Sodium-Ion Twin System)
The electrochemical behavior is resolved using a Doyle-Fuller-Newman (DFN) framework implemented in PyBaMM. This captures the coupled evolution of State of Charge (SOC), State of Health (SOH), and heat generation.

### 2. DFN-Based NFPP Cell Optimization Framework
A hierarchical Material-Structural framework optimizes the NFPP-based sodium-ion cells.
- **Design Space**: Structural parameters (thickness, porosity, particle size) and material parameters (dopants, electrolyte composition).
- **Objectives**: Energy capacity, power capability, and thermo-mechanical stability.

### 3. Model-Informed Energy Dispatch (Power Plant Framework)
The system is modeled as a real-time partitioning of stochastic solar power into physically constrained sinks.

#### 3.1 Fundamental Energy Decomposition
We control the partition:
$P_{solar}(t) = P_{load}(t) + P_{bat}(t) + P_{reactive}(t) + P_{harmonic}(t) + P_{dump}(t) + P_{loss}(t)$

Where each term represents a distinct energy channel:
1. **$P_{load}$ (Useful real power delivery)**: Energy consumed by system load. Primary objective is to maximize this.
2. **$P_{bat}$ (Electrochemical buffering energy)**: State transition constraint actuator (limited by SOC, SOH, and thermal state).
3. **$P_{reactive}$ (Grid-forming stability energy)**: Electromagnetic field support for voltage stability. $Q(t) \neq 0 \Rightarrow$ voltage stability improvement.
4. **$P_{harmonic}$ (Unwanted spectral energy)**: Inverter switching distortion and nonlinear load coupling. Minimized as a penalty state.
5. **$P_{dump}$ (Safety dissipation sink)**: Controlled failure absorption channel (resistive dump loads) used when battery/load are saturated.
6. **$P_{loss}$ (Unavoidable physical inefficiency)**: Conduction and switching losses (uncontrollable).

#### 3.2 Optimization Objectives
The flow partition policy $\pi: P_{solar}(t) \rightarrow \{P_{load}, P_{bat}, P_{reactive}, P_{dump}\}$ is optimized to:
1. **Maximize Useful Energy Delivery**: $\max \mathbb{E}[P_{load}(t)]$
2. **Ensure System Availability**: $\mathbb{P}(\text{instability}) \le \epsilon$
3. **Maximize Operational Life**: $\min \Delta SOH(t) + \Delta R_{inverter}(t)$
4. **Energy Utilization Efficiency**: $\eta = \frac{\int P_{load}(t) dt}{\int P_{solar}(t) dt}$

#### 3.3 System Stability
Stability is maintained across four dimensions:
1. **Energy Stability**: $P_{in} \approx P_{out}$
2. **Electrical Stability**: Voltage regulation and frequency damping.
3. **Dynamic Stability**: Transient response timing and ramp rate control.
4. **Spectral Stability**: Harmonic suppression and switching noise containment.

#### 3.4 Minimum System Load
$P_{min}(t) = P_{load}^{required} + P_{stability\_reserve}$
Physics requires continuous dissipation pathways. If $P_{solar} > P_{min}$, the system activates battery absorption, dump sinks, or controlled load increases to avoid instability.
