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

#### 3.1 Fundamental Energy Decomposition (Core Object)
You are controlling the partition:
$P_{solar}(t) = P_{load}(t) + P_{bat}(t) + P_{reactive}(t) + P_{harmonic}(t) + P_{dump}(t) + P_{loss}(t)$

Where each term is a different energy channel, not a scheduling variable.

**Interpretation of each channel:**
- **$P_{load}$ (Useful real power delivery)**: Energy consumed by system load. Primary objective: maximize.
- **$P_{bat}$ (Electrochemical buffering energy)**: Not “storage scheduling”. It is a state transition constraint actuator limited by: SOC, SOH, and thermal state.
- **$P_{reactive}$ (Grid-forming stability energy)**: Electromagnetic field support for voltage stability. Used to stabilize voltage collapse, damp oscillations, and regulate transient response. Constraint: $Q(t) \neq 0 \Rightarrow$ voltage stability improvement.
- **$P_{harmonic}$ (Unwanted spectral energy)**: Minimized. Represents inverter switching distortion, nonlinear load coupling, and resonance effects. Treated as a penalty state.
- **$P_{dump}$ (Safety dissipation sink)**: Used when battery/load are saturated or reactive control is insufficient. Examples: resistive dump loads, thermal diversion. A controlled failure absorption channel.
- **$P_{loss}$ (Unavoidable physical inefficiency)**: Conduction losses, switching losses, thermal dissipation. Not controllable.

#### 3.2 Optimization Objectives (Core Contribution)
The system optimizes a flow partition policy $\pi: P_{solar}(t) \rightarrow \{P_{load}, P_{bat}, P_{reactive}, P_{dump}\}$ to maximize:
1. **Useful energy delivery**: $\max \mathbb{E}[P_{load}(t)]$
2. **System availability**: $\mathbb{P}(\text{instability}) \le \epsilon$ (no collapse constraint)
3. **Operational life maximization**: $\min \Delta SOH(t) + \Delta R_{inverter}(t)$
4. **Energy utilization efficiency**: $\eta = \frac{\int P_{load}(t) dt}{\int P_{solar}(t) dt}$

#### 3.3 Minimum System Load
This is the minimum real-power absorption required to keep the system in a stable operating manifold:
$P_{min}(t) = P_{load}^{required} + P_{stability\_reserve}$
Stability reserve includes: reactive compensation margin, battery headroom, and transient absorption capacity.
Physics requires continuous dissipation pathways to avoid instability if $P_{solar}(t) > P_{min}(t)$.

#### 3.4 System Stability
Stability includes:
1. **Energy stability**: $P_{in} \approx P_{out}$
2. **Electrical stability**: Voltage regulation and frequency damping.
3. **Dynamic stability**: Transient response timing and ramp rate control.
4. **Spectral stability**: Harmonic suppression and switching noise containment.
