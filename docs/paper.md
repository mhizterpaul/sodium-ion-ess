# NFPP Sodium-Ion BESS Performance Benchmarking and Latent Distribution Network State Estimation Using Network Realization Signatures

## Methodology

Base Cell Model (Literature-Aligned NFPP Sodium-Ion Twin System)
1. Electrochemical Core (DFN-Compatible Reaction)
The sodium iron pyrophosphate (NFPP) cathode operates via reversible sodium intercalation:
Na₂FePO₄P₂O₇ ⇌ NaₓFePO₄P₂O₇ + (2 − x)Na⁺ + (2 − x)e⁻
Theoretical specific capacity: ~95–100 mAh g⁻¹, consistent with reported polyanionic NFPP sodium-ion cathode systems used in pouch-scale prototypes.
2. Cathode Electrode Architecture (Composite Design)
NFPP cathodes in practical sodium-ion full cells follow a carbon–binder–domain composite structure processed using N-methyl-2-pyrrolidone (NMP)-based slurry casting.
Fixed composition:
	Sodium iron pyrophosphate (NFPP) active material: 85 wt% 
	Conductive carbon additive (carbon black / acetylene black): 8 wt% 
	Binder: polyvinylidene fluoride (PVDF): 7 wt% 
This structure reflects standard aluminum current collector-based cathodes used in sodium-ion pouch cells with high-density electrode compaction.
3. Anode Design (Hard Carbon System)
Hard carbon anodes are implemented as disordered carbon networks with nanopore and turbostratic domains enabling sodium storage through adsorption, intercalation, and pore filling mechanisms.
Fixed formulation:
	Hard carbon active material: 88 wt% 
	Conductive carbon additive: 6 wt% 
	Binder: polyvinylidene fluoride (PVDF): 6 wt% 
Practical specific capacity: 250–300 mAh g⁻¹, consistent with full-cell hard carbon sodium storage behavior.
4. Electrolyte System (Carbonate-Based Sodium Salt System)
The electrolyte follows a standard sodium-ion full-cell carbonate formulation:
	Sodium hexafluorophosphate (NaPF₆): 1.0 molar concentration 
	Sodium difluoro(oxalato)borate (NaDFOB): 0.2 molar concentration 
	Solvent system: ethylene carbonate and propylene carbonate in 1:1 volumetric ratio 
	Ionic conductivity: ~10 mS cm⁻¹ at 25°C 
5. Electrolyte Additive System (Interphase Engineering)
Interfacial stability is controlled using electrolyte additives that regulate both solid electrolyte interphase and cathode electrolyte interphase formation:
	Fluoroethylene carbonate (FEC): 3 wt%
→ promotes stable solid electrolyte interphase (SEI) formation on the hard carbon anode 
	Vinylene carbonate (VC): 2 wt%
→ enhances SEI uniformity and suppresses continuous electrolyte decomposition 
	Sodium difluoro(oxalato)borate (NaDFOB): functions as both co-salt and cathode electrolyte interphase (CEI) stabilizer 
The SEI is a passivation layer formed on the anode that regulates sodium-ion transport and prevents continuous electrolyte decomposition, while the CEI stabilizes cathode surface reactions and mitigates structural degradation.
6. Pouch Cell Mechanical Architecture (Stacked Design)
The full cell follows a stacked pouch configuration consistent with sodium-ion prototype manufacturing systems:
	Form factor: stacked Z-fold pouch cell architecture 
	Nominal voltage: 3.0–3.2 volts 
	Target capacity class: 10 ampere-hour design point 
Layer stack:
	Cathode current collector: aluminum foil (~15 micrometers) 
	Anode current collector: copper foil (~10 micrometers) 
	Separator: polyolefin trilayer membrane (~20 micrometers) 
	External casing: poly-based moisture barrier (no aluminum laminate)
	Inner sealant: polypropylene-based sealing layer 

Base Model Validation and BESS Robustness Evaluation Framework
1. Electrochemical–Thermal Driver Model
The cell behavior is resolved using a Doyle-Fuller-Newman (DFN) electrochemical framework coupled with a lumped thermal model.
This captures the co-evolution of state of charge (SOC), state of health (SOH) degradation trajectory, and transient thermal fields $T(t)$ under complex, multi-stage charge-discharge cycling.
The BESS evaluation implements realistic grid outage scenarios, PV-firming profiles, and oscillating C-rate stress tests.

2. High-Fidelity Efficiency Benchmarking
The digital twin validates the performance of the BESS by programmatically evaluating three distinct physically-grounded efficiency metrics across simulated dispatches:
- **Coulombic Efficiency ($\eta_C$):** Evaluated via the integration of current flow:
  $$ \eta_C = \frac{\int_{discharge} I dt}{\int_{charge} I dt} $$
  using robust current-direction tracking based on changes in the discharge capacity.
- **Energy Efficiency ($\eta_E$):** Evaluated by integrating terminal power:
  $$ \eta_E = \frac{\int_{discharge} V \cdot I dt}{\int_{charge} V \cdot I dt} $$
- **Voltage Efficiency ($\eta_V$):** Derived directly as the ratio of energy efficiency to Coulombic efficiency:
  $$ \eta_V = \frac{\eta_E}{\eta_C} $$

3. Thermoelastic Strain-Based Structural Integrity Model
The spatial temperature and stoichiometric gradients act as drivers for mechanical expansion and contraction within the electrode-electrolyte-interphase continuum. The mechanical response is modeled as a 1D-to-3D thermoelastic continuum.
Deformation-producing strains are resolved along the electrode thickness axis to evaluate structural fatigue.
A hard constraint-based robustness score is computed by comparing the maximum spatial strain against a critical lower-bound failure envelope ($\epsilon_{crit}$), which represents the minimum strain at which irreversible physical degradation initiates. If the local induced strain exceeds this critical envelope, mechanical failure is triggered, and the cycle-life endurance metric ($n_{crit}$, $t_{crit}$) is evaluated.

### DFN-Based NFPP Cell Optimization Framework
Methodological Scope Statement
This work presents a multiphysics optimization framework for NFPP-based sodium-ion cells, where material variation is restricted to dopant-level and electrolyte (salt/solvent) chemistry, coupled with structural and thermal co-optimization.

Objective Definition
The cell design is optimized using a hierarchical Material-Structural framework. The primary objective is to discover chemistry modifications (dopants/salts/solvents) compatible with the already-validated NFPP architecture while simultaneously fine-tuning structural parameters. Cost reduction and performance gains.

    **Design Space:**
*   **Structural Parameters ($\theta_s$):** Electrode thickness ($L_c, L_a$), porosity ($\epsilon_c, \epsilon_a, \epsilon_{sep}$), tortuosity ($\tau$), active material loading and particle size ($r_p$).
*   **Material Parameters ($\theta_m$):** NFPP fraction, conductive carbon fraction, and electrolyte composition (concentration/salts)

1. Stage A-C: Layered Material Mapping & Physics Framework
This phase resolves performance properties for chemistry modifications using a decoupled architecture: a **Material Mapping Engine** for data resolution and a **Physics Layer** for property-to-parameter transformation.
*   **Decoupled Mapping Engine:** The framework implements a prioritized resolution flow (OQMD Exact $\rightarrow$ MP Exact $\rightarrow$ Class Baselines) for a fixed candidate space (Mn/Cr/Ni dopants, NaBOB/NaTCP salts, MTMS functionalization). Strict stability-sorting ensures ground-state accuracy.
*   **Physics Channel Models:** Performance deltas are derived through channel-specific physics models: Nernstian proxies for voltage shifts ($ΔV \propto -ΔE_f$), exponential thermal activation mapping for conductivity ($\sigma \propto \exp(-E_g/2kT)$), and interphase kinetic models for SEI growth, all scaled by a bounded stability realization factor.
*   **Electrolyte & Fluorine Reduction:** Selection of non-fluorinated salts to reduce environmental burden and cost. Primary candidates include **NaBOB** (Sodium bis(oxalato)borate) for stability and **NaTCP** (Sodium tricyanomethanide) for high performance.
*   **Electrode Doping:** Fe-site doping for cathodes using **Cr** (Cr³⁺ stabilizer), **Mn** (voltage booster), and **Ni** is evaluated via sensitivity-based optimization.
*   **Alkyl Silane Functionalization:** Implementation of hard carbon electrode functionalization using **methyltrimethoxysilane (MTMS)**. This process replaces surface –OH groups with –Si–O–R groups on the hard carbon electrode, increasing hydrophobicity and promoting a more uniform SEI layer. The model accounts for reduced SEI kinetics (slower growth and lower irreversible capacity fade), slower interfacial resistance growth over cycles, and optimized exchange current density resulting from improved surface wetting and local ion accessibility.

2. Stage D: Electrochemical Projection Layer
Selected material modifications are translated into DFN-compatible perturbations through a physics-based parameter update stage rather than a simple multiplicative rescaling. For each candidate material pair, the workflow computes thermodynamic, transport, kinetic, and structural deltas, and applies them to the baseline parameter set using the implementation's transform routine:
$\theta' = \mathcal{T}(\theta_{base}, \Delta \theta_{material})$
where $\mathcal{T}$ denotes the parameter-mapping step used to preserve consistency with the calibrated DFN model.

3. Stage E-F: Sensitivity-Driven Optimization & Validation
The projected design space ($\theta = [\theta_s, \theta_m]$) is explored with a hierarchical workflow that combines sensitivity screening, objective-specific SG-CEM refinement, and expensive stability filtering. In the implementation, the design vector is first perturbed around a nominal point to estimate the Jacobian of the energy, power, and stability responses; only the most influential variables for each objective are retained for optimization instead of searching the full design space at once.
*   **Phase 1: Sensitivity Screening:** For each material combination, the code computes the Jacobian of the three objective responses around the nominal design, normalizes the sensitivity magnitudes, and keeps only the variables whose normalized influence exceeds the objective-specific threshold (approximately 50% of the largest sensitivity).
*   **Phase 2: Objective-Specific Optimization:** Three separate single-objective searches are executed with a Sensitivity-Guided Cross-Entropy Method (SG-CEM) on the reduced active-variable sets. Each objective is evaluated from DFN-based metrics and scaled using a baseline reference value; infeasible configurations are penalized, including thermal-limit violations and the electrode-thickness ordering constraint.
*   **Phase 3: Candidate Filtering:** The optimized candidates are re-evaluated with the expensive thermo-mechanical stability solver. Only designs that pass the PDE-based strain check remain eligible for ranking.
*   **Phase 4: Final Design Composition:** Among the retained candidates, the point with the highest stability score is selected as the dominant design, and the final representative point is formed by blending this best candidate with the mean of the valid set using an 80/20 weighting to improve robustness.
*   **Phase 5: Validation:** The selected configuration is re-simulated with the DFN workflow, and the final outputs include the chosen material pair, representative design vector, sensitivity matrix, and performance metrics used for downstream assessment. While this work focuses on a foundational design space, the cell architecture remains amenable to further performance enhancement via composite electrode structuring, advanced pore network engineering, perturbing other dopant sites (beyond the Fe-site), and exploring a broader range of electrolyte systems (solvents and additives) to further enhance cycle life and energy density. The current optimization scope is intentionally streamlined to accommodate the computational constraints of the DFN solver while effectively demonstrating the viability of physics-based optimization for enhancing the cost-efficiency and performance of sodium-ion energy storage systems.
* **Computed cell-level performance metrics include:**  Energy capacity (kWh), Nominal voltage (V), Continuous current (A), Peak current (A), Charge time (h or min under rated C-rate), Power capability (kW or C-rate equivalent), Cycle life (cycles to end-of-life under defined SOH threshold) 

---

# NFPP Sodium-Ion BESS Performance Benchmarking and Latent Distribution Network State Estimation Using Network Realization Signatures

# 1. Research Objective

The objective of this work is to determine whether the internal operating state and structural characteristics of an unknown downstream distribution network can be inferred solely from synchronized electrical measurements acquired at the known distribution station boundary.

Unlike conventional Distribution System State Estimation (DSSE), where the complete network topology and bus model are assumed known, this research considers a partially observable network in which only the upstream distribution station is known while the downstream network remains hidden.

The realization problem is formulated as

[
X_R=\Phi(M)
]

where

* (M) denotes synchronized measurements acquired at feeders and distribution transformers,
* (X_R) is a latent realization state describing the hidden network,
* (\Phi(\cdot)) is an unknown realization operator learned empirically from simulated operating scenarios.

The emphasis is therefore on discovering which hidden network properties are electrically observable at the distribution station interface and how these observables evolve under changing operating conditions.

---

# 2. System Model

## Known Plant for Latent Network Realization

The upstream distribution station is completely known and serves as the boundary for observing downstream states.

It consists of:

```text
        Utility Source (Swing Bus)

                  │

      Distribution Substation Transformer

                  │

        Main Distribution Bus (Point of Common Coupling) ── Power Conditioning Unit (PCU) ── Generator

                  │
      ┌───────────┼───────────┐
      │           │           │

   Feeder 1    Feeder 2    Feeder 3
      │           │           │

Distribution  Distribution  Distribution
Transformer   Transformer   Transformer
      │           │           │

 Unknown LV   Unknown LV   Unknown LV
 Distribution Distribution Distribution
  Networks     Networks     Networks
```

The plant model contains strictly distribution network elements and local sources to facilitate Latent Network Realization:
* **Utility Source (Swing Bus)**: Represents the steady connection to the transmission grid.
* **Distribution Substation Transformer**: Substation transformer supplying the medium-voltage bus.
* **Main Distribution Bus (Point of Common Coupling)**: Serves as the central bus where the generators and feeders connect.
* **Generator**: A shared local generator providing primary active generation capacity, representing local active energy generation to simplify the microgrid known plant.
* **Power Conditioning Unit (PCU)**: A single unit interfacing the generator, modeled without internal step-up transformers or switchgear to simplify the modeling of the boundary assets.
* **Switchgear**: Medium-voltage switchgear and protection components modeled as a separate block in the plant.
* **Three Outgoing Feeders**: Radial lines extending from the substation, each characterized by known feeder lengths and impedances.
* **Fixed Set of Transformers**: Step-down distribution transformers whose primary-side terminals serve as the boundary measurement interfaces.
* **Measurement and Monitoring Devices**: Electrical sensors capturing voltage, current, active/reactive power, and sequence components at each feeder head and transformer primary terminal.

No BESS benchmarking or BESS-specific multi-physics digital twins are modeled within this system layer; BESS evaluations are kept separate at the electrochemical layer.

---

# 3. Assumptions

## Known

✓ Distribution station configuration

✓ Swing bus voltage

✓ Source/generator model

✓ Distribution transformer parameters

✓ Feeder impedances

✓ Feeder lengths

✓ Transformer locations

✓ Nominal voltage levels

✓ Load model categories

---

## Unknown

✗ Downstream buses

✗ Downstream topology

✗ Feeder branching structure

✗ Switch status

✗ Customer connectivity

✗ Actual customer loading

✗ Time-varying load dynamics

---

# 4. Simulation Platform

## Primary Simulator

OpenDSS is used to model the distribution station and downstream distribution network.

It provides

* Three-phase power flow
* Quasi-static time-series simulation
* Distribution feeder modelling
* Distribution transformer modelling
* Voltage regulator operation
* Capacitor bank switching
* Load switching
* Protection device modelling
* Python integration for automated simulation studies

---

# 5. Hybrid Simulation Framework

The proposed simulation framework combines quasi-static and transient analyses.

```text
              OpenDSS

      Distribution Station Model

                │

      Operating Point Solution

                │

      Operating Condition Generator

                │

    Optional Electromagnetic Simulator

                │

      Voltage & Current Waveforms

                │

      Signal Processing Pipeline

                │

        Feature Extraction

                │

Distributed Dynamic State Estimation
```

The transient simulator, where used, reproduces waveform responses associated with

* Transformer energization
* Capacitor switching
* Motor starting
* Feeder switching
* Temporary faults

These simulations complement the steady-state information obtained from OpenDSS.

---

# 6. Measurement Architecture

Measurements are obtained from two sensing layers: feeder monitoring and transformer edge monitoring.

## A. Feeder Measurements

Each outgoing feeder is instrumented to acquire

### Electrical Quantities

* Three-phase voltage magnitude and phase angle
* Three-phase current magnitude and phase angle
* Active power ((P))
* Reactive power ((Q))
* Apparent power ((S))
* Power factor

### Network Quality Metrics

* Frequency
* Rate of Change of Frequency (ROCOF)
* Voltage unbalance
* Current unbalance
* Positive-, negative-, and zero-sequence components

### Dynamic Measurements

Where transient simulation is available

* Harmonic distortion (THD)
* Voltage waveform samples
* Current waveform samples
* Switching event timestamps

---

## B. Transformer Measurements

Each distribution transformer serves as an edge measurement node representing the interface to an unknown downstream network.

Measurements include

### Primary Electrical Measurements

* High-voltage terminal voltage magnitude and phase angle
* High-voltage terminal current magnitude and phase angle
* Active power
* Reactive power
* Apparent power
* Power factor

### Transformer Operating State

* Transformer loading

[
Loading=\frac{S}{S_{rated}}
]

* Voltage regulation
* Tap position (if applicable)
* Estimated secondary demand
* Copper losses
* Core losses
* Sequence components
* Estimated transformer impedance

[
Z=\frac{V}{I}
]

### Dynamic Quantities

Where supported

* Loading rate
* Overload duration
* Load recovery characteristics
* Transformer temperature (optional)
* Transient voltage and current waveforms

---

# 7. Operating Scenario Generation

Rather than attempting to recover a predefined downstream network, the simulation framework systematically perturbs the unknown downstream network while maintaining a fixed upstream distribution station.

The perturbation process modifies hidden network characteristics including

* Number of downstream buses
* Network connectivity
* Distribution line parameters
* Load allocation
* Load composition
* Load switching sequences
* Motor penetration
* Capacitor placement
* Transformer loading
* Distributed energy resource penetration (optional)

Each perturbed network is simulated under a range of operating conditions to generate synchronized feeder and transformer measurements.

The objective is to determine how variations in hidden network structure and operating state manifest in the measurable electrical response at the known distribution station boundary.

This produces a comprehensive dataset relating hidden network perturbations to observable boundary measurements for subsequent realization and distributed dynamic state estimation.

---

# 8. Feature Extraction

Rather than using raw measurements directly, the synchronized measurements are transformed into physics-informed features that are expected to generalize across operating conditions.

## Steady-State Features

* Voltage magnitude
* Voltage phase angle
* Current magnitude
* Current phase angle
* Active and reactive power flow
* Apparent power
* Power factor
* Feeder losses
* Transformer loading
* Voltage regulation
* Sequence components
* Equivalent impedance estimates
* Electrical distance indicators
* Network stiffness indices

## Dynamic Features

Where transient simulations are available

* Fast Fourier Transform (FFT) spectra
* Wavelet coefficients
* Spectral centroid
* Dominant oscillation frequencies
* Oscillation damping ratios
* ROCOF
* Voltage recovery time
* Current recovery time
* Phase-angle evolution
* Switching signatures
* Motor-start signatures
* Capacitor-switching signatures
* Transformer energization signatures
* Cross-correlation between feeder measurements
* Mutual information between sensing locations

These features collectively describe both the steady-state and dynamic behaviour of the hidden downstream network.

---

# 9. Distributed Dynamic State Estimation

The realization state is not prescribed a priori but is inferred from synchronized feeder and transformer measurements.

The estimation problem is expressed as

[
X_R=\Phi(M)
]

where the realization operator (\Phi(\cdot)) is learned from the simulated operating scenarios.

The estimated realization state may include

* Effective electrical distance to active loads
* Aggregate network impedance
* Phase-coupling indices
* Voltage sensitivity indices
* Reactive power support indices
* Network stiffness
* Transformer loading state
* Feeder coherency
* Dominant power-flow modes
* Synchronization indices
* Dynamic spectral modes

These latent coordinates evolve continuously as the hidden downstream network changes and collectively characterize its instantaneous operating condition.

---

# 10. Validation Strategy

The methodology is validated by evaluating the realization algorithm over a large ensemble of simulated downstream network realizations generated through systematic perturbation of hidden network parameters.

Validation focuses on answering the following research questions.

1. **Hidden Network Observability**

   Which structural and operational characteristics of the hidden downstream network are observable from synchronized feeder and transformer measurements?

2. **Network Complexity**

   As the hidden network size increases (e.g., increasing numbers of downstream buses), how does the observability and estimation accuracy of the realization algorithm change?

3. **Measurement Sufficiency**

   What combination of feeder and transformer measurements provides sufficient information for accurate distributed dynamic state estimation?

4. **Sensitivity to Hidden Network Perturbations**

   Which classes of downstream perturbations—including topology changes, load redistribution, switching events, transformer loading, and line parameter variations—produce measurable changes at the distribution station boundary?

Performance is assessed using metrics including latent state estimation error, observability of hidden parameters, sensitivity to measurement noise, robustness across operating conditions, computational efficiency, and scalability with increasing downstream network complexity.

The validation establishes the practical limits of boundary-based realization and identifies the sensing architecture required for distributed dynamic state estimation in partially observable distribution networks.
