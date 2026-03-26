## **Recommendation Model (Decision Engine) **

---

## **1. Domain Summary – Purpose & Scope**

A **multi-objective decision engine** that ranks and selects optimal sodium-ion candidates from the reduced set (~10³), under:

* Electrochemical performance
* Thermal stress (Nigeria-calibrated)
* Grid disturbance exposure
* Economic constraints (FX-adjusted)

The engine outputs a **Pareto-optimal recommendation set (K = 3–5)** with:

* Performance estimates
* Risk/uncertainty bounds
* Sensitivity gradients

---

## **2. Core Entities / Data Model**

```yaml
RecommendationInput:
  material_id: string
  E_density: float
  eta: float
  L_cycle: float
  S_T: float
  S_grid_avg: float
  C_adj: float
  stability: float

  # Environment context
  R_th: float
  C_th: float
  T_cool: float
  w_v: float
  w_f: float
  T_amb_profile: timeseries

DerivedDecisionFeatures:
  material_id: string
  L_adj: float
  P_thermal: float

ModelPrediction:
  material_id: string
  E_hat: float
  eta_hat: float
  L_hat: float
  D_hat: float

UncertaintyEstimate:
  material_id: string
  sigma_E: float
  sigma_eta: float
  sigma_L: float
  sigma_D: float

ObjectiveScore:
  material_id: string
  F: float
  components:
    E: float
    eta: float
    L_adj: float
    D_grid: float
    C_adj: float
    P_thermal: float

ParetoCandidate:
  material_id: string
  objectives: vector
  dominance_rank: int

Recommendation:
  material_id: string
  performance: ModelPrediction
  uncertainty: UncertaintyEstimate
  risk:
    P_thermal: float
  sensitivity: map<string, float>
```

---

## **3. ViewModel Interfaces**

```java
interface RecommendationViewModel {

    void loadCandidates(List<RecommendationInput> inputs);

    // Feature augmentation
    DerivedDecisionFeatures computeDerivedFeatures(RecommendationInput input);

    // Model inference
    ModelPrediction predict(RecommendationInput input);
    UncertaintyEstimate estimateUncertainty(RecommendationInput input);

    // Objective evaluation
    ObjectiveScore evaluateObjective(RecommendationInput input,
                                     ModelPrediction pred,
                                     DerivedDecisionFeatures d);

    // Optimization stages
    List<ParetoCandidate> runNSGA2();
    List<ParetoCandidate> runBayesianRefinement(List<ParetoCandidate> pareto);

    // Final selection
    List<Recommendation> selectTopK(int k);

    // Execution
    void run();

    // State access
    List<Recommendation> getRecommendations();
}
```

---

## **4. Service Workflows**

---

### **4.1 Feature Augmentation**

#### Climate-adjusted lifetime:

[
L_{adj} = L_{cycle} \cdot e^{-k_T}
]

Where:
[
k_T = \lambda \cdot \max(0, T_{amb}^{mean} - T_{ref})
]

---

### **4.2 Thermal Risk Modeling**

Estimate:
[
P_{thermal} = \Pr(T > T_{crit})
]

Using:

* Gaussian approximation:
  [
  T \sim \mathcal{N}(\mu_T, \sigma_T^2)
  ]

[
P_{thermal} = 1 - \Phi\left(\frac{T_{crit} - \mu_T}{\sigma_T}\right)
]

---

### **4.3 Ensemble Surrogate Model**

#### Model Stack

**1. Gradient Boosting**

* LightGBM / XGBoost
* Handles tabular nonlinearity well

**2. Neural Network**

* Captures cross-feature coupling:
  [
  f(x) = W_2 \sigma(W_1 x + b_1)
  ]

#### Final Prediction:

[
\hat{y} = \alpha f_{GB}(x) + (1-\alpha) f_{NN}(x)
]

---

### **4.4 Uncertainty Quantification**

#### Method A: Quantile Regression

* Predict:

  * (Q_{0.1}, Q_{0.5}, Q_{0.9})

[
\sigma \approx \frac{Q_{0.9} - Q_{0.1}}{2.56}
]

#### Method B: MC Dropout (NN)

[
\sigma^2 = \frac{1}{N} \sum (y_i - \bar{y})^2
]

---

### **4.5 Objective Function**

[
F(c) =
\alpha E +
\beta \eta +
\gamma L_{adj}
--------------

## \delta D_{grid}

## \epsilon C_{adj}

\zeta P_{thermal}
]

**Vector form (multi-objective):**
[
\mathbf{f}(c) =
(E,\ \eta,\ L_{adj},\ -D_{grid},\ -C_{adj},\ -P_{thermal})
]

---

### **4.6 NSGA-II (Global Exploration)**

#### Steps:

1. Initialize population (random subset)

2. Evaluate objectives

3. Non-dominated sorting:

   * Rank fronts (F_1, F_2, ...)

4. Crowding distance:
   [
   d_i = \sum_j \frac{f_j^{i+1} - f_j^{i-1}}{f_j^{max} - f_j^{min}}
   ]

5. Selection + crossover + mutation

**Complexity:**
[
O(MN^2)
]

Where:

* (M): objectives
* (N): population size (~200–500)

---

### **4.7 Bayesian Optimization (Local Exploitation)**

Refines top Pareto region.

#### Surrogate:

* Gaussian Process:
  [
  f(x) \sim \mathcal{GP}(\mu(x), k(x,x'))
  ]

#### Acquisition:

Expected Improvement:
[
EI(x) = \mathbb{E}[\max(0, f(x) - f^+)]
]

---

### **4.8 Combined Strategy**

```text
NSGA-II → Pareto Front → Bayesian Refinement → Final Ranking
```

---

### **4.9 Sensitivity Analysis**

Gradient:

[
\frac{\partial F}{\partial x_i}
]

Computed via:

* Finite differences:
  [
  \frac{F(x+\epsilon e_i) - F(x)}{\epsilon}
  ]

or

* Auto-diff (if NN used)

---

## **5. ViewModel Execution Flow**

```text
run()
  ├── computeDerivedFeatures()
  ├── predict()
  ├── estimateUncertainty()
  ├── evaluateObjective()
  ├── runNSGA2()
  ├── runBayesianRefinement()
  ├── selectTopK(K=3–5)
  └── return recommendations
```

---

## **6. NFR – Non-Functional Requirements**

---

### **Performance**

* NSGA-II:
  [
  O(MN^2) \approx 10^6 \text{ ops manageable}
  ]

* Bayesian:
  [
  O(n^3) \text{ (limit to small subset)}
  ]

---

### **Scalability**

* Parallel evaluation of candidates
* GPU acceleration for NN

---

### **Numerical Stability**

* Normalize inputs:
  [
  x' = \frac{x - \mu}{\sigma}
  ]

* Clip probabilities:
  [
  P_{thermal} \in [10^{-6}, 1-10^{-6}]
  ]

---

### **Critical Failure Points (Handled)**

#### 1. Feature Leakage

* Strict separation:

  * Training data ≠ simulation outputs

---

#### 2. Over-Pruning

* Enforced diversity:

  * Input must come from DPP-reduced set

---

#### 3. Thermal Misparameterization

* Sensitivity audit:
  [
  \frac{\partial F}{\partial R_{th}},\quad \frac{\partial F}{\partial C_{th}}
  ]

---

#### 4. Economic Distortion (Nigeria Context)

[
C_{real} = C \cdot (1 + fx_{risk} + import_{factor})
]

Stress test:

* ±30% FX shock scenarios

---

### **Robustness**

* Adversarial scenario testing:

  * Grid instability spikes
  * Heatwaves (T_amb > 45°C)

---

### **Extensibility**

* Plug alternative optimizers:

  * MOEA/D
  * CMA-ES

* Replace surrogate models

---

## **Recommended Stack**

* **ML:** LightGBM + PyTorch
* **Optimization:** pymoo (NSGA-II), BoTorch (Bayesian)
* **Numerics:** NumPy + SciPy
* **Acceleration:** CUDA (optional)

---

## **Certainty Rating**

**0.93**

Key uncertainty:

* Surrogate model fidelity due to limited real Na-ion lifecycle datasets
* Accuracy of thermal probability estimation under sparse Nigerian climate extremes
