## **Reduction Engine**

---

## **1. Domain Summary – Purpose & Scope**

A **stateful pipeline ViewModel** that:

* Pulls Nigerian grid + irradiance datasets
* Streams quantum material datasets
* Applies **environment-calibrated reduction (10⁶ → 10³)**

This layer acts as:

> **Execution controller + dataflow graph + state container**

---

## **2. Core Entities / Data Model**

(unchanged structurally, but extended for ViewModel state)

```yaml
PipelineState:
  job_id: string
  status: enum [INIT, RUNNING, REDUCING, COMPLETE, FAILED]
  progress: float
  processed_count: int
  retained_count: int

StreamBuffer:
  chunk_id: int
  materials: list<MaterialCandidate>

ReductionState:
  reservoir: list<MaterialCandidate>
  kernel_matrix: matrix
  current_size: int

EnvironmentState:
  config: EnvironmentConfig
  calibration: LFPCalibration
```

---

## **3. ViewModel Interfaces**

```java
interface DataReductionViewModel {

    // Phase 0
    void calibrateEnvironment();

    // Data acquisition
    void fetchGridData();
    void fetchIrradianceData();

    // Streaming control
    void startMaterialStream();
    StreamBuffer nextChunk();

    // Core processing
    DerivedFeatures computeFeatures(MaterialCandidate m);
    boolean applyHardFilters(DerivedFeatures f);

    // Reduction
    void updateReduction(MaterialCandidate m);

    // Conditioning
    MaterialCandidate applyEnvironmentalTransform(MaterialCandidate m);

    // Execution control
    void runPipeline();

    // State access
    PipelineState getState();
    List<MaterialCandidate> getReducedSet();
}
```

---

## **4. Service Workflows (ViewModel Execution Graph)**

### **4.1 Execution Entry Point**

```text
runPipeline()
  ├── calibrateEnvironment()
  ├── fetchGridData()
  ├── fetchIrradianceData()
  ├── startMaterialStream()
  ├── loop (chunk)
  │     ├── nextChunk()
  │     ├── for material in chunk:
  │     │     ├── computeFeatures()
  │     │     ├── applyHardFilters()
  │     │     ├── updateReduction()
  ├── finalizeReduction()
  ├── applyEnvironmentalTransform()
  └── COMPLETE
```

---

### **4.2 Streaming Engine (Core Constraint)**

**Design Requirement:**
Avoid:
[
O(N) \text{ memory}
]

**Implementation:**

* Generator / iterator pattern
* Chunk size:
  [
  B \approx 10^3
  ]

```python
def material_stream():
    for source in sources:
        for batch in source.fetch_batches():
            yield batch
```

---

### **4.3 Hard Filter Optimization (Critical Path)**

Reorder constraints by **cost-to-reject ratio**:

1. Cheap scalar checks:

   * (V_{redox})
   * (D_{Na})

2. Medium:

   * (C_{adj})

3. Expensive:

   * Thermal
   * Grid integrals

**Expected pruning efficiency:**
[
P(\text{survive}) \approx 0.01 - 0.1
]

---

### **4.4 Online DPP Reduction (Key Innovation)**

Full DPP is infeasible in streaming. Use:

### **Reservoir + Greedy DPP Approximation**

Maintain:

* Reservoir size (k \approx 1000)

#### **Algorithm**

For candidate (x):

1. Compute similarity vector:
   [
   k_i = \exp(-||x - x_i||^2 / \sigma^2)
   ]

2. Marginal gain:
   [
   \Delta = \log \det(K_{new}) - \log \det(K)
   ]

3. Decision:

* If reservoir not full → accept
* Else:

  * Replace element with lowest contribution if (\Delta > \epsilon)

#### **Optimization**

Avoid full determinant:

[
\log \det(K) = \sum \log \lambda_i
]

Use:

* Cholesky updates:
  [
  O(k^2)
  ]

---

### **4.5 Environmental Calibration (Numerical Core)**

Solve:

[
\min_\theta \sum_t (T_{model}(t; \theta) - T_{obs}(t))^2
]

**Discretized thermal model:**

[
T_{t+1} = T_t + \Delta t \left(\frac{Q^*}{C_{th}} - \frac{T_t - T_{amb}}{R_{th} C_{th}}\right)
]

**Fitting method:**

* L-BFGS (fast convergence)
* Constraints:

  * (R_{th} > 0, C_{th} > 0)

---

### **4.6 Environmental Conditioning**

Applied **after reduction** to avoid bias during selection.

```python
def transform(m):
    m.S_T *= w_thermal_env
    m.S_grid *= w_grid_env
    return m
```

---

## **5. ViewModel State Transitions**

```text
INIT → CALIBRATING → STREAMING → FILTERING → REDUCING → CONDITIONING → COMPLETE
```

Failure states:

* DATA_UNAVAILABLE
* NUMERICAL_DIVERGENCE
* MEMORY_LIMIT

---

## **6. NFR – Non-Functional Requirements**

### **Performance**

Target:

[
T = O(N \cdot d + k^2 N_{accepted})
]

Where:

* (d): feature dimension
* (k = 10^3)

---

### **Memory**

[
O(kd + B d)
]

* Reservoir + active chunk only

---

### **Concurrency Model**

* Parallelize per chunk:

  * Thread pool (CPU-bound)
* Avoid shared kernel writes:

  * Use lock-free queue or staged merge

---

### **Numerical Stability**

* Normalize features before DPP:
  [
  x' = \frac{x}{||x||}
  ]

* Add jitter:
  [
  K = K + \epsilon I
  ]

---

### **Gotchas**

#### 1. **Streaming Bias**

Early samples dominate reservoir.

**Mitigation:**
[
P(\text{accept}) = \min(1, \frac{k}{t})
]

---

#### 2. **Thermal Overestimation**

LFP → Na-ion mismatch.

Mitigate via:
[
S_T^{eff} = \beta S_T,\quad \beta \in [0.6, 1.2]
]

---

#### 3. **Grid Noise Instability**

Frequency data often missing.

Model:
[
df = \theta(\mu - f)dt + \sigma dW_t
]

(Ornstein–Uhlenbeck)

---

#### 4. **Cost Model Drift (Nigeria)**

FX spikes non-linear.

Better model:
[
C_{adj} = C(1 + \sigma_{fx}^2 + import_factor)
]

---

### **Extensibility**

* Plug alternative reducers:

  * PCA + farthest point sampling
* Replace DPP kernel:

  * cosine similarity
  * Mahalanobis distance

---

## **Recommended Implementation Stack**

* **Core Engine:** Python (NumPy + SciPy)
* **Performance-critical (DPP):** Rust (via PyO3)
* **Data Access:**

  * Materials → REST adapters wrapped internally
  * Grid → CSV + scraping fallback
* **Local Storage:** DuckDB (fast analytical queries)

---

## **Certainty Rating**

**0.94**

Remaining uncertainty:

* Stability of online DPP under highly skewed material distributions
* Quality of Nigerian grid disturbance reconstruction without high-resolution SCADA data

