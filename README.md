# Capstone Project – Black-Box Optimization Challenge

This repository contains the code and analysis for the **Black-Box Optimization (BBO) Challenge**, part of the Imperial College Professional Certificate in Machine Learning & Artificial Intelligence.

---

## Section 1: Project Overview

### What is the BBO Capstone Project?

The Black-Box Optimization (BBO) Capstone Challenge tasks participants with optimizing 8 unknown functions over multiple weeks. The functions are "black-box" meaning we have no knowledge of their mathematical form, derivatives, or structure. We can only observe input-output pairs by submitting queries and receiving scalar responses.

### Overall Goal

The goal is to **maximize** each of the 8 unknown functions by intelligently proposing query points within a bounded domain. Each week, we submit one query per function and receive the function's response. Over time, we build a model of each function and use it to propose increasingly optimal query points.

### Real-World ML Relevance

Black-box optimization is ubiquitous in real-world machine learning:

- **Hyperparameter Tuning**: Optimizing learning rates, regularization, and architecture choices without closed-form gradients
- **AutoML**: Automated model selection and configuration
- **Drug Discovery**: Optimizing molecular properties where simulations are expensive
- **A/B Testing**: Finding optimal configurations with limited experimental budget
- **Industrial Process Optimization**: Maximizing yield or quality when the underlying physics is complex or unknown

The constraint of extremely limited queries (1 per week per function) mirrors real scenarios where evaluations are expensive—whether due to computational cost (training a large neural network), time (waiting for experimental results), or monetary cost (manufacturing prototypes).

### Career Relevance

This project directly supports career development in several ways:

1. **Applied ML Engineering**: Practical experience with Bayesian Optimization, Gaussian Processes, and acquisition functions—techniques used at companies like Google, Meta, and Amazon for hyperparameter optimization
2. **Decision-Making Under Uncertainty**: Learning to balance exploration vs exploitation is a transferable skill for product decisions, resource allocation, and strategic planning
3. **Sample-Efficient Learning**: Understanding how to extract maximum value from limited data—critical in domains with expensive data collection
4. **Scientific Method Application**: Iteratively forming hypotheses, testing them, and refining approaches based on evidence

---

## Section 2: Inputs and Outputs

### Inputs

**Domain**: Each function operates on a d-dimensional unit hypercube [0, 1]^d

| Function | Dimensionality | Input Space |
|----------|----------------|-------------|
| 1 | 2D | [0,1]² |
| 2 | 2D | [0,1]² |
| 3 | 3D | [0,1]³ |
| 4 | 4D | [0,1]⁴ |
| 5 | 4D | [0,1]⁴ |
| 6 | 5D | [0,1]⁵ |
| 7 | 6D | [0,1]⁶ |
| 8 | 8D | [0,1]⁸ |

**Query Format**: Hyphen-separated decimal coordinates

```
Example (4D function): 0.362718-0.273413-0.996088-0.997538
```

### Outputs

**Response**: A single real-valued scalar y ∈ ℝ representing the function's value at the queried point.

**Observed Ranges by Function** (from collected data):

| Function | Output Range | Characteristics |
|----------|--------------|-----------------|
| 1 | [-0.004, ~0] | Extremely sparse; most outputs near-zero |
| 2 | [-0.07, 0.61] | Multimodal with positive region |
| 3 | [-0.40, -0.04] | All negative; monotonic trends |
| 4 | [-32.6, 0.6] | Mostly negative; rare positive region |
| 5 | [0.11, 1618.5] | Extremely wide range; corner optimum |
| 6 | [-2.57, -0.71] | All negative; narrow range |
| 7 | [0.003, 2.29] | Multimodal; moderate variance |
| 8 | [5.59, 9.90] | All positive; narrow high-value range |

### Example Input/Output Pair

```python
# Function 5 (4D)
input:  x = [0.362718, 0.273413, 0.996088, 0.997538]
output: y = 1618.49

# Function 2 (2D)
input:  x = [0.912345, 0.456789]
output: y = 0.611
```

---

## Section 3: Challenge Objectives

### Primary Goal

**Maximize** each of the 8 unknown black-box functions. The objective is to find query points that produce the highest possible function values.

### Constraints and Limitations

1. **Extreme Query Budget**: Only 1 query per function per week
   - Over 3 weeks: only 3 data points added per function
   - Total samples: 12-42 per function (including initial random samples)

2. **Unknown Function Structure**: No access to:
   - Mathematical form or closed-form expression
   - Gradient information
   - Continuity/smoothness guarantees
   - Number or location of optima

3. **Response Delay**: Results are not immediate; weekly submission cycles require planning ahead

4. **No Re-querying**: Each query is permanent—cannot undo or retry a poor choice

5. **Diverse Dimensionality**: Functions range from 2D to 8D, requiring scalable strategies

### Success Metrics

- **Maximum Value Found**: The highest observed output per function
- **Consistency**: Avoiding catastrophic drops from good regions
- **Learning Efficiency**: Improvement rate given the extreme sample constraint

---

## Section 4: Technical Approach

*This section is a living record updated as the approach evolves.*

### Core Methodology: Bayesian Optimization

The primary approach uses **Bayesian Optimization (BO)** with **Gaussian Processes (GPs)** as the surrogate model.

**Why Bayesian Optimization?**
- Principled uncertainty quantification
- Natural exploration-exploitation trade-off
- Sample-efficient—designed for expensive evaluations

#### Surrogate Model

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
    length_scale=np.ones(dim),
    length_scale_bounds=(1e-2, 10),
    nu=2.5
)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
```

- **Matern Kernel (ν=2.5)**: Provides appropriate smoothness assumptions; more flexible than RBF
- **Constant Kernel Scaling**: Allows model to adapt to output magnitude

#### Acquisition Functions

1. **Upper Confidence Bound (UCB)**: `α(x) = μ(x) + κ·σ(x)`
   - Balances exploitation (high mean μ) with exploration (high uncertainty σ)
   - κ parameter controls exploration aggressiveness

2. **Expected Improvement (EI)**: Measures expected gain over current best
   - Used for fine-tuning when exploitation is prioritized

### Week-by-Week Strategy Evolution

#### Week 1: Exploratory Phase

**Strategy**: GP with UCB, function-adaptive κ values

| Function | κ Value | Rationale |
|----------|---------|-----------|
| 1 | 10.0 | All initial outputs ~0; aggressive exploration needed |
| 2-8 | 1.96 | Standard 95% confidence bound; balanced approach |

**Key Insight**: Function 1's degenerate initial data (all near-zero) required fundamentally different treatment.

#### Week 2: Hybrid Exploitation-Exploration

**Paradigm Shift**: Introduced function-specific strategy switching

| Strategy | Functions | Method |
|----------|-----------|--------|
| **Exploitation** | F5, F8 | Local perturbation around best known point |
| **Exploration** | F1, F2, F3, F6 | Increased κ (2.5-5.0) or targeted regions |
| **Hybrid** | F4, F7 | Context-dependent switching |

**Critical Lesson Learned**:
- F4: Week 1 value 0.60 → Week 2 over-explored → -1.33 (major regression)
- F7: Week 1 value 2.29 → Week 2 exploration → 0.33 (degradation)

**Conclusion**: Protect good solutions; over-exploration is risky with limited budget.

#### Week 3: Conservative Exploitation

**Strategy**: Highly tailored per-function approaches based on cumulative data

| Function | Strategy | Perturbation Scale | Rationale |
|----------|----------|-------------------|-----------|
| 1 | Grid Exploration | N/A | Systematic search for sparse needle |
| 2 | Local Exploit | 0.08 | Return to high-value region |
| 3 | UCB (κ=2.0) | N/A | Continue improving trend |
| 4 | Strong Exploit | 0.03 | Protect only positive point found |
| 5 | Fine-tune + Bias | 0.02 | Directional nudge toward x₂,x₃→1 |
| 6 | Targeted Exploit | 0.10 | Explore around best initial point |
| 7 | Careful Exploit | 0.05 | Week 1 peak is global best |
| 8 | Micro-perturbation | 0.02 | Near-optimal; fine-tuning only |

### Advanced Techniques Developed

#### 1. Local Perturbation for Exploitation

```python
def local_perturbation(best_x, bounds, scale=0.05):
    perturbation = np.random.normal(0, scale, size=len(best_x))
    new_x = best_x + perturbation
    return np.clip(new_x, bounds[:, 0], bounds[:, 1])
```

**Rationale**: With limited data, GP extrapolation is unreliable. Gaussian perturbation around known good points provides safer exploitation.

#### 2. Directional Biasing (Function 5)

**Observation**: High x₂, x₃ values correlate with high outputs
**Solution**: Add bias vector to nudge queries toward promising region

```python
bias = np.array([0.0, 0.0, 0.003, 0.002])  # Nudge x2, x3 toward 1.0
new_x = best_x + perturbation + bias
```

#### 3. Adaptive Perturbation Scaling

**Rule**: Smaller perturbations for higher-quality solutions
- F8 (best=9.90): scale=0.02 (very fine)
- F6 (best=-0.71): scale=0.10 (broader search)

### Exploration vs Exploitation Balance

The balance evolved across weeks:

| Week | Exploration % | Exploitation % | Lesson |
|------|---------------|----------------|--------|
| 1 | 80% | 20% | Map the space; find promising regions |
| 2 | 50% | 50% | Switch to exploitation where successful |
| 3 | 15% | 85% | Protect good solutions; refine carefully |

**Key Insight**: With only 3 queries per function, we cannot afford wasted exploration after finding good regions. The cost of losing a good solution outweighs the potential benefit of finding a marginally better one.

### What Makes This Approach Thoughtful

1. **Adaptive Strategy**: No one-size-fits-all; each function gets tailored treatment based on observed behavior

2. **Learning from Mistakes**: Week 2's over-exploration failures directly informed Week 3's conservative approach

3. **Quantitative Decision-Making**: Perturbation scales inversely proportional to solution quality

4. **Hybrid Methods**: Combining principled BO with pragmatic heuristics (perturbation, grid search) based on function characteristics

5. **Feature Importance Detection**: Identifying which dimensions matter (F5: x₂, x₃ dominate) and exploiting this structure

### Considered but Not Fully Implemented

- **SVMs**: Could be used for classification (good vs bad regions) but GP regression provides more information
- **Ensemble Methods**: Multiple surrogate models could improve robustness
- **Trust Region Methods**: Could formalize the local perturbation approach
- **Multi-task Learning**: Sharing information across functions (not pursued due to different dimensionalities)

---

## Project Structure

```text
├── data/                     # Contains samples.csv for each function (initial + new data)
├── notebooks/                # Jupyter notebooks for weekly analysis and query generation
├── src/                      # Source code for reusable logic
│   ├── utils.py              # Helper functions (data loading, submission logging)
│   └── initialize_samples.py # Script to reset/init data from .npy files
├── submissions/              # Log of submitted queries
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1.  **Clone the repository**:

    ```bash
    git clone <repo-url>
    cd capstone-project-icl
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize Data** (if starting fresh):
    ```bash
    python src/initialize_samples.py
    ```

## Usage

### Weekly Workflow

1.  **Run the Weekly Notebook**: Open the appropriate notebook in `notebooks/` and run all cells to generate queries
2.  **Submit Queries**: Copy generated queries from notebook output or `submissions/submission_log.csv`
3.  **Update Data**: After receiving results, update `samples.csv` files in `data/function_X/`
4.  **Reflect**: Document strategy and reasoning in the notebook

### Notebooks

| Notebook | Week | Focus |
|----------|------|-------|
| `01_Module_12.ipynb` | 1 | Initial exploration; function-adaptive UCB |
| `02_Module_13.ipynb` | 2 | Hybrid exploitation-exploration strategy |
| `03_Module_14.ipynb` | 3 | Conservative exploitation with local perturbation |

---

## Author

Benjamin Baumann
