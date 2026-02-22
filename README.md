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

**Observed Ranges by Function** (from collected data through Week 8):

| Function | Output Range | Best Found | Characteristics |
|----------|--------------|------------|-----------------|
| 1 | [-0.004, **1.773**] | 1.773 (W8) | Breakthrough at [0.63, 0.62]; rough surface |
| 2 | [-0.07, 0.667] | 0.667 (W4) | Multimodal; stagnant since Week 4 |
| 3 | [-0.40, -0.0145] | -0.0145 (W7) | All negative; improving slowly |
| 4 | [-32.6, **0.629**] | 0.629 (W8) | Mostly negative; breakthrough in Week 8 |
| 5 | [0.11, 1618.5] | 1618.5 (W1) | Extremely wide range; boundary optimum at x₂,x₃≈1 |
| 6 | [-2.57, **-0.586**] | -0.586 (W8) | All negative; 2 consecutive improvements (W7-W8) |
| 7 | [0.003, **2.433**] | 2.433 (W8) | Smooth function; steady improvement |
| 8 | [5.59, **9.928**] | 9.928 (W8) | All positive; near optimal |

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

#### Week 4: Neural Network Surrogate Models

**Paradigm Shift**: Replace Gaussian Processes with neural network ensembles

**Why Neural Networks?**
- **Gradient access via backpropagation**: Can compute ∂y/∂x to guide optimization
- **Scalability**: O(n) training vs GP's O(n³)
- **Feature learning**: Hidden layers can learn which dimensions matter
- **Flexible approximation**: Universal approximator for complex surfaces

**Implementation**:
```python
# Ensemble of MLPs for uncertainty quantification
class EnsembleSurrogate:
    - 5 MLPs with different random initializations
    - Prediction uncertainty = variance across ensemble
    - Gradient computation via backpropagation
```

| Function | Strategy | Method |
|----------|----------|--------|
| 1 | NN-UCB | High exploration (κ=3.0) |
| 2 | NN-Gradient | Gradient ascent from best initial |
| 3 | NN-UCB | Continue exploration |
| 4 | **Exact Return** | Return to Week 1 (only positive) |
| 5 | NN-Gradient | Gradient refinement of 1618 peak |
| 6 | NN-UCB | Exploration with κ=2.5 |
| 7 | NN-Exploit | Small perturbation of recovered peak |
| 8 | NN-Gradient | Micro-tuning of new best (9.91) |

#### Week 5: Breakthrough Discovery

**Major Result**: F1 achieved 8x improvement (0.196 → 1.626) at [0.634, 0.636]

| Function | Strategy | Result |
|----------|----------|--------|
| 1 | Grid search near [0.65, 0.65] | **1.626** (breakthrough!) |
| 2 | Micro-perturbation | 0.583 (regression) |
| 3 | NN-gradient | -0.040 (near best) |
| 4 | Exact Return | **0.600** (recovered) |
| 5 | Exact Return | **1618.5** (recovered) |
| 6 | Micro-perturbation | -0.735 (regression) |
| 7 | NN-gradient | **2.403** (new best!) |
| 8 | Exact Return | **9.915** (recovered) |

**Key Lessons**:
- EXACT_RETURN is reliable for protecting known optima
- MICRO_PERTURB can cause regressions (F2, F6)
- NN-gradient works well for steady improvement (F7)

#### Week 6: Exploratory Strategy

**Philosophy**: Explore aggressively now; consolidate in final week

| Function | Strategy | Rationale |
|----------|----------|-----------|
| 1 | Trust Region Gradient | Exploit breakthrough safely |
| 2 | NN-Gradient | Try to beat 0.667 |
| 3 | NN-Gradient | Escape -0.035 stagnation |
| 4 | Micro-Perturb | Tiny exploration around 0.600 |
| 5 | Boundary Push | Push x2, x3 toward 1.0 |
| 6 | NN-Gradient | Try to beat -0.714 |
| 7 | NN-Gradient | Continue improvement |
| 8 | NN-Gradient | Try to beat 9.915 |

#### Week 7: NeurIPS 2020 BBO Challenge Techniques

**Philosophy**: Apply state-of-the-art techniques from NeurIPS 2020 BBO Challenge winners

**Key Techniques Implemented**:

1. **TuRBO (Trust Region Bayesian Optimization)**: Adaptive trust regions that expand on success and contract on failure
2. **Multi-Kernel GP Ensemble**: Matern (ν=0.5, 1.5, 2.5) + RBF kernels weighted by marginal likelihood
3. **Hybrid NN-GP Ensemble**: Combines neural network ensembles with GP ensembles for robust predictions
4. **Thompson Sampling**: Posterior sampling for principled exploration within trust regions
5. **Sobol Sequences**: Quasi-random space-filling sequences for candidate generation

| Function | Strategy | Rationale |
|----------|----------|-----------|
| 1 | TURBO_EXPLOIT | Small trust region (0.05) around [0.634, 0.636] - exploit breakthrough |
| 2 | KERNEL_ENSEMBLE | Multi-kernel GP - escape stagnation at 0.667 |
| 3 | TURBO_EXPLORE | Medium trust region (0.1) - W6 showed improvement is possible |
| 4 | TURBO_EXPLOIT | Small TR (0.03) around 0.600 - explore nearby for potential improvement |
| 5 | BOUNDARY_REFINE | Push x2, x3 even closer to 1.0 boundary - exploit corner structure |
| 6 | KERNEL_ENSEMBLE | Multi-kernel GP - stagnant at -0.714 |
| 7 | TURBO_EXPLOIT | Small trust region (0.05) around W5 best - continue momentum |
| 8 | TURBO_EXPLOIT | Small TR (0.02) around 9.915 - fine-tune for potential improvement |

**Key Insight**: No EXACT_RETURN in Week 7 - with more weeks remaining, returning to already-queried coordinates wastes exploration opportunities. EXACT_RETURN is reserved for the final week only.

**Week 6 Analysis**: Most functions regressed due to unreliable NN gradient estimates for large steps. Multi-kernel GP analysis revealed functions F1-F6 have rough landscapes (Matern-0.5 best) while F7-F8 are smoother (Matern-2.5 best).

#### Week 8: Multi-Kernel Ensemble Breakthrough

**Best week of the project: 5 out of 8 new bests.**

**Key Techniques**:
- Multi-kernel GP ensemble (Matern ν=0.5, 1.5, 2.5) weighted by log marginal likelihood
- TuRBO trust regions with function-specific radii
- Combined acquisition functions (UCB, EI, PI, Thompson Sampling)

| Function | Strategy | Result |
|----------|----------|--------|
| 1 | Exploit (PI) | **1.773** (NEW BEST, +9%) |
| 2 | TuRBO | 0.584 (regression) |
| 3 | Exploit | -0.017 (slight regression) |
| 4 | Ensemble (EI) | **0.629** (NEW BEST, +5%) |
| 5 | Exploit | 1415.4 (regression — pulled from boundary) |
| 6 | Exploit | **-0.586** (NEW BEST, +14%) |
| 7 | TuRBO | **2.433** (NEW BEST, +1%) |
| 8 | Exploit | **9.928** (NEW BEST, +0.1%) |

#### Week 9: Deep Analysis, Bug Fixes & Corrected Strategies

**Major course correction** based on comprehensive audit of all 8 functions across all weeks.

**Critical Bugs Found & Fixed**:
1. **Domain clipping [0.01, 0.99]**: Both trust region classes clipped candidates to 98% of domain. F5 lost ~200 points on 7 of 8 submissions because x2/x3 couldn't reach >0.99. Fixed to [0, 1].
2. **Thompson Sampling override**: Threshold `0.01 * |y_best|` was scale-dependent. For F1, TS selected x1=0.592 (0.030 from peak — certain regression). Removed entirely.
3. **BoundaryAwareTrustRegion pinning order**: Pinning was applied BEFORE clipping, so np.clip(0.99) overrode the pin. Now pin AFTER clip.
4. **EI reversed F6's improvement trajectory**: x1 was consistently decreasing (0.155→0.145→0.126), but EI selected x1=0.155 (back to W4 value). Fixed with center_override.

**Key Innovations**:
- **`selection` parameter**: Explicit control over acquisition function (PI, EI, UCB, Mean, EI+UCB) — no more implicit TS overrides
- **`center_override` parameter**: Override trust region center when GP best ≠ strategic target
- **PI for exploitation**: PI asks "probability of beating current best" (conservative); EI rewards uncertainty (exploratory) — PI is correct for peaked functions
- **Trajectory-predicted centers**: For F6, extrapolated improvement direction to set trust region center
- **Boundary-aware optimization**: Pin F5's x₂, x₃ to (0.995, 1.0) with center at W1 optimum

| Function | Strategy | Selection | Key Fix |
|----------|----------|-----------|---------|
| F1 | Exploit (r=0.008) | PI | Radius matched to peak width (~0.02) |
| F2 | Exploit (r=0.015) | PI | Tight exploit; all previous tries pushed x1 too high |
| F3 | Exploit (r=0.010) | PI | Removed TS override that caused exploratory selection |
| F4 | Directional (r=0.03) | EI | Unchanged — already well-configured |
| F5 | Boundary (r=0.008) | PI | Fixed clipping bug; center at W1 point; x2/x3 near 1.0 |
| F6 | Directional (r=0.020) | EI | Trajectory center override to continue x1 decrease |
| F7 | Exploit (r=0.025) | PI | Center override to test x0 < 0.01 (was stuck at clip floor) |
| F8 | Exploit (r=0.010) | PI | Tighter radius; PI selection for conservative refinement |

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
| 4 | 30% | 70% | NN gradients enable smarter exploitation |
| 5 | 40% | 60% | Balance recovery with targeted exploration |
| 6 | 75% | 25% | Explore aggressively; can consolidate later |
| 7 | 25% | 75% | Use NeurIPS 2020 techniques; protect critical gains |
| 8 | 20% | 80% | Multi-kernel ensemble; exploit recent breakthroughs |
| 9 | 12% | 88% | Bug fixes; PI for exploitation; tight trust regions |

**Key Insight**: With limited queries, the strategy evolved from cautious exploitation (Weeks 3-4) to aggressive exploration (Week 6), then back to tight exploitation with bug fixes (Week 9). The Week 9 audit revealed that several "exploration" results were actually caused by bugs (domain clipping, TS override), not intentional strategy. With bugs fixed, exploitation is now more reliable and targeted.

### What Makes This Approach Thoughtful

1. **Adaptive Strategy**: No one-size-fits-all; each function gets tailored treatment based on observed behavior

2. **Learning from Mistakes**: Week 2's over-exploration failures directly informed Week 3's conservative approach

3. **Quantitative Decision-Making**: Perturbation scales inversely proportional to solution quality

4. **Hybrid Methods**: Combining principled BO with pragmatic heuristics (perturbation, grid search) based on function characteristics

5. **Feature Importance Detection**: Identifying which dimensions matter (F5: x₂, x₃ dominate) and exploiting this structure

### Techniques Implemented in Later Weeks

- **Neural Network Ensembles**: 7 diverse models for uncertainty quantification (Week 4+)
- **Trust Region Methods**: Formalized local perturbation with gradient guidance (Week 6)
- **Gradient-Based Query Optimization**: Using ∂y/∂x from NN surrogates (Week 4+)
- **TuRBO (Trust Region BO)**: Adaptive trust regions from NeurIPS 2019 (Week 7)
- **Multi-Kernel GP Ensemble**: 4 kernels (Matern ν=0.5/1.5/2.5, RBF) weighted by marginal likelihood (Week 7)
- **Hybrid NN-GP Ensemble**: Combines NN and GP predictions for robust uncertainty (Week 7)
- **Thompson Sampling**: Posterior sampling for exploration within trust regions (Week 7)
- **Sobol Sequences**: Quasi-random space-filling for candidate generation (Week 7)

### Considered but Not Pursued

- **Multi-task Learning**: Sharing information across functions (different dimensionalities make this difficult)
- **Bayesian Neural Networks**: Full posterior inference (ensembles provide sufficient uncertainty)
- **Meta-learning**: Learning acquisition function from past weeks (limited data)

---

## Project Structure

```text
├── data/                     # Contains samples.csv for each function (initial + new data)
│   └── function_N/           # Data for function N (N=1-8)
│       └── samples.csv       # All observed (x, y) pairs with source labels
├── docs/                     # Documentation
│   ├── datasheet.md          # Dataset documentation (datasheet)
│   ├── model_card.md         # Optimisation approach documentation (model card)
│   ├── methodology.md        # BO loop diagram, design decisions, citations
│   └── week9_deep_analysis.md # Comprehensive per-function audit (bugs, patterns, strategies)
├── notebooks/                # Jupyter notebooks for weekly analysis and query generation
│   ├── 01_Module_12.ipynb    # Week 1: GP-based exploration
│   ├── 02_Module_13.ipynb    # Week 2: Hybrid strategies
│   ├── 03_Module_14.ipynb    # Week 3: Conservative exploitation
│   ├── 04_Module_15.ipynb    # Week 4: Neural network surrogates
│   ├── 05_Module_16.ipynb    # Week 5: Breakthrough discovery
│   ├── 06_Module_17.ipynb    # Week 6: Trust region exploration
│   └── 07_Module_18.ipynb    # Week 7: NeurIPS 2020 BBO techniques
├── src/                      # Source code for reusable logic
│   ├── utils.py              # Helper functions (data loading, submission logging)
│   └── initialize_samples.py # Script to reset/init data from .npy files
├── submissions/              # Log of submitted queries
│   └── submission_log.csv    # All queries with timestamps
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
| `04_Module_15.ipynb` | 4 | Neural network surrogate with gradient-guided optimization |
| `05_Module_16.ipynb` | 5 | Breakthrough discovery; F1 peak found at [0.63, 0.64] |
| `06_Module_17.ipynb` | 6 | Trust region exploration; CNN/NN reflections |
| `07_Module_18.ipynb` | 7 | NeurIPS 2020 BBO techniques: TuRBO, Multi-Kernel GP, Hybrid Ensembles |
| `08_Module_19.ipynb` | 8 | Multi-kernel ensemble + TuRBO; LLM-centred strategy reflection |
| `09_Module_20.ipynb` | 9 | Directional exploitation, boundary-aware search, scaling & emergence |

---

## Documentation

- [Datasheet](docs/datasheet.md) - Dataset documentation: motivation, composition, collection process, and usage guidelines
- [Model Card](docs/model_card.md) - Optimisation approach documentation: strategy evolution, performance, assumptions, and limitations
- [Methodology](docs/methodology.md) - BO loop diagram, key design decisions, and academic citations
- [Week 9 Deep Analysis](docs/week9_deep_analysis.md) - Comprehensive audit of all 8 functions across all weeks: bugs found, data patterns, corrected strategies

---

## Key Results (After 8 Weeks)

| Function | Initial Best | Current Best | Week Found | Improvement |
|----------|-------------|-------------|------------|-------------|
| F1 | ~0 | **1.773** | Week 8 | Breakthrough + refinement |
| F2 | 0.611 | 0.667 | Week 4 | +9% (stagnant) |
| F3 | -0.035 | **-0.0145** | Week 7 | +59% |
| F4 | 0.600 | **0.629** | Week 8 | +5% (breakthrough) |
| F5 | 1618.5 | 1618.5 | Week 1 | Protected (boundary optimum) |
| F6 | -0.714 | **-0.586** | Week 8 | +18% (2 consecutive improvements) |
| F7 | 2.290 | **2.433** | Week 8 | +6% |
| F8 | 9.066 | **9.928** | Week 8 | +10% |

---

## Author

Benjamin Baumann
