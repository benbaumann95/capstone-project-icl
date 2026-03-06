# Methodology: Bayesian Optimization for Black-Box Functions

This document describes the core methodology, design decisions, and academic foundations for the BBO Capstone Challenge.

---

## Bayesian Optimization Loop

The following diagram illustrates the iterative BO process used throughout this project:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BAYESIAN OPTIMIZATION LOOP                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │  Initialize  │
                              │ with random  │
                              │   samples    │
                              └──────┬───────┘
                                     │
                                     ▼
           ┌─────────────────────────────────────────────────────┐
           │                                                     │
           │  ┌─────────────────────────────────────────────┐   │
           │  │           1. FIT SURROGATE MODEL            │   │
           │  │                                             │   │
           │  │   Weeks 1-3: Gaussian Process (GP)          │   │
           │  │   Weeks 4-6: Neural Network Ensemble        │   │
           │  │   Weeks 7-11: Multi-Kernel GP Ensemble      │   │
           │  │                                             │   │
           │  │   Input: D = {(x_i, y_i)}_{i=1}^n          │   │
           │  │   Output: p(y|x, D) with uncertainty        │   │
           │  └─────────────────┬───────────────────────────┘   │
           │                    │                               │
           │                    ▼                               │
           │  ┌─────────────────────────────────────────────┐   │
           │  │        2. OPTIMIZE ACQUISITION FUNCTION     │   │
           │  │                                             │   │
           │  │   UCB: α(x) = μ(x) + κ·σ(x)                │   │
           │  │                                             │   │
           │  │   Find: x* = argmax α(x)                    │   │
           │  │              x ∈ [0,1]^d                    │   │
           │  └─────────────────┬───────────────────────────┘   │
           │                    │                               │
           │                    ▼                               │
           │  ┌─────────────────────────────────────────────┐   │
           │  │           3. QUERY BLACK-BOX               │   │
           │  │                                             │   │
           │  │   Submit x* to unknown function f           │   │
           │  │   Receive y* = f(x*) + ε                    │   │
           │  └─────────────────┬───────────────────────────┘   │
           │                    │                               │
           │                    ▼                               │
           │  ┌─────────────────────────────────────────────┐   │
           │  │           4. UPDATE DATASET                │   │
           │  │                                             │   │
           │  │   D ← D ∪ {(x*, y*)}                        │   │
           │  └─────────────────┬───────────────────────────┘   │
           │                    │                               │
           │                    │                               │
           └────────────────────┴───────────────────────────────┘
                                │
                                ▼
                         [Repeat weekly]
```

---

## Surrogate Model Evolution

### Phase 1: Gaussian Processes (Weeks 1-3)

```
Input x ──► GP with Matern Kernel ──► μ(x), σ²(x)
              │
              ├── Kernel: k(x,x') = σ² · Matern(ν=2.5)
              ├── Training: O(n³) complexity
              └── Uncertainty: Analytical posterior
```

**Advantages:**
- Principled uncertainty quantification
- Works well with small data (n < 20)
- Smooth interpolation

**Limitations:**
- Cubic scaling with dataset size
- No gradient of prediction w.r.t. input
- Assumes stationarity

### Phase 2: Neural Network Ensembles (Weeks 4-6)

```
                    ┌─────────────┐
                    │   NN_1      │──► ŷ_1
                    └─────────────┘
Input x ──►         ┌─────────────┐
      │             │   NN_2      │──► ŷ_2
      │             └─────────────┘           ┌───────────────┐
      ├──────────►  ┌─────────────┐     ──►   │ μ = mean(ŷ)  │
      │             │   NN_3      │──► ŷ_3    │ σ = std(ŷ)   │
      │             └─────────────┘           └───────────────┘
      │             ┌─────────────┐
      │             │   ...       │──► ...
      │             └─────────────┘
      │             ┌─────────────┐
      └──────────►  │   NN_7      │──► ŷ_7
                    └─────────────┘

Architecture per NN:
┌────────┐   ┌───────────┐   ┌──────┐   ┌─────────┐   ┌────────┐
│ Input  │──►│ LayerNorm │──►│ ReLU │──►│ Dropout │──►│ Output │
│  (d)   │   │  + Linear │   │      │   │         │   │  (1)   │
└────────┘   └───────────┘   └──────┘   └─────────┘   └────────┘
                  ↑                           │
                  └───────────────────────────┘
                        (repeated 2-3x)
```

**Advantages:**
- O(n) training complexity
- Gradient ∂y/∂x via backpropagation
- Can learn non-stationary patterns
- Diverse architectures improve uncertainty

### Phase 3: NeurIPS 2020 BBO Techniques (Week 7)

Based on winning solutions from the NeurIPS 2020 Black-Box Optimization Challenge.

#### TuRBO (Trust Region Bayesian Optimization)

```
┌────────────────────────────────────────────────────────────────┐
│                    TURBO ALGORITHM                              │
└────────────────────────────────────────────────────────────────┘

Initialize: center = x_best, length = 0.1

While not converged:
    │
    ├── Generate Sobol candidates within trust region
    │       candidates ∈ [center - length/2, center + length/2]
    │
    ├── Thompson Sampling: sample from GP posterior
    │       Select x* = argmax(posterior_sample)
    │
    ├── Query f(x*), observe y*
    │
    └── Update trust region:
            │
            ├── If y* > best:  success_counter++
            │       If success_counter ≥ 3:
            │           length = min(2 * length, 0.5)
            │           success_counter = 0
            │
            └── If y* ≤ best:  failure_counter++
                    If failure_counter ≥ 3:
                        length = length / 2
                        failure_counter = 0
                        If length < 0.01: RESTART
```

**Key Insight**: Traditional BO assumes stationarity. TuRBO adapts search radius based on local success rate.

#### Multi-Kernel GP Ensemble

```
Input x ──► ┌─────────────────┐     Weight by
            │ GP (Matern-0.5) │──► ŷ_1  ──┐  marginal
            └─────────────────┘          │  likelihood
            ┌─────────────────┐          │
      ──►   │ GP (Matern-1.5) │──► ŷ_2  ──┼──► μ = Σ w_i ŷ_i
            └─────────────────┘          │     σ² = Σ w_i σ²_i
            ┌─────────────────┐          │
      ──►   │ GP (Matern-2.5) │──► ŷ_3  ──┤
            └─────────────────┘          │
            ┌─────────────────┐          │
      ──►   │ GP (RBF)        │──► ŷ_4  ──┘
            └─────────────────┘

Weights: w_i = softmax(log_marginal_likelihood_i)
```

**Why Multiple Kernels?**
- Matern ν=0.5: Models rough, non-differentiable functions
- Matern ν=1.5: Once-differentiable functions
- Matern ν=2.5: Twice-differentiable (smooth)
- RBF: Infinitely differentiable (very smooth)

**Finding**: Functions F1-F6 are best modeled by Matern-0.5 (rough), while F7-F8 prefer Matern-2.5 (smooth).

#### Hybrid NN-GP Ensemble

```
                    ┌───────────────────┐
                    │  NN Ensemble (7)  │──► μ_nn, σ_nn
                    └───────────────────┘
Input x ──►                                    ┌─────────────────┐
                    ┌───────────────────┐      │ Weighted Avg:   │
                    │  GP Ensemble (4)  │──► ──►│ μ = α·μ_nn +   │
                    └───────────────────┘      │     (1-α)·μ_gp  │
                                               └─────────────────┘

α = 0.5 (equal weighting by default)
```

**Benefits**: Combines NN's gradient access with GP's principled uncertainty.

#### BOUNDARY_REFINE Strategy

For functions with optima near domain boundaries (like F5 where x₂, x₃ → 1.0):

```
BOUNDARY_REFINE:
    1. Generate Sobol candidates within small trust region
    2. Filter candidates biased toward boundary
       (e.g., x₂ > threshold, x₃ > threshold)
    3. Thompson Sampling among boundary-biased candidates
    4. Select candidate that both:
       - Has high posterior sample
       - Pushes relevant dimensions toward boundary
```

**Rationale**: F5's optimum at [0.36, 0.27, 0.996, 0.998] shows x₂, x₃ should be maximized. BOUNDARY_REFINE exploits this structure by biasing exploration toward the boundary while still using principled Thompson Sampling.

---

## Key Design Decisions

### 1. Acquisition Function: UCB over EI

| Criterion | UCB | EI |
|-----------|-----|-----|
| **Formula** | μ + κσ | E[max(y - y_best, 0)] |
| **Tuning** | Single parameter κ | Implicit |
| **Exploration** | Explicit via κ | Implicit |
| **Interpretation** | Confidence bound | Expected gain |

**Decision:** UCB chosen for explicit exploration-exploitation control via κ.

```
κ values used:
  - κ = 1.5-2.0: Balanced (default)
  - κ = 3.0-5.0: Aggressive exploration (F1 early weeks)
  - κ = 1.0: Exploitation focus (late weeks)
```

### 2. Ensemble Size: 7 Models

Based on Lakshminarayanan et al. (2017), 5 models suffice for most tasks. We use 7 for:
- More stable uncertainty estimates
- Better coverage of function hypothesis space
- Odd number breaks ties in agreement voting

### 3. Trust Region Constraints

For exploiting known good regions safely:

```
Trust Region Step:
  x_new = x_best + r · (∇f / ||∇f||)

Where:
  r = trust radius (0.01-0.05)
  ∇f = ensemble gradient at x_best

Constraint: ||x_new - x_best|| ≤ r
```

### 4. Strategy Selection Framework

```
┌────────────────────────────────────────────────────────────────┐
│                    STRATEGY SELECTION TREE                     │
└────────────────────────────────────────────────────────────────┘

Is this the FINAL week?
    │
    ├── YES → EXACT_RETURN to best known point (protect gains)
    │
    └── NO → ALL exploration (never waste queries on known values):
              │
              ├── Found breakthrough (>>2x improvement)?
              │       └── TURBO_EXPLOIT (small TR, careful exploration)
              │
              ├── Near boundary optimum (e.g., F5)?
              │       └── BOUNDARY_REFINE (push toward bounds)
              │
              ├── Consistent improvement trend?
              │       └── TURBO_EXPLOIT (continue momentum)
              │
              ├── Stagnant at local optimum?
              │       └── KERNEL_ENSEMBLE (try different smoothness)
              │
              └── Unknown/unstable?
                      └── TURBO_EXPLORE (moderate trust region)

KEY INSIGHT: EXACT_RETURN wastes a query on already-known values.
Reserve it ONLY for the final week to lock in best scores.
```

### 5. Boundary Handling

All queries constrained to [0, 1] (the full domain):
- **Critical bug fix (Week 9)**: The original [0.01, 0.99] clipping was a bug that prevented F5 from reaching its boundary optimum at x2, x3 near 1.0, costing ~200 points across 7 submissions
- Boundary-aware trust regions handle functions with optima near domain edges
- Functions with boundary optima (e.g., F5) use pinning to lock specific dimensions at boundary values

---

## Week-by-Week Strategy Summary

| Week | Module | Surrogate | Primary Strategy | Key Insight |
|------|--------|-----------|------------------|-------------|
| 1 | 12 | GP | UCB exploration | Map the space |
| 2 | 13 | GP | Hybrid exploit/explore | Protect good solutions |
| 3 | 14 | GP | Conservative exploit | Local perturbation works |
| 4 | 15 | NN Ensemble | Gradient-guided | NNs enable ∂y/∂x |
| 5 | 16 | NN Ensemble | Function-specific | F1 breakthrough at [0.63, 0.64] |
| 6 | 17 | NN Ensemble | Exploratory | Explore now, consolidate later |
| 7 | 18 | Hybrid NN-GP | TuRBO + Multi-Kernel | NeurIPS 2020 winning techniques |
| 8 | 19 | Multi-Kernel GP | PI exploitation | 5/8 new bests; best week overall |
| 9 | 20 | Multi-Kernel GP | Bug fixes + PI | Domain clipping fix; 4/8 new bests |
| 10 | 21 | Multi-Kernel GP | Coordinate-wise + constraints | 4/8 new bests; F1 +12.4% |
| 11 | 22 | Multi-Kernel GP | HEBO-inspired upgrades | Output warping, noisy EI, multi-acquisition |

---

## Results Summary

### Best Values Found (as of Week 11)

| Function | Dim | Best Value | Week Found | Strategy |
|----------|-----|------------|------------|----------|
| F1 | 2D | **1.993** | Week 10 | Coordinate-wise exploitation |
| F2 | 2D | 0.667 | Week 4 | Stagnant; confirmed noisy |
| F3 | 3D | **-0.0117** | Week 9 | Trajectory following |
| F4 | 4D | **0.629** | Week 8 | Tight PI exploitation |
| F5 | 4D | **1675.3** | Week 10 | Boundary pinning (x2, x3 near 1.0) |
| F6 | 5D | **-0.586** | Week 8 | Half-step trajectory |
| F7 | 6D | **2.480** | Week 10 | Directional exploitation |
| F8 | 8D | **9.937** | Week 10 | Ultra-tight exploitation |

---

## Academic References

### Core Bayesian Optimization

1. **Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2016).**
   *Taking the Human Out of the Loop: A Review of Bayesian Optimization.*
   Proceedings of the IEEE, 104(1), 148-175.
   - Comprehensive survey of BO methods
   - Foundation for acquisition function selection

2. **Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010).**
   *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design.*
   ICML 2010.
   - Theoretical justification for GP-UCB
   - Regret bounds for acquisition functions

3. **Snoek, J., Larochelle, H., & Adams, R. P. (2012).**
   *Practical Bayesian Optimization of Machine Learning Hyperparameters.*
   NeurIPS 2012.
   - Practical BO for hyperparameter tuning
   - Trust region and high-dimensional considerations

### Neural Network Uncertainty

4. **Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).**
   *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.*
   NeurIPS 2017.
   - Foundation for our ensemble approach
   - Shows 5+ models provide calibrated uncertainty

5. **Gal, Y., & Ghahramani, Z. (2016).**
   *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.*
   ICML 2016.
   - Theoretical justification for dropout-based uncertainty

### NeurIPS 2020 BBO Challenge Techniques

6. **Eriksson, D., Pearce, M., Gardner, J., Turner, R., & Poloczek, M. (2019).**
   *Scalable Global Optimization via Local Bayesian Optimization.*
   NeurIPS 2019.
   - TuRBO: Trust Region Bayesian Optimization
   - Adaptive trust regions based on local success rate
   - Foundation for Week 7 implementation

7. **Turner, R., Eriksson, D., et al. (2021).**
   *Bayesian Optimization is Superior to Random Search for Machine Learning Hyperparameter Tuning: Analysis of the Black-Box Optimization Challenge 2020.*
   arXiv:2104.10201.
   - Analysis of NeurIPS 2020 BBO Challenge results
   - Comparison of winning strategies

8. **Optuna Team (2020).**
   *bboc-optuna-developers - 5th Place Solution.*
   GitHub: optuna/bboc-optuna-developers
   - Multi-kernel GP ensemble approach
   - Kernel selection by marginal likelihood

9. **RAPIDS AI Team (2020).**
   *rapids-ai-BBO-2nd-place-solution.*
   GitHub: daxiongshu/rapids-ai-BBO-2nd-place-solution
   - Hybrid optimizer ensemble (TuRBO + scikit-optimize)
   - Improved scores from 88.9 → 92.9

10. **Cowen-Rivers, A. I., Lyu, W., Tutunov, R., Wang, Z., Grosnit, A., Griffiths, R. R., ... & Bou-Ammar, H. (2022).**
   *HEBO: Pushing the Limits of Sample-Efficient Hyperparameter Optimisation.*
   Journal of Artificial Intelligence Research, 74, 1269-1349.
   - NeurIPS 2020 BBO Challenge winner (Huawei Noah's Ark Lab)
   - Output warping, noisy EI, multi-acquisition ensemble
   - Foundation for Week 11 improvements

### Gaussian Processes

6. **Rasmussen, C. E., & Williams, C. K. I. (2006).**
   *Gaussian Processes for Machine Learning.*
   MIT Press.
   - Definitive reference for GP theory
   - Kernel selection and hyperparameter optimization

### Deep Learning Foundations

7. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).**
   *Deep Learning.*
   MIT Press.
   - Neural network architectures
   - Optimization and regularization techniques

8. **He, K., Zhang, X., Ren, S., & Sun, J. (2015).**
   *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.*
   ICCV 2015.
   - Kaiming initialization used in our NNs

### Normalization Techniques

9. **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).**
   *Layer Normalization.*
   arXiv:1607.06450.
   - LayerNorm used in our ensemble architecture

---

## Software and Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.x | Neural network implementation |
| scikit-learn | 1.x | Gaussian Processes, StandardScaler |
| NumPy | 1.x | Numerical operations |
| Pandas | 2.x | Data management |
| Matplotlib | 3.x | Visualization |

---

## Reproducibility

All experiments use fixed random seeds:
```python
np.random.seed(42)
torch.manual_seed(42)
```

Notebooks are designed to be run end-to-end with deterministic results.
