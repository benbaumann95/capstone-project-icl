# Model Card: Multi-Kernel GP Ensemble with Trust Region Bayesian Optimization

## Overview

| Field | Details |
|-------|---------|
| **Name** | Multi-Kernel GP Ensemble with Trust Region Bayesian Optimization |
| **Type** | Bayesian Optimization system (surrogate model + acquisition function + trust region) |
| **Version** | v10 (Week 10 / Module 21) |
| **Author** | Benjamin Baumann |
| **Framework** | scikit-learn (Gaussian Processes), PyTorch (neural network ensembles), SciPy (optimization) |
| **Context** | Imperial College BBO Capstone Challenge |

---

## Intended Use

### Suitable tasks

- **Sample-efficient optimization** of expensive black-box functions where gradient information is unavailable
- **Low-to-moderate dimensional problems** (2-8 dimensions, tested up to 8D)
- **Sequential decision-making** under extreme budget constraints (as few as 1 evaluation per iteration)
- Real-world analogues: hyperparameter tuning, drug discovery screening, A/B testing, industrial process optimization

### Use cases to avoid

- **High-dimensional problems** (>20D): the GP surrogate scales poorly with dimensionality, and the trust region approach becomes ineffective as the volume of the search space grows exponentially
- **Batch parallel queries**: the current pipeline is designed for sequential, single-query submissions
- **Real-time optimization**: GP fitting involves matrix operations that may not meet latency requirements for time-critical applications
- **Functions with known structure**: if gradients, convexity, or other structural information is available, specialized optimizers will outperform this general-purpose approach

---

## Details: Strategy Across Ten Rounds

The optimization approach evolved significantly over 10 weeks of iterative refinement, driven by observed results and lessons learned from failures.

### Phase 1: Exploration and Baseline (Weeks 1-3)

The initial phase used standard Gaussian Processes with a single Matern kernel and Upper Confidence Bound (UCB) acquisition. The strategy was largely exploratory, with function-adaptive exploration parameters. Key lessons:

- Aggressive exploration on low-dimensional functions helped discover promising regions
- Over-exploration on functions with known good values caused regressions
- Local perturbation around known optima provided safer exploitation than GP-guided exploration

### Phase 2: Neural Network Surrogates (Week 4)

Replaced GPs with neural network ensembles (7 diverse MLPs) to leverage gradient-based query optimization via backpropagation. This enabled:

- Direct gradient computation for optimization direction
- Ensemble-based uncertainty quantification
- Scalable training compared to GP's cubic complexity

However, NN gradient estimates proved unreliable for large optimization steps, leading to regressions in several functions.

### Phase 3: Breakthrough and Trust Regions (Weeks 5-6)

A pivotal period that produced the largest single improvement (F1: 8x improvement via targeted grid search). Key developments:

- Introduction of trust region methods to formalize local exploitation
- Discovery that EXACT_RETURN (re-submitting known best coordinates) is a reliable safety net
- Recognition that micro-perturbation can cause regressions if not carefully controlled

### Phase 4: State-of-the-Art Techniques (Week 7)

Adopted techniques from the NeurIPS 2020 BBO Challenge, including:

- **TuRBO**: Trust Region Bayesian Optimization with adaptive radius
- **Multi-kernel GP ensemble**: Matern kernels with smoothness parameters ν = {0.5, 1.5, 2.5}, weighted by log marginal likelihood
- **Thompson Sampling**: posterior sampling for principled exploration within trust regions
- **Sobol sequences**: quasi-random space-filling for candidate generation

### Phase 5: Refined Exploitation (Weeks 8-10)

The mature phase of the approach, producing the majority of new best values. Key refinements:

- **Kernel weighting**: combined log marginal likelihood (60%) and leave-one-out cross-validation (40%) for robust kernel selection
- **Probability of Improvement (PI)** replaced Expected Improvement (EI) for exploitation — PI asks "what is the probability of beating the current best?" while EI rewards uncertainty, which can push queries away from known optima
- **Hard constraints**: per-dimension bounds to prevent queries from entering known-bad regions
- **Coordinate-wise search**: for extremely peaked functions, perturbing one dimension at a time reduces risk
- **Directional validation**: post-selection checks to ensure queries move in the expected improvement direction
- **Critical bug fixes**: corrected domain clipping that had prevented boundary-optimal queries from reaching the true boundary

### Key Innovations

1. **`selection` parameter**: explicit control over acquisition function choice (PI, EI, UCB, Mean), eliminating implicit overrides that caused unintended exploration
2. **`center_override`**: allows the trust region centre to be set independently of the GP's predicted optimum, enabling trajectory-following and hypothesis-testing strategies
3. **`constraints` parameter**: hard per-dimension bounds that filter candidates before acquisition function evaluation
4. **Boundary-aware trust regions**: specialized handling for functions with optima at or near domain boundaries

---

## Performance

### Results Summary (After 10 Weeks)

| Function | Dim | Initial Best | Current Best | Week Found | Strategy |
|----------|-----|-------------|-------------|------------|----------|
| F1 | 2D | ~0 | 1.773 | Week 8 | Coordinate-wise exploitation |
| F2 | 2D | 0.611 | 0.667 | Week 4 | Stagnant; exploring new directions |
| F3 | 3D | -0.035 | -0.012 | Week 9 | Trajectory following |
| F4 | 4D | 0.600 | 0.629 | Week 8 | Tight PI exploitation |
| F5 | 4D | 1618.5 | 1674.2 | Week 9 | Boundary pinning |
| F6 | 5D | -0.714 | -0.586 | Week 8 | Half-step trajectory |
| F7 | 6D | 2.290 | 2.448 | Week 9 | Directional exploitation |
| F8 | 8D | 9.066 | 9.933 | Week 9 | Ultra-tight exploitation |

### Performance Metrics

- **Maximum value found**: the primary metric — the highest observed output for each function
- **Consistency**: 7 of 8 functions improved from their initial best values; the best-performing weeks (W8, W9) achieved new records on 5 and 4 functions respectively
- **Learning efficiency**: the approach produced most improvements in weeks 7-9 as the surrogate model and strategy matured, demonstrating effective learning from accumulated observations

### Notable Results

- **F5 boundary fix** (Week 9): correcting a domain clipping bug produced the single largest improvement of the entire project, demonstrating that systematic code review can yield outsized returns
- **F1 breakthrough** (Week 5-8): discovering and refining an extremely peaked optimum that was invisible to initial random sampling
- **F8 steady refinement**: consistent marginal improvements over multiple weeks through tight exploitation, accumulating to meaningful gains

---

## Assumptions and Limitations

### Assumptions

1. **Stationarity**: the GP assumes the function's statistical properties are uniform across the domain. This is likely violated — some functions exhibit very different behaviour in different regions (e.g., extremely peaked optima surrounded by flat landscapes).

2. **Gaussian observation noise**: the model assumes evaluations are corrupted by i.i.d. Gaussian noise. Evidence from repeated evaluations suggests noise exists but may not follow this distribution.

3. **Kernel appropriateness**: the ensemble assumes one of three Matern smoothness levels (ν = 0.5, 1.5, 2.5) adequately describes each function. Functions with periodic structure, discontinuities, or other exotic properties would be poorly modelled.

4. **Local continuity of improvement trajectories**: for functions showing consistent directional improvement, the approach extrapolates this trend — but the trajectory could curve, plateau, or reverse at any point.

5. **Heuristic trust region sizing**: radii are chosen based on observed regression magnitudes rather than a principled criterion. There is no formal guarantee that the chosen radius is optimal.

### Limitations

1. **Single-query budget prevents hedging**: with only one evaluation per round, the approach cannot diversify between exploration and exploitation. Every submission is an irreversible binary bet.

2. **Exploitation bias**: the data becomes increasingly concentrated near known optima, making it difficult to discover distant, potentially superior regions. The GP's uncertainty in unsampled areas is high but cannot be reduced without dedicating queries to exploration.

3. **No principled stopping criterion for exploration**: when exploration fails repeatedly (as with F2), the approach cannot distinguish between "the current best is near-optimal" and "we have been exploring in the wrong directions."

4. **Scalability**: the GP ensemble has O(n^3) complexity in the number of observations and scales poorly beyond moderate dimensions. The approach was designed for the specific problem setting (2-8D, <50 observations) and would require significant adaptation for larger problems.

5. **Retrospective strategy selection**: the choice between PI, EI, UCB, and other acquisition functions is made based on post-hoc analysis of what worked in previous weeks, rather than a formal meta-learning framework. This introduces a risk of overfitting strategy choices to past outcomes.

---

## Ethical Considerations

### Transparency and reproducibility

- **Full reproducibility**: all notebooks use fixed random seeds (seed=42), and the complete pipeline — from data loading to query generation — is deterministic and can be rerun to produce identical results
- **Decision documentation**: each weekly notebook contains detailed commentary explaining the strategic reasoning behind every query, including what was tried, what failed, and why the approach was changed
- **Open source**: all code, data, and documentation are available in the GitHub repository

### Real-world adaptation

The techniques developed here transfer to real-world optimization scenarios:

- **Hyperparameter tuning**: the trust region approach is directly applicable to tuning ML model hyperparameters where each evaluation is expensive
- **Drug discovery**: the sample-efficient methodology suits scenarios where each experiment is costly or time-consuming
- **A/B testing**: the exploration-exploitation framework applies to sequential experimentation with limited budgets

### Risk considerations

- **No direct ethical risks**: the approach optimizes synthetic benchmark functions with no real-world consequences
- **Potential for misuse**: as a general optimization framework, it could theoretically be applied to adversarial objectives. However, the methodology itself is standard in the optimization literature and widely published
- **Bias in automation**: if deployed for real-world decisions (e.g., automated hiring pipelines), the exploitation-heavy strategy could reinforce existing biases in the training data by concentrating queries near historically successful regions

### How transparency supports this work

Transparent documentation of the decision-making process — including failures, bugs, and course corrections — makes the approach reproducible and auditable. Other practitioners can evaluate whether the strategic choices are sound, adapt the methodology to their own problems, and avoid repeating the same mistakes (e.g., using EI for exploitation, or applying overly aggressive domain clipping).

---

*Model card created following the framework from Mitchell et al. (2019), "Model Cards for Model Reporting."*
