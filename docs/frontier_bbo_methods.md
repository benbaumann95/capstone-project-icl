# Frontier Black-Box Optimization Methods

> Research compiled for the Imperial College BBO Capstone Challenge. Focus on competition-winning techniques applicable to our setting: 8 functions, 2-8D, ~25 data points, [0,1]^d domain.

---

## 1. HEBO — Winner of NeurIPS 2020 BBO Challenge

**Team**: Huawei Noah's Ark Lab (Cowen-Rivers et al., 2022)
**Paper**: "HEBO: Heteroscedastic Evolutionary Bayesian Optimisation"
**GitHub**: https://github.com/huawei-noah/HEBO

### Key Innovations (What Made It Win)

1. **Input Warping (Kumaraswamy Transform)**
   - Maps inputs x ∈ [0,1] → warped x via Kumaraswamy CDF
   - Handles non-stationarity: different regions of input space can have different length scales
   - The Kumaraswamy distribution has learnable concentration parameters (a, b)
   - Effect: stretches/compresses input space so a stationary kernel becomes appropriate
   - **We don't do this** — our GPs assume stationary kernels on raw inputs

2. **Output Warping (Power Transform / Box-Cox)**
   - Transforms y values to be more Gaussian before GP fitting
   - Handles skewed, heavy-tailed, or non-Gaussian output distributions
   - Uses Yeo-Johnson or Box-Cox transform with learned lambda
   - **We don't do this** — we only standardize (mean=0, std=1)

3. **Heteroscedastic Noise Model**
   - Models input-dependent noise: different regions have different noise levels
   - Critical for noisy functions (like our F2!)
   - Uses a separate GP or neural network to model noise variance σ²(x)
   - **We don't do this** — we use a single WhiteKernel (homoscedastic noise)

4. **Multi-Objective Acquisition Ensemble**
   - Instead of picking ONE acquisition function, optimizes MULTIPLE simultaneously
   - Uses NSGA-II (multi-objective evolutionary algorithm) to find Pareto front
   - Acquisition functions used together: UCB, EI, PI, and mean prediction
   - Selects from the Pareto front of candidates
   - **We don't do this** — we pick one acquisition function per function

5. **Evolutionary Acquisition Optimization**
   - Uses NSGA-II instead of random/Sobol sampling + argmax
   - Better at finding the true optimum of the acquisition surface
   - Especially important in higher dimensions where random sampling misses peaks
   - **We partially do this** — we use Sobol QMC (better than random, worse than evolutionary)

### HEBO Ablation Study Results
The ablation study showed that **input warping** and **multi-objective acquisition** were the two most impactful components. Removing either caused significant performance degradation.

---

## 2. Other NeurIPS 2020 BBO Challenge Participants

### JetBrains Research (2nd Place)
- **Method**: Learning Search Space Partition for Local Bayesian Optimization
- Partitions the search space into regions and runs local BO in the most promising partition
- Similar concept to TuRBO's trust regions but with learned partitions

### Optuna Team
- **Method**: TPE (Tree-structured Parzen Estimator) variant
- GitHub: https://github.com/optuna/bboc-optuna-developers
- TPE is a non-GP surrogate that models p(x|y) instead of p(y|x)
- Competitive but didn't win — GP-based methods were superior in this challenge

---

## 3. High-Dimensional BO: What Works Best

**Source**: "Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB" (ACM, 2024)

### Top Performers on BBOB Benchmarks
1. **SAASBO** (Sparse Axis-Aligned Subspace BO) — Eriksson & Jankowiak, 2021
   - Uses sparsity-inducing priors on GP length scales
   - Automatically identifies which dimensions matter most
   - Best for functions where only a few dimensions are important
   - **Applicable to us**: F5 (x2/x3 dominate), F8 (x0 is critical)

2. **TuRBO** (Trust Region BO) — Eriksson et al., 2019
   - We already use this concept (trust regions, local exploitation)
   - Key: adaptive trust region sizing (expand on success, contract on failure)
   - **We do this manually** — could automate the expansion/contraction

3. **Standard GP** — "Standard Gaussian Process is All You Need" (2024)
   - Surprising finding: standard GP with proper configuration often beats complex methods
   - Key: proper kernel selection, good hyperparameter optimization, sufficient restarts
   - **Validates our approach** — our multi-kernel ensemble is a strong baseline

---

## 4. BOTorch — Meta's State-of-the-Art BO Framework

**Paper**: Balandat et al., 2020
**GitHub**: https://github.com/pytorch/botorch

### Key Features We're Missing

1. **Knowledge Gradient (KG)**
   - Look-ahead acquisition: "what is the expected improvement in our model's recommendation after observing this point?"
   - Often outperforms EI/PI/UCB, especially in low-budget settings
   - Computationally more expensive but better decisions per query
   - **Highly relevant**: with only 1-2 queries left, making each one count is critical

2. **Input Warping (Kumaraswamy CDF)**
   - Same as HEBO — BOTorch has native support
   - `Warp` transform with learnable concentration parameters
   - Maps [0,1] → [0,1] with flexible nonlinear stretching
   - Handles non-stationary functions where different regions need different length scales

3. **Outcome Transform (Output Warping)**
   - `Standardize`: z-score normalization (we do this)
   - `Power`: Box-Cox/Yeo-Johnson transform (we don't do this)
   - `Log`: log transform for positive-valued functions (useful for F5 with values ~1600)

4. **Monte Carlo Acquisition Functions**
   - MC-based EI, PI, UCB using posterior sampling
   - More robust than analytical forms when GP posterior is non-Gaussian
   - Supports batch optimization (less relevant for us — 1 query/week)

5. **Noisy Expected Improvement (NEI)**
   - Specifically designed for noisy observations
   - Uses the posterior mean's best, not the observed best
   - **Critical for F2** which is confirmed noisy

---

## 5. Key Techniques We Should Consider Adopting

### Priority 1: Input Warping (High Impact, Moderate Effort)
- Both HEBO (winner) and BOTorch recommend this
- Kumaraswamy CDF warping: `w(x) = 1 - (1 - x^a)^b`
- Learn a, b per dimension during GP fitting
- Handles non-stationarity that our current model can't capture
- **Implementation**: Use `scipy.optimize` to jointly optimize kernel hyperparams + warp params

### Priority 2: Output Warping (High Impact, Low Effort)
- Power transform (Yeo-Johnson) on y values before GP fitting
- Handles skewed distributions, heavy tails
- Especially useful for F5 (y ~ 1600) and F1 (y ~ 2.0 but with values near 0)
- **Implementation**: `sklearn.preprocessing.PowerTransformer(method='yeo-johnson')`

### Priority 3: Multi-Objective Acquisition (Medium Impact, Medium Effort)
- Instead of picking PI vs EI vs UCB, compute all three
- Find candidates that are good under multiple criteria
- Select from the Pareto front
- **Implementation**: Compute PI, EI, UCB scores for all candidates, normalize, select top candidates across all three

### Priority 4: Noisy EI for F2 (Medium Impact, Low Effort)
- Use posterior mean's best (not observed best) as the incumbent
- This is robust to noise — doesn't chase noisy high observations
- **Implementation**: Replace `y_best = max(y_observed)` with `y_best = max(gp.predict(X))`

### Priority 5: Knowledge Gradient (High Impact, High Effort)
- Best acquisition function for low-budget settings
- "What is the value of information from evaluating this point?"
- Requires BOTorch or custom implementation
- **Implementation**: Would require switching to BOTorch (PyTorch-based GP)

---

## 6. What We Do Well (Validated by Literature)

- **Multi-kernel ensemble**: Our 3-kernel Matern ensemble with LML+LOO-CV weighting is solid
- **Trust regions**: TuRBO-style local optimization is a proven winner
- **PI for exploitation**: Literature confirms PI is appropriate for tight exploitation
- **Sobol QMC candidates**: Better than random sampling for candidate generation
- **StandardScaler normalization**: Basic but effective

---

## 7. Gap Analysis: Our Implementation vs HEBO

| Component | Our Method | HEBO (Winner) | Gap |
|-----------|-----------|---------------|-----|
| **Surrogate** | Multi-kernel GP ensemble | GP with learned warping | We're close |
| **Input warping** | None | Kumaraswamy CDF | **Major gap** |
| **Output warping** | StandardScaler only | Power/Box-Cox transform | **Major gap** |
| **Noise model** | Homoscedastic (WhiteKernel) | Heteroscedastic | Gap for F2 |
| **Acquisition** | Single (PI/EI/UCB) | Multi-objective ensemble | **Major gap** |
| **Acq. optimization** | Sobol + argmax | NSGA-II evolutionary | Moderate gap |
| **Kernel selection** | 3 fixed Matern + weighting | Auto-selected | We're good |

---

## 8. Practical Recommendations for Remaining Weeks

### Quick Wins (Can implement in Week 11 notebook)
1. **Output warping with Yeo-Johnson**: 3 lines of code, potentially large impact
2. **Noisy EI for F2**: Use `y_best = max(gp.predict(X_train))` instead of `max(y)`
3. **Multi-acquisition selection**: Compute PI, EI, UCB; pick candidate that ranks well across all three

### Medium Effort (Worth trying)
4. **Input warping**: Learn Kumaraswamy parameters per dimension
5. **Adaptive trust region**: Auto-expand on improvement, contract on regression

### Too Complex for 1-2 Weeks
6. Full BOTorch migration (Knowledge Gradient, MC acquisition)
7. Heteroscedastic noise modeling
8. NSGA-II acquisition optimization

---

## References

1. Cowen-Rivers et al. (2022). "HEBO: Heteroscedastic Evolutionary Bayesian Optimisation." JAIR.
2. Eriksson et al. (2019). "Scalable Global Optimization via Local Bayesian Optimization." NeurIPS.
3. Eriksson & Jankowiak (2021). "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces." UAI.
4. Balandat et al. (2020). "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization." NeurIPS.
5. Snoek et al. (2014). "Input Warping for Bayesian Optimization of Non-Stationary Functions." ICML.
6. "Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization." (2024). arXiv:2402.02746.
7. "Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB." ACM TELO, 2024.
8. "Unexpected Improvements to Expected Improvement for Bayesian Optimization." NeurIPS 2023.
