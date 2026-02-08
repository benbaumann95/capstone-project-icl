# CLAUDE.md - Project Context for AI Assistants

> **IMPORTANT**: Keep this file updated as the project evolves. After making significant changes or learning new patterns, update the relevant sections.

## Project Overview

This is an **Imperial College Black-Box Optimization (BBO) Capstone Challenge** repository. The goal is to maximize 8 unknown black-box functions with an extreme query budget (1 submission per week per function). We have NO access to function structure, derivatives, or gradients—only observed input-output pairs.

**Real-world relevance**: Mirrors hyperparameter tuning, drug discovery, A/B testing, and AutoML scenarios.

## Repository Structure

```
capstone-project-icl/
├── data/function_{1-8}/       # Observed samples for each function
│   ├── samples.csv            # All (x, y) pairs with source labels
│   ├── initial_inputs.npy     # Original random samples
│   └── initial_outputs.npy    # Original outputs
├── notebooks/                 # Weekly analysis and query generation
│   └── 0X_Module_YY.ipynb     # One notebook per week (Module 12-18)
├── src/
│   ├── utils.py               # load_data(), save_submission()
│   └── initialize_samples.py  # Reset data from .npy files
├── docs/
│   ├── methodology.md         # BO theory, algorithms, citations
│   └── week7_trust_regions.png
├── submissions/
│   └── submission_log.csv     # All queries with timestamps
└── requirements.txt           # numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, torch
```

## Function Specifications

| Function | Dim | Best Value | Best Location | Status |
|----------|-----|------------|---------------|--------|
| F1 | 2D | 1.626 | [0.6346, 0.6356] | Stable (Week 5) |
| F2 | 2D | 0.667 | [0.7026, 0.9266] | Stagnant |
| F3 | 3D | **-0.0145** | [0.5198, 0.6294, 0.3797] | **NEW BEST Week 7** |
| F4 | 4D | 0.600 | [0.4046, 0.4148, 0.3574, 0.3990] | Protected |
| F5 | 4D | 1618.5 | [0.3627, 0.2734, 0.9961, 0.9975] | Protected |
| F6 | 5D | **-0.681** | [0.7084, 0.1454, 0.7533, 0.7305, 0.0527] | **NEW BEST Week 7** |
| F7 | 6D | 2.403 | [0.0100, 0.1564, 0.5383, 0.2527, 0.3992, 0.7464] | Improving |
| F8 | 8D | 9.915 | [0.0251, 0.0950, 0.1630, 0.0358, 0.8874, 0.3193, 0.1665, 0.2045] | Fine-tuning |

**Query format**: Hyphen-separated decimals, e.g., `"0.634-0.636"` for 2D

## Key Technologies

- **Gaussian Processes**: `sklearn.gaussian_process.GaussianProcessRegressor` with Matern kernels
- **Neural Networks**: PyTorch ensembles (7 models) with LayerNorm, ReLU, Dropout
- **Acquisition Functions**: UCB, Expected Improvement, Thompson Sampling
- **Advanced BO**: TuRBO (trust regions), Multi-Kernel GP Ensemble, Sobol sequences

## Code Patterns

### Loading Data
```python
from src.utils import load_data
df = load_data(func_id)  # Returns DataFrame
X = df.iloc[:, :-2].values  # All columns except y and source
y = df['y'].values
```

### Training GP
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X, y)
mu, sigma = gp.predict(X_test, return_std=True)
```

### UCB Acquisition
```python
def ucb(mu, sigma, kappa=1.96):
    return mu + kappa * sigma
```

### Saving Submissions
```python
from src.utils import save_submission
query_str = '-'.join(map(str, x_next))
save_submission(func_id, query_str, module_name="Module XX")
```

## Important Conventions

- **Domain**: All inputs in [0, 1]^d; use [0.01, 0.99] to avoid edge issues
- **Random seeds**: Always set `np.random.seed(42)`, `torch.manual_seed(42)` for reproducibility
- **Column naming**: x0, x1, ..., xD for inputs; y for output; source for tracking
- **Trust regions**: 0.02-0.1 radius depending on solution quality

## Key Learnings (from 8 weeks)

1. **Kernel smoothness matters**: F1-F6 are rough (Matern ν=0.5 best), F7-F8 are smooth (Matern ν=2.5 best)
2. **Perturbation scales inversely with quality**: Good solutions need fine-tuning (0.02), poor solutions need broader search (0.10)
3. **Over-exploration early damages later consolidation**: Trust regions and careful step sizes prevent catastrophic failures
4. **Thompson Sampling > UCB for high-dimensional spaces**: Posterior sampling naturally balances exploration
5. **Neural networks enable gradients but are unreliable with few data points**: Best combined with GP uncertainty (hybrid approach)
6. **Multi-kernel GP ensembles work well**: Weighting by log marginal likelihood automatically selects appropriate smoothness
7. **17 data points is enough for reliable GP fits**: Ensemble weights become stable, predictions more accurate
8. **Week 7 breakthroughs (F3, F6)**: Persistent exploration in promising regions eventually pays off

## Strategy Selection Framework

```
Is this the FINAL week?
├─ YES → EXACT_RETURN (return to best known point)
└─ NO → Explore new regions:
    ├─ Found 2x+ improvement? → TURBO_EXPLOIT (small trust region)
    ├─ Near boundary optimum? → BOUNDARY_REFINE
    ├─ Consistent improvement? → TURBO_EXPLOIT
    ├─ Stagnant? → KERNEL_ENSEMBLE (try different smoothness)
    └─ Unknown? → TURBO_EXPLORE (moderate trust region)
```

## Common Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reset data from .npy files
python src/initialize_samples.py

# Run notebooks
jupyter notebook notebooks/

# After receiving results, update data/function_X/samples.csv manually
# Add row with source='week_N_submission'
```

## Git Workflow

- **Main branch**: `main`
- **Weekly branches**: `week-N` for each week's work
- **Commit pattern**: `"Week N: [description of strategy and results]"`

## Academic References

Key papers implemented/cited:
- Eriksson et al. (2019): TuRBO algorithm
- Srinivas et al. (2010): GP-UCB theory
- Lakshminarayanan et al. (2017): Deep ensembles
- Rasmussen & Williams (2006): GP theory

See [docs/methodology.md](docs/methodology.md) for full citations and algorithmic details.

---

## Notes for Future Sessions

- **Week 8 submitted** (Module 19 notebook)
- **Week 7 Results Summary**:
  - F3: NEW BEST -0.0145 (improved from -0.035)
  - F6: NEW BEST -0.681 (improved from -0.714)
  - Other functions: Minor regression or stable
- **Week 8 Strategy**:
  - Multi-kernel GP ensemble (weights: ν=0.5, 1.5, 2.5 by log marginal likelihood)
  - TuRBO-style trust regions with function-specific radii
  - Exploit strategy for F3, F6 (recent improvements)
  - F4 has high predicted improvement potential (1.24 vs 0.60 current)
- **Final weeks strategy**: Consider EXACT_RETURN in final week to lock in best known points

---

*Last updated: Week 8 (Module 19)*
