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
| F1 | 2D | **1.773** | [0.6307, 0.6218] | **NEW BEST Week 8** |
| F2 | 2D | 0.667 | [0.7026, 0.9266] | Stagnant (since Week 4) |
| F3 | 3D | -0.0145 | [0.5198, 0.6294, 0.3797] | Best from Week 7 |
| F4 | 4D | **0.629** | [0.4234, 0.3779, 0.4125, 0.4247] | **NEW BEST Week 8** |
| F5 | 4D | 1618.5 | [0.3627, 0.2734, 0.9961, 0.9975] | Protected (boundary optimum) |
| F6 | 5D | **-0.586** | [0.6902, 0.1258, 0.7578, 0.7367, 0.0510] | **NEW BEST Week 8** (2 consecutive) |
| F7 | 6D | **2.433** | [0.0100, 0.1076, 0.5812, 0.2060, 0.3653, 0.7405] | **NEW BEST Week 8** |
| F8 | 8D | **9.928** | [0.0259, 0.0952, 0.1534, 0.0495, 0.8710, 0.3337, 0.1697, 0.2230] | **NEW BEST Week 8** |

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

- **Domain**: All inputs in [0, 1]^d. Clip to [0, 1] (NOT [0.01, 0.99] — the old margin was a bug)
- **Random seeds**: Always set `np.random.seed(42)`, `torch.manual_seed(42)` for reproducibility
- **Column naming**: x0, x1, ..., xD for inputs; y for output; source for tracking
- **Trust regions**: 0.008-0.03 radius depending on peak width (see per-function analysis)
- **Selection method**: Use PI for exploitation, EI for directional, UCB for exploration. Never use TS overrides.

## Key Learnings (from 9 weeks)

### Technical
1. **Kernel smoothness matters**: F1-F6 are rough (Matern ν=0.5 best), F7-F8 are smooth (Matern ν=2.5 best)
2. **Multi-kernel GP ensembles work well**: Weighting by LML (60%) + LOO-CV (40%) automatically selects appropriate smoothness
3. **Thompson Sampling needs diagonal approx**: Full covariance on 20K candidates creates 20K×20K matrix (3.2GB OOM) — use `mu + sigma * randn` instead
4. **18+ data points is enough for reliable GP fits**: Ensemble weights become stable, predictions more accurate

### Strategic
5. **Trust region radius must match peak width**: F1's peak is ~0.02 wide, so r=0.008. F8 is flat, so r=0.010 is fine. Using r=0.025 on F1 caused certain regression.
6. **PI > EI for exploitation**: EI rewards uncertainty (selects points FURTHER from best). PI asks "probability of beating current best" — correct for peaked functions.
7. **Never use TS override for exploitation**: The threshold `0.01 * |y_best|` is scale-dependent and caused F1, F3 to select exploratory points during exploitation.
8. **Improvement trajectories are real but fragile**: F6 showed clear directional improvement (W7→W8), but EI can select points that REVERSE the trajectory (the x1 reversal bug).
9. **Center overrides are essential**: When the GP's predicted best ≠ the strategic target (e.g., F6 trajectory, F7 boundary test), override the trust region center.

### Bugs Found & Fixed (Week 9)
10. **Domain clipping [0.01, 0.99] was a critical bug**: Prevented F5's x2/x3 from reaching >0.99 (optimum at 0.996/0.998). Destroyed 7 of 8 F5 submissions. Fixed to [0, 1].
11. **TS override with EI caused secret exploration**: For F1, TS selected x1=0.592 (0.030 from peak) during "exploitation". Removed entirely.
12. **F5 BoundaryAwareTrustRegion pinning was overridden by clip**: Pinning set x2/x3 to (0.993, 0.999), then clip capped at 0.99. Now pin AFTER clip.
13. **F7's x0 was stuck at 0.01 floor**: All top results had x0=0.010 (the clip floor). Now testing x0 < 0.01.

### Function-Specific
14. **F1 is extremely peaked**: Values drop from 1.77 to 0.85 with 0.025 shift. Radius must be < 0.01.
15. **F2 may be noisy**: Same point gave 0.611 (initial) and 0.667 (W4). Every exploration attempt has failed.
16. **F5 has extreme x2/x3 sensitivity**: Dropping x2 from 0.996 to 0.990 costs ~100 points. x0/x1 sensitivity is 10x lower.

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

- **Week 9 submitted** (Module 20 notebook)
- **Week 8 Results Summary** (best week — 5/8 new bests):
  - F1: **NEW BEST 1.773** (improved from 1.626, +9.0%)
  - F4: **NEW BEST 0.629** (improved from 0.600, +4.8%)
  - F6: **NEW BEST -0.586** (improved from -0.681, +13.9%, 2nd consecutive improvement)
  - F7: **NEW BEST 2.433** (improved from 2.403, +1.2%)
  - F8: **NEW BEST 9.928** (improved from 9.915, +0.1%)
  - F2: Regression to 0.584 (stagnant since Week 4)
  - F3: Slight regression to -0.017 (best remains -0.0145 from Week 7)
  - F5: Regression to 1415.4 (boundary clipping bug pulled x2/x3 away from 1.0)
- **Week 9 Major Fixes**:
  - Fixed domain clipping bug: [0.01, 0.99] → [0, 1] (critical for F5, F7, F8)
  - Removed TS override (caused secret exploration during exploitation)
  - Added `selection` parameter to `optimize_function` (PI, EI, UCB, Mean, EI+UCB)
  - Added `center_override` parameter (for when GP best ≠ strategic target)
  - Fixed BoundaryAwareTrustRegion: pinning now applied AFTER clipping
  - Full deep analysis documented in `docs/week9_deep_analysis.md`
- **Week 9 Corrected Queries**:
  - F1: `0.630748-0.618965` (PI, r=0.008 — matches peak width ~0.02)
  - F2: `0.696774-0.941425` (PI, r=0.015 — tight exploit, untested direction)
  - F3: `0.529607-0.638966-0.389611` (PI, r=0.010)
  - F4: `0.422458-0.396694-0.385944-0.406530` (EI, directional, r=0.03)
  - F5: `0.367932-0.276045-0.999957-0.999960` (PI, boundary, x2/x3 near 1.0!)
  - F6: `0.652850-0.126519-0.771238-0.757384-0.031689` (EI, trajectory center)
  - F7: `0.014132-0.131467-0.580441-0.217157-0.371219-0.747590` (PI, x0 near 0)
  - F8: `0.032640-0.087004-0.148791-0.058486-0.880818-0.342923-0.166261-0.227331` (PI, r=0.010)
- **4 weeks remaining** (Weeks 10-13)
- **Final week strategy (Week 13)**: EXACT_RETURN — submit best known points to lock in results

---

*Last updated: Week 9 (Module 20)*
