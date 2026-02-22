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
| F1 | 2D | 1.773 | [0.6307, 0.6218] | Best from Week 8 (W9 regressed) |
| F2 | 2D | 0.667 | [0.7026, 0.9266] | Stagnant (since Week 4) |
| F3 | 3D | **-0.0117** | [0.5296, 0.6390, 0.3896] | **NEW BEST Week 9** |
| F4 | 4D | 0.629 | [0.4234, 0.3779, 0.4125, 0.4247] | Best from Week 8 (W9 regressed) |
| F5 | 4D | **1674.2** | [0.3679, 0.2760, 0.9999, 0.9999] | **NEW BEST Week 9** (boundary fix!) |
| F6 | 5D | -0.586 | [0.6902, 0.1258, 0.7578, 0.7367, 0.0510] | Best from Week 8 (W9 regressed) |
| F7 | 6D | **2.448** | [0.0141, 0.1315, 0.5804, 0.2172, 0.3712, 0.7476] | **NEW BEST Week 9** |
| F8 | 8D | **9.933** | [0.0326, 0.0870, 0.1488, 0.0585, 0.8808, 0.3429, 0.1663, 0.2273] | **NEW BEST Week 9** |

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

## Key Learnings (from 10 weeks)

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

### Week 9 Results & New Learnings
14. **F1 gradient is ~67/unit in x1**: Shifting x1 by just 0.003 (0.6218→0.6190) caused 10.7% regression (1.773→1.583). Coordinate-wise line search needed.
15. **F5 boundary fix was project's biggest win**: Pushing x2/x3 from 0.996/0.998 to 0.9999/0.9999 gained +55.7 points (+3.4%).
16. **F4 directional EI (r=0.03) was disastrous**: Shifted x2 from 0.413→0.386, causing 37% regression. Never use EI with large radius on peaked functions.
17. **F6 trajectory overshoot**: W9 took 2-3x the successful W7→W8 step size and regressed. Half-steps are safer.
18. **F7 x0=0.014 > x0=0.010**: Small but real improvement, contradicting x0→0 hypothesis.

### Function-Specific
19. **F1 is extremely peaked**: Values drop from 1.77 to 0.85 with 0.025 shift. Radius must be < 0.01. Even 0.003 shift = 10.7% regression.
20. **F2 may be noisy**: Same point gave 0.611 (initial) and 0.667 (W4). 6 consecutive exploration failures. EXACT_RETURN only.
21. **F5 has extreme x2/x3 sensitivity**: x2/x3 should be pinned at 1.0. x0/x1 optimum slightly above W1 values (~0.368, 0.276).

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

- **Week 10 in progress** (Module 21 notebook)
- **Week 9 Results Summary** (4/8 new bests):
  - F3: **NEW BEST -0.0117** (improved from -0.0145, trajectory continues)
  - F5: **NEW BEST 1674.2** (improved from 1618.5, +55.7! Boundary fix was critical)
  - F7: **NEW BEST 2.448** (improved from 2.433, x0=0.014 > x0=0.010)
  - F8: **NEW BEST 9.933** (improved from 9.928, tight exploitation works)
  - F1: Regression to 1.583 (x1 shifted 0.003, lost 10.7% — peak is insanely narrow)
  - F2: Regression to 0.510 (6th consecutive failed exploration)
  - F4: Regression to 0.398 (directional EI r=0.03 was catastrophic)
  - F6: Regression to -0.668 (trajectory overshoot — step was 2-3x too large)
- **Week 10 Strategy**:
  - F1: Coordinate-wise line search (hold x0, nudge x1 +0.002), r=0.005, PI
  - F2: Exploit lower x1 direction (x1<0.927 untested), r=0.015, PI
  - F3: Continue trajectory (+0.007 step), r=0.008, PI
  - F4: Tight recovery around W8 best, r=0.010, PI
  - F5: Push x2/x3 to exactly 1.0, fine-tune x0/x1, r=0.006, PI
  - F6: Half-step trajectory from W8, r=0.010, PI
  - F7: Directional exploit following W8→W9 direction, r=0.015, PI
  - F8: Ultra-tight exploit, r=0.008, PI, constrain x0<0.035
- **3 weeks remaining** (Weeks 10-12), Week 13 = EXACT_RETURN
- **Final week strategy (Week 13)**: EXACT_RETURN — submit best known points to lock in results

---

*Last updated: Week 10 (Module 21)*
