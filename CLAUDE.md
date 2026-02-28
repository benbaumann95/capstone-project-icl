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
| F1 | 2D | **1.993** | [0.6297, 0.6287] | **NEW BEST Week 10** (+12.4%, biggest single-week gain) |
| F2 | 2D | 0.667 | [0.7026, 0.9266] | Stagnant (since Week 4, confirmed noisy) |
| F3 | 3D | -0.0117 | [0.5296, 0.6390, 0.3896] | Best from Week 9 (W10 overshot) |
| F4 | 4D | 0.629 | [0.4234, 0.3779, 0.4125, 0.4247] | Best from Week 8 (W9, W10 regressed) |
| F5 | 4D | **1675.3** | [0.3688, 0.2765, 1.0, 1.0] | **NEW BEST Week 10** (near-optimal) |
| F6 | 5D | -0.586 | [0.6902, 0.1258, 0.7578, 0.7367, 0.0510] | Best from Week 8 (W9, W10 regressed) |
| F7 | 6D | **2.480** | [0.0136, 0.1335, 0.5825, 0.2327, 0.3714, 0.7454] | **NEW BEST Week 10** (3 consecutive improvements) |
| F8 | 8D | **9.937** | [0.0339, 0.0872, 0.1480, 0.0589, 0.8784, 0.3502, 0.1668, 0.2351] | **NEW BEST Week 10** (3 consecutive improvements) |

**Query format**: Hyphen-separated decimals, e.g., `"0.634-0.636"` for 2D

## Key Technologies

- **Gaussian Processes**: `sklearn.gaussian_process.GaussianProcessRegressor` with Matern kernels
- **Output Warping**: Yeo-Johnson PowerTransformer (HEBO-inspired, added Week 11)
- **Acquisition Functions**: Multi-Acquisition Ensemble (PI+EI+UCB), Noisy EI, PI, EI, UCB
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
- **Selection method**: Use multi-acquisition ensemble (PI+EI+UCB) as default. Noisy EI for noisy functions (F2). Never use TS overrides.

## Key Learnings (from 11 weeks)

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

### Week 10 Results & New Learnings
19. **F1 coordinate-wise search unlocked +12.4%**: Moving x1 from 0.622→0.629 gave 1.773→1.993. Peak is near [0.630, 0.629], not [0.631, 0.622].
20. **F2 is definitively noisy**: Three evaluations of exact same point [0.7026, 0.9266] gave 0.611, 0.667, 0.590. Use Noisy EI (GP-predicted incumbent).
21. **F3 overshoots consistently**: +0.019/dim (W8) and +0.015/dim (W10) both regressed from W9 best. Peak very close to W9 position.
22. **F4 is even more peaked than thought**: r=0.010 from W8 best caused 15% regression. Needs r≤0.003.
23. **F6 trajectory hypothesis falsified**: Both full step (W9) and half step (W10) from W8 regressed. Abandon trajectory.
24. **F7 x3-increasing direction confirmed**: 3 consecutive improvements following this direction.
25. **Output warping (Yeo-Johnson) added in W11**: HEBO's #1 technique. Normalizes skewed y-distributions for better GP fitting.

### Function-Specific
26. **F1 is extremely peaked**: Peak near [0.630, 0.629]. Radius must be ≤ 0.005. Even 0.003 shift = 10.7% regression.
27. **F2 is noisy**: Same point gave 0.611, 0.667, 0.590. Use Noisy EI with GP-predicted incumbent.
28. **F5 has extreme x2/x3 sensitivity**: x2/x3 should be pinned at 1.0. x0/x1 optimum near (~0.369, 0.277).

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

- **Week 11 submitted** (Module 22 notebook) — HEBO-inspired upgrades
- **Week 10 Results Summary** (4/8 new bests):
  - F1: **NEW BEST 1.993** (improved from 1.773, +12.4%! Coordinate-wise search breakthrough)
  - F5: **NEW BEST 1675.3** (improved from 1674.2, near-optimal)
  - F7: **NEW BEST 2.480** (improved from 2.448, 3rd consecutive improvement)
  - F8: **NEW BEST 9.937** (improved from 9.933, 3rd consecutive improvement)
  - F2: Regression to 0.590 (EXACT_RETURN, confirms noise — 3 evals of same point: 0.611, 0.667, 0.590)
  - F3: Regression to -0.027 (trajectory overshot +0.015/dim from W9 best)
  - F4: Regression to 0.536 (r=0.010 still too large for this peaked function)
  - F6: Regression to -0.628 (half-step trajectory also failed — hypothesis falsified)
- **Week 11 Strategy** (HEBO-inspired):
  - **3 new techniques**: Output Warping (Yeo-Johnson), Noisy EI (F2), Multi-Acquisition Ensemble (PI+EI+UCB)
  - F1: Ultra-tight exploit [0.6295, 0.6295], r=0.003, Multi-acquisition
  - F2: Noisy EI + x1<0.920 constraint, r=0.015 (testing untested low-x1 region)
  - F3: GP-guided around W9 best, r=0.005, Multi-acquisition
  - F4: Ultra-tight around W8 best, r=0.003, Multi(0.6PI+0.2EI+0.2UCB)
  - F5: Pin x2/x3 near 1.0, fine-tune x0/x1, r=0.004, Multi-acquisition
  - F6: Abandon trajectory, GP-guided perpendicular search, r=0.005, Multi-acquisition
  - F7: Directional exploit (x3-increasing), r=0.010, Multi-acquisition
  - F8: Ultra-tight + x0<0.035 constraint, r=0.006, Multi-acquisition
- **1 exploration week remaining** (Week 12), Week 13 = EXACT_RETURN
- **Final week strategy (Week 13)**: EXACT_RETURN — submit best known points to lock in results
- **Reference**: See `docs/frontier_bbo_methods.md` for research on competition-winning methods

---

*Last updated: Week 11 (Module 22)*
