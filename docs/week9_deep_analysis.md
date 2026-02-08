# Week 9 Deep Analysis: Function-by-Function Audit

> Generated 2026-02-08. Critical review of all 8 functions, every submission,
> identified bugs, and corrected strategies for remaining 5 weeks (9-13).

---

## Table of Contents
1. [Identified Bugs & Systemic Issues](#bugs)
2. [F1 Analysis](#f1)
3. [F2 Analysis](#f2)
4. [F3 Analysis](#f3)
5. [F4 Analysis](#f4)
6. [F5 Analysis](#f5)
7. [F6 Analysis](#f6)
8. [F7 Analysis](#f7)
9. [F8 Analysis](#f8)
10. [Corrected Week 9 Strategies](#corrected)
11. [Remaining Weeks Roadmap](#roadmap)

---

## <a name="bugs"></a>1. Identified Bugs & Systemic Issues

### BUG 1: Domain Clipping [FIXED]
- **What**: `np.clip(candidates, 0.01, 0.99)` in both `DirectionalTrustRegion` and `BoundaryAwareTrustRegion`
- **Impact**: Restricted search to 98% of [0,1] domain. Prevented F5's x2/x3 from reaching >0.99 (optimum at 0.996/0.998). Prevented F7's x0 from going below 0.01.
- **Fix**: Changed to `np.clip(candidates, 0.0, 1.0)`. For `BoundaryAwareTrustRegion`, pinning now happens AFTER clipping.

### BUG 2: TS Override Threshold Too Sensitive
- **What**: `if mu[ts_idx] > mu[ei_idx] + 0.01 * abs(y_best)` — the 0.01 * |y_best| threshold
- **Impact**: For F3 (y_best = -0.0145), threshold = 0.000145 — any trivial difference triggers TS override. For F1 (y_best = 1.773), threshold = 0.018 — also too easy to trigger.
- **Result**: TS overrides EI for F1 and F3, selecting EXPLORATORY points predicted BELOW the current best. For F1, TS selected a point at x1=0.592 (0.030 below the best at 0.622) — a massive shift for this peaked function.
- **Fix needed**: Use a fixed threshold or disable TS override for exploit strategies. When exploiting, we want the most conservative/highest-mean prediction, not an exploratory sample.

### BUG 3: F5 Pinned Range Too Narrow
- **What**: `pinned_range=(0.993, 0.999)` but best known x2=0.9961, x3=0.9975
- **Impact**: The pinned range cannot reach the actual best values. x2=0.996 is ABOVE 0.993 but the uniform sampling over (0.993, 0.999) centers around 0.996, so coverage is OK but not optimal.
- **Fix needed**: Widen to (0.994, 1.0) to allow exploration above the known best.

### BUG 4: GP Predictions All Below Current Best
- **What**: For 6 out of 8 functions, the GP's predicted value at the selected point is BELOW the current best (negative delta)
- **Functions affected**: F1 (-0.075), F2 (-0.060), F3 (-0.002), F5 (-22.2), F6 (-0.047), F7 (-0.021)
- **Impact**: We are systematically choosing points the GP believes will be WORSE than what we already have. This is exploration, not exploitation.
- **Root cause**: EI and UCB reward uncertainty, not predicted performance. In tight trust regions, the GP often has low uncertainty near the best point (it's a training point) and higher uncertainty further away.
- **Fix needed**: For exploitation strategies, consider using pure GP mean maximization or PI (Probability of Improvement) instead of EI.

### BUG 5: Submission Log Accumulating Duplicates
- **What**: The notebook has been run 3+ times, creating duplicate entries in submission_log.csv (lines 286-309 are 3 copies of Week 9 queries)
- **Impact**: No functional impact but the log is messy. Need to clean up before the final submission.

---

## <a name="f1"></a>2. F1 (2D) — Extremely Peaked Function

### Data Summary
| Week | x0 | x1 | y | Status |
|------|------|------|------|--------|
| init best | 0.650 | 0.682 | -0.004 | |
| W1 | 0.787 | 0.491 | 4.4e-31 | Terrible |
| W2 | 0.369 | 0.056 | 5.5e-96 | Terrible |
| W3 | 0.150 | 0.750 | -7.1e-128 | Terrible |
| W4 | 0.650 | 0.650 | 0.196 | First signal |
| W5 | 0.635 | 0.636 | **1.626** | Breakthrough |
| W6 | 0.624 | 0.619 | 1.503 | Regression (moved 0.017 away) |
| W7 | 0.610 | 0.627 | 0.854 | Worse regression (moved 0.025 away) |
| W8 | 0.631 | 0.622 | **1.773** | NEW BEST (corrected course) |

### Key Observations
1. **Incredibly peaked**: Values go from 1.773 to ~0 within a distance of 0.03
2. **W1-W3 were blind**: The function is essentially zero everywhere except a tiny peak near (0.63, 0.62)
3. **Sensitivity**: Moving 0.011 from W5 to W6 dropped from 1.626 to 1.503. Moving 0.025 from W5 to W7 dropped to 0.854.
4. **W8 success**: Found the true peak by returning near (0.63, 0.62)

### Current Week 9 Query: (0.632, 0.592) — PROBLEMATIC
- x1 shifted from 0.622 to 0.592 — a **0.030 shift**, larger than the W6/W7 regression shifts
- GP predicted 1.698 ± 0.162, which is **0.075 below** the current best
- TS override selected this exploratory point over EI
- **This query will almost certainly regress**

### Corrected Strategy
- **Radius: 0.008** (not 0.025). The function's peak width is ~0.02, so radius must be < 0.01
- **Use PI or pure mean**: Don't use EI or TS which favor uncertainty
- **Disable TS override**: No reason to explore on this function
- **Center tightly on W8**: (0.631, 0.622) is the best point

---

## <a name="f2"></a>3. F2 (2D) — Possibly Noisy, Very Sensitive

### Data Summary
| Week | x0 | x1 | y | Status |
|------|------|------|------|--------|
| init | 0.7026 | 0.9266 | 0.611 | |
| W1 | 0.718 | 0.003 | 0.465 | Low x1 → decent |
| W2 | 0.956 | 0.990 | 0.006 | Bad |
| W3 | 0.739 | 0.909 | 0.345 | |
| W4 | 0.7026 | 0.9266 | **0.667** | EXACT RETURN → higher than init! |
| W5 | 0.683 | 0.954 | 0.583 | Regression |
| W6 | 0.647 | 0.945 | 0.390 | Worse |
| W7 | 0.703 | 0.955 | 0.577 | Slightly different from W4 |
| W8 | 0.693 | 0.974 | 0.584 | Still below W4 |

### Key Observations
1. **POSSIBLY NOISY**: The initial sample at (0.7026, 0.9266) gave y=0.611. W4 submitted the same point (rounded to 6dp) and got y=0.667. Same point, different values → noise.
2. **Every perturbation from the best is worse**: W5-W8 all tried nearby points and all regressed
3. **The function is very peaked near (0.703, 0.927)** with a possible noise floor
4. **Exploration has never found anything better**: W1 tried low x1, W2 tried high x0 — both much worse

### Current Week 9 Query: (0.697, 0.811) — VERY PROBLEMATIC
- x1 shifted from 0.927 to 0.811 — a **0.116 shift**
- GP predicted 0.607 ± 0.047, which is **0.060 below** the current best
- This is exploration, not exploitation
- **Every past exploration of F2 has been worse than the best**

### Corrected Strategy
If noisy: **Return to exact best point** (0.702637, 0.926564). Multiple evaluations of a noisy function at the best point can yield a higher value. The expected value is 0.639 (average of 0.611 and 0.667) with a chance of exceeding 0.667.

If not noisy: **Very tight exploitation** (radius 0.005) around (0.703, 0.927) using PI. The tiny difference between initial and W4 values could be from 6dp rounding of coordinates.

**Recommended**: Return to exact best point. Risk is zero (we already know this point gives 0.611-0.667).

---

## <a name="f3"></a>4. F3 (3D) — Steady Improvement Trajectory

### Data Summary
| Week | x0 | x1 | x2 | y | Status |
|------|------|------|------|------|--------|
| init best | 0.493 | 0.612 | 0.340 | -0.035 | |
| W1 | 0.899 | 0.004 | 0.000 | -0.194 | Bad |
| W2 | 0.010 | 0.010 | 0.306 | -0.133 | Bad |
| W3 | 0.496 | 0.728 | 0.091 | -0.041 | |
| W4 | 0.470 | 0.625 | 0.297 | -0.085 | Regression |
| W5 | 0.493 | 0.612 | 0.340 | -0.040 | Near init return |
| W6 | 0.501 | 0.607 | 0.372 | -0.031 | Improvement |
| W7 | 0.520 | 0.629 | 0.380 | **-0.0145** | NEW BEST |
| W8 | 0.539 | 0.648 | 0.400 | -0.0167 | Slight regression |

### Key Observations
1. **Clear improvement trajectory W5→W7**: Moving in direction (+0.027, +0.017, +0.040) per step
2. **W8 overshot**: Continued the same direction but went too far (+0.019, +0.019, +0.020)
3. **The optimum is near (0.520, 0.629, 0.380)** — very well localized now
4. **W8 was only 0.002 worse than W7** — we're in the right neighborhood

### Current Week 9 Query: (0.528, 0.631, 0.392) — Acceptable
- Close to the W7 best (0.520, 0.629, 0.380), shifted (+0.008, +0.002, +0.012)
- GP predicted -0.017 ± 0.012
- The shift in x2 (+0.012) is concerning — W8 showed that increasing x2 beyond 0.380 hurts

### Corrected Strategy
- **Radius: 0.010** (not 0.015). We have 6 data points within 0.03 of the optimum
- **Center on W7 best**: (0.520, 0.629, 0.380), NOT W8 (which regressed)
- **Use PI**: Maximize probability of beating -0.0145
- **Disable TS override**: Function is well-localized, no exploration needed

---

## <a name="f4"></a>5. F4 (4D) — Recent Breakthrough, High Uncertainty

### Data Summary
| Week | x0 | x1 | x2 | x3 | y | Status |
|------|------|------|------|------|------|--------|
| W1 | 0.405 | 0.415 | 0.357 | 0.399 | 0.600 | Good start |
| W2 | 0.437 | 0.419 | 0.268 | 0.448 | -1.326 | Bad |
| W3 | 0.391 | 0.464 | 0.314 | 0.434 | -1.265 | Bad |
| W4 | 0.750 | 0.750 | 0.750 | 0.750 | -24.5 | Terrible |
| W5 | 0.405 | 0.415 | 0.357 | 0.399 | 0.600 | EXACT_RETURN (confirmed deterministic) |
| W6 | 0.423 | 0.443 | 0.367 | 0.382 | 0.287 | Regression |
| W7 | 0.394 | 0.410 | 0.348 | 0.388 | 0.364 | Better than W6 |
| W8 | 0.423 | 0.378 | 0.413 | 0.425 | **0.629** | NEW BEST |

### Key Observations
1. **W5 EXACT_RETURN confirmed deterministic**: Same point → same 0.600
2. **W8 breakthrough at a different location**: x2 jumped from 0.357 to 0.413 (+0.056)
3. **The positive region is very narrow**: W1 and W8 (both ~0.6) are only 0.07 apart, but W2/W3/W6 (all within similar range) gave negative values
4. **Very high GP uncertainty**: Predicted 0.803 ± 0.340 — nearly 50% relative uncertainty

### Current Week 9 Query: (0.422, 0.397, 0.386, 0.407) — Reasonable
- Between W1 and W8 in most dimensions
- High uncertainty means moderate risk, but with 4 weeks left this is OK
- EI selection is reasonable here

### Assessment
Current strategy is acceptable. The function is poorly understood (only 2 positive evaluations out of 8 submissions) and needs exploration within the positive region. Keep directional exploitation.

---

## <a name="f5"></a>6. F5 (4D) — Boundary Optimum, Clipping Bug Was Fatal

### Data Summary
| Week | x0 | x1 | x2 | x3 | y | Status |
|------|------|------|------|------|------|--------|
| init best | 0.224 | 0.846 | 0.879 | 0.879 | 1089 | |
| **W1** | **0.363** | **0.273** | **0.996** | **0.998** | **1618.5** | **ALL-TIME BEST** |
| W2 | 0.389 | 0.274 | 0.990 | 0.981 | 1454.7 | x2/x3 clipped → -164 |
| W3 | 0.376 | 0.240 | 0.990 | 0.990 | 1509.0 | x2/x3 clipped → -110 |
| W4 | 0.308 | 0.213 | 0.990 | 0.963 | 1289.3 | x2/x3 clipped → -329 |
| W5 | 0.363 | 0.273 | 0.996 | 0.998 | 1618.5 | EXACT_RETURN (confirmed deterministic) |
| W6 | 0.373 | 0.283 | 0.990 | 0.990 | 1514.8 | x2/x3 clipped → -104 |
| W7 | 0.363 | 0.273 | 0.987 | 0.990 | 1487.0 | x2/x3 clipped → -132 |
| W8 | 0.381 | 0.293 | 0.976 | 0.990 | 1415.4 | x2/x3 clipped → -203 |

### Key Observations
1. **Clipping bug destroyed 7 of 8 submissions**: Weeks 2-4, 6-8 all had x2/x3 capped at 0.990 or lower
2. **W5 EXACT_RETURN confirmed deterministic**: Same coordinates → same 1618.5
3. **Extreme sensitivity to x2/x3 near 1.0**:
   - W1 (x2=0.996, x3=0.998) → 1618.5
   - W6 (x2=0.990, x3=0.990) → 1514.8 (6 thousandths lower in x2 = -104 in y)
   - W4 (x2=0.990, x3=0.963) → 1289.3 (even lower x3 = even worse)
4. **x0/x1 sensitivity is moderate**:
   - W3 (x0=0.376, x1=0.240, x2/x3=0.99/0.99) → 1509
   - W6 (x0=0.373, x1=0.283, x2/x3=0.99/0.99) → 1515
   - Difference of 6 points for significant x0/x1 shift

### Current Week 9 Query: (0.385, 0.295, 0.999, 0.999) — STILL PROBLEMATIC
- x2/x3 are now properly at 0.999 (the clipping fix worked)
- BUT x0=0.385 vs best x0=0.363 (shift of +0.022) and x1=0.295 vs best x1=0.273 (shift of +0.022)
- These shifts are larger than the W3↔W6 comparison, suggesting ~6-10 point cost in x0/x1 misplacement
- The x2/x3 increase from 0.996→0.999 might gain more than it costs, but it's uncertain

### Corrected Strategy
- **Pin x0, x1 near the W1 values**: x0 ∈ (0.355, 0.370), x1 ∈ (0.265, 0.280)
- **Explore x2, x3 near 1.0**: x2 ∈ (0.995, 1.0), x3 ∈ (0.996, 1.0)
- **The ideal query**: Something like (0.363, 0.273, 0.998, 0.999) — keep x0/x1 at the known best and push x2/x3 slightly higher
- **Alternative**: Just return to exact W1 point (0.362718, 0.273413, 0.996088, 0.997538) which guarantees 1618.5
- **DO NOT vary x0/x1 more than ±0.008 from the W1 values**

---

## <a name="f6"></a>7. F6 (5D) — Strong Momentum, Direction Matters

### Data Summary
| Week | x0 | x1 | x2 | x3 | x4 | y | Status |
|------|------|------|------|------|------|------|--------|
| init best | 0.728 | 0.155 | 0.733 | 0.694 | 0.056 | -0.714 | |
| W1 | 0.368 | 0.008 | 0.877 | 0.995 | 0.022 | -0.959 | Bad |
| W2 | 0.365 | 0.338 | 0.052 | 0.990 | 0.010 | -1.057 | Bad |
| W3 | 0.642 | 0.017 | 0.711 | 0.682 | 0.066 | -0.769 | |
| W4 | 0.728 | 0.155 | 0.733 | 0.694 | 0.056 | -0.717 | Near init |
| W5 | 0.721 | 0.142 | 0.724 | 0.702 | 0.055 | -0.735 | Regression |
| W6 | 0.733 | 0.152 | 0.732 | 0.692 | 0.060 | -0.803 | Worse |
| **W7** | 0.708 | 0.145 | 0.753 | 0.731 | 0.053 | **-0.681** | **NEW BEST** |
| **W8** | 0.690 | 0.126 | 0.758 | 0.737 | 0.051 | **-0.586** | **NEW BEST** |

### Improvement Trajectory (W4 → W7 → W8)
| Dim | W4 | W7 | W8 | Direction |
|-----|------|------|------|-----------|
| x0 | 0.728 | 0.708 | 0.690 | **Decreasing** (-0.019/step) |
| x1 | 0.155 | 0.145 | 0.126 | **Decreasing** (-0.015/step) |
| x2 | 0.733 | 0.753 | 0.758 | **Increasing** (+0.012/step) |
| x3 | 0.694 | 0.731 | 0.737 | **Increasing** (+0.022/step) |
| x4 | 0.056 | 0.053 | 0.051 | **Decreasing** (-0.003/step) |

### Current Week 9 Query: (0.662, 0.155, 0.770, 0.758, 0.025) — DIRECTION ERROR
- x0=0.662: Continues the decreasing trend (good, -0.028 from W8)
- **x1=0.155: REVERSED from 0.126 to 0.155** — this goes AGAINST the improvement direction!
  - x1 should be ~0.107-0.115 (continuing the decrease), not 0.155 (W4's value)
- x2=0.770: Continues increasing (good)
- x3=0.758: Continues increasing (good)
- x4=0.025: Continues decreasing (good)
- **The x1 reversal is the critical problem** — the EI selected a point that reverses the strongest trend

### Corrected Strategy
- **Follow the trajectory strictly**: x0≈0.670, x1≈0.107, x2≈0.763, x3≈0.743, x4≈0.048
- **Radius: 0.020** (tight around the trajectory prediction)
- **Use PI or pure mean**: EI's uncertainty-seeking behavior caused the x1 reversal
- **Validate that the selected point follows all 5 dimensional trends**

---

## <a name="f7"></a>8. F7 (6D) — x0 Should Be Near Zero

### Data Summary
| Week | x0 | x1 | x2 | x3 | x4 | x5 | y | Status |
|------|------|------|------|------|------|------|------|--------|
| init best | 0.058 | 0.492 | 0.247 | 0.218 | 0.420 | 0.731 | 1.365 | |
| W1 | 0.027 | 0.156 | 0.568 | 0.208 | 0.373 | 0.795 | 2.290 | |
| W2 | 0.010 | 0.010 | 0.990 | 0.010 | 0.342 | 0.990 | 0.335 | Bad |
| W3 | 0.012 | 0.188 | 0.549 | 0.113 | 0.366 | 0.679 | 2.229 | |
| W4 | 0.037 | 0.152 | 0.584 | 0.239 | 0.376 | 0.785 | 2.396 | |
| **W5** | **0.010** | 0.156 | 0.538 | 0.253 | 0.399 | 0.746 | **2.403** | |
| W6 | 0.010 | 0.154 | 0.538 | 0.254 | 0.401 | 0.747 | 2.394 | Near W5 |
| W7 | 0.015 | 0.146 | 0.541 | 0.235 | 0.414 | 0.730 | 2.365 | |
| **W8** | **0.010** | 0.108 | 0.581 | 0.206 | 0.365 | 0.740 | **2.433** | **NEW BEST** |

### Key Observations
1. **x0 = 0.010 in every top result**: W5 (2.403), W6 (2.394), W8 (2.433) all have x0=0.010
2. **0.010 was the old clip floor** — the true optimum may be at x0 < 0.01 or even x0 = 0
3. **Now that clipping is fixed to [0, 1], we should try x0 closer to 0**
4. **W8's key difference from W5**: Lower x1 (0.108 vs 0.156) and higher x2 (0.581 vs 0.538)

### Current Week 9 Query: (0.046, 0.117, 0.620, 0.242, 0.372, 0.747) — PROBLEMATIC
- **x0=0.046**: Much higher than the optimal 0.010. Every good result has x0 ≤ 0.037, and the best two have x0=0.010.
- x2=0.620: Higher than any previous submission (W8 was 0.581). Extrapolation risk.
- Selected by EI+UCB combined metric, which rewards uncertainty

### Corrected Strategy
- **x0 should be 0.005 or lower**: Now that clipping allows it, test if x0→0 helps
- **Radius: 0.025** around (0.005, 0.108, 0.581, 0.206, 0.365, 0.740)
- **Use PI**: Probability of beating 2.433 is more informative than EI here
- **Key exploration**: x0 ∈ [0, 0.015], x1 ∈ [0.085, 0.130]

---

## <a name="f8"></a>9. F8 (8D) — Near-Optimal, Fine-Tuning

### Data Summary
| Week | x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 | y | Status |
|------|-----|-----|-----|-----|-----|-----|-----|-----|------|--------|
| W1 | 0.018 | 0.114 | 0.159 | 0.011 | 0.905 | 0.298 | 0.175 | 0.170 | 9.897 | |
| W2 | 0.010 | 0.081 | 0.231 | 0.010 | 0.990 | 0.177 | 0.124 | 0.081 | 9.768 | Regression |
| **W3** | 0.025 | 0.095 | 0.162 | 0.036 | 0.887 | 0.318 | 0.167 | 0.205 | **9.915** | |
| W4 | 0.053 | 0.056 | 0.243 | 0.010 | 0.938 | 0.357 | 0.152 | 0.176 | 9.877 | Regression |
| W5 | 0.025 | 0.095 | 0.162 | 0.036 | 0.887 | 0.318 | 0.167 | 0.205 | 9.915 | EXACT_RETURN |
| W6 | 0.025 | 0.095 | 0.163 | 0.036 | 0.887 | 0.319 | 0.166 | 0.205 | 9.915 | Near-exact |
| W7 | 0.020 | 0.093 | 0.171 | 0.043 | 0.894 | 0.325 | 0.161 | 0.214 | 9.915 | Near-exact |
| **W8** | 0.026 | 0.095 | 0.153 | 0.049 | 0.871 | 0.334 | 0.170 | 0.223 | **9.928** | **NEW BEST** |

### Key Observations
1. **Confirmed deterministic**: W5 exact return → 9.915. W6/W7 near-exact → also 9.915
2. **Very flat near optimum**: Perturbations of 0.01-0.02 barely change the output (W3/W5/W6/W7 all gave 9.915)
3. **W8 found genuine improvement**: x3 increased (0.036→0.049), x4 decreased (0.887→0.871), x5 increased (0.318→0.334), x7 increased (0.205→0.223)
4. **Pattern**: Low x0 (0.018-0.026), low x3 (0.011-0.049), high x4 (0.871-0.905)

### Current Week 9 Query: (0.041, 0.084, 0.139, 0.061, 0.879, 0.348, 0.184, 0.231) — MODERATE RISK
- **x0=0.041**: Higher than every good result (W1: 0.018, W3: 0.025, W8: 0.026). W4 had x0=0.053 and regressed.
- **x3=0.061**: Higher than any submission that didn't regress. Risky.
- GP predicted 9.953 ± 0.021 but this is the GP's optimistic estimate

### Corrected Strategy
- **Radius: 0.010** (very tight around W8)
- **Use PI**: We only want to beat 9.928
- **Constrain x0 ≤ 0.030**: Historical data shows x0 > 0.035 leads to regression
- **Center on W8**: (0.026, 0.095, 0.153, 0.049, 0.871, 0.334, 0.170, 0.223)

---

## <a name="corrected"></a>10. Corrected Week 9 Strategies

### Summary of Changes

| Function | Old Strategy | Problem | Corrected Strategy |
|----------|-------------|---------|-------------------|
| F1 | Directional, r=0.025, TS override | x1 shifted 0.030 — certain regression | Exploit, r=0.008, PI, no TS override |
| F2 | Explore, r=0.12 | Every exploration attempt failed | EXACT_RETURN to best point |
| F3 | Exploit, r=0.015, TS override | TS override, x2 slightly too high | Exploit, r=0.010, PI, center on W7 |
| F4 | Directional, r=0.03 | Reasonable | Keep (acceptable risk) |
| F5 | Boundary, r=0.025 | x0/x1 shifted too far from W1 | Boundary, r=0.008 for x0/x1, pin x2/x3 to (0.995, 1.0) |
| F6 | Directional, r=0.03 | EI reversed x1 direction | Directional, r=0.02, validate trajectory |
| F7 | TuRBO, r=0.04 | x0=0.046 too high | Exploit, r=0.025, force x0 < 0.010 |
| F8 | Exploit, r=0.015 | x0=0.041 too high | Exploit, r=0.010, constrain x0 ≤ 0.030 |

### Ideal Week 9 Queries (Manual Override Where Needed)

| Function | Query | Rationale |
|----------|-------|-----------|
| F1 | ~(0.631, 0.620) | Tiny perturbation from W8 best |
| F2 | (0.702637, 0.926564) | Exact best point (noise may give higher value) |
| F3 | ~(0.522, 0.630, 0.378) | Tight around W7 best, slight adjustment |
| F4 | ~(0.422, 0.397, 0.386, 0.407) | Keep current (reasonable) |
| F5 | ~(0.363, 0.273, 0.998, 0.999) | Keep x0/x1 at W1, push x2/x3 higher |
| F6 | ~(0.671, 0.107, 0.763, 0.743, 0.048) | Follow ALL 5 dimension trends |
| F7 | ~(0.005, 0.108, 0.581, 0.206, 0.365, 0.740) | Test x0 near zero |
| F8 | ~(0.024, 0.095, 0.150, 0.052, 0.868, 0.338, 0.172, 0.226) | Very tight around W8 |

---

## <a name="roadmap"></a>11. Remaining Weeks Roadmap

### Week 9 (This Week): Exploit + Fix Bugs
- Fix clipping, TS override, radius sizes
- Conservative queries for peaked functions (F1, F5)
- Test x0→0 for F7

### Week 10: Evaluate and Adjust
- If F1 improved: continue exploiting; if regressed: return to W8 point
- If F2 exact return gives higher value: noise confirmed, keep returning
- If F5 with high x2/x3 improved: explore x2/x3 > 0.998
- If F7 with low x0 improved: explore x0→0 further

### Week 11: Consolidate
- For improving functions: continue trajectory
- For stagnant functions: shift to EXACT_RETURN strategy early
- Start considering final rankings

### Week 12: Penultimate — Last Exploration
- Final opportunity for any exploratory bets
- Lock in strategies for each function
- Prepare EXACT_RETURN coordinates

### Week 13 (Final): EXACT_RETURN
- Submit best known point for EVERY function
- No exploration, no risk
- Best coordinates from samples.csv, maximum precision

---

## Code Fixes Required

### 1. Selection Logic Fix
```python
# BEFORE (buggy):
if mu[ts_idx] > mu[ei_idx] + 0.01 * abs(y_best):
    selected_idx = ts_idx  # TS override too easy to trigger

# AFTER:
# For exploit/boundary strategies, use PI (conservative)
# For directional strategies, use EI but validate direction
# Never use TS override for exploitation
```

### 2. F5 Trust Region Fix
```python
# Use per-dimension radii: tight for x0/x1, wide for x2/x3
# x0: center=0.363, range (0.355, 0.370)
# x1: center=0.273, range (0.265, 0.280)
# x2: range (0.995, 1.000)
# x3: range (0.996, 1.000)
```

### 3. F7 Lower Bound Fix
```python
# Allow x0 to go to 0:
# Center trust region at x0=0.005 (not 0.010)
# Or use boundary-aware trust region with x0 pinned to (0, 0.015)
```

### 4. Post-Selection Validation
```python
# For directional strategies, verify the selected point follows the improvement direction
# For each dimension, check: sign(x_selected - x_best) == sign(direction[dim])
# If not, prefer PI or mean-maximization selection
```
