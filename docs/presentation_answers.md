# BBO Capstone Project — Presentation Notes

## 1. An overview of your BBO approach

**What are we trying to achieve?**

The objective is to maximise 8 unknown black-box functions, each operating over a different number of input dimensions (from 2D up to 8D), where all inputs lie in [0, 1]. We have no knowledge of the functions' structure, no access to gradients, and a severely constrained budget: just one query per function per week, over 13 weeks. The challenge mirrors real-world scenarios like hyperparameter tuning, drug discovery, and A/B testing — situations where each evaluation is expensive and you must make every query count.

**Overall strategy**

The core approach is Bayesian Optimisation (BO): we fit a surrogate model — a Gaussian Process (GP) — to observed input-output pairs, then use an acquisition function to decide where to query next. The GP provides both a prediction and an uncertainty estimate at any point in the input space, which lets us reason about whether to explore (sample uncertain regions) or exploit (refine near known good points).

In practice, the pipeline works as follows:
1. **Fit a multi-kernel GP ensemble** — three Matern kernels (v=0.5, 1.5, 2.5) weighted by log marginal likelihood and leave-one-out cross-validation, so the model automatically adapts its smoothness assumptions to each function.
2. **Apply output warping** (Yeo-Johnson transform) to handle skewed output distributions, inspired by HEBO, the NeurIPS 2020 BBO competition winner.
3. **Generate candidates** within a trust region (a small hypercube around the current best) using Sobol quasi-random sequences.
4. **Score candidates** using a multi-acquisition ensemble (PI + EI + UCB, normalised and weighted), then submit the top-scoring candidate.

Each function gets a tailored configuration — different trust region radii, acquisition weightings, directional biases, and per-dimension constraints — based on what we've learnt about its landscape.

---

## 2. How your strategy has evolved

**Key changes since the early rounds**

The approach has gone through several distinct phases:

- **Weeks 1–3 (GP-UCB + local perturbation):** We started with vanilla GPs and Upper Confidence Bound acquisition. The main insight was that with only 10 initial data points, GPs extrapolate poorly — they tend to push queries towards the boundaries of the input space. We introduced local Gaussian perturbation around promising points as a workaround.

- **Weeks 4–6 (Neural network surrogates):** We replaced GPs with ensembles of 7 neural networks, gaining gradient access and faster training. This led to F1's first major breakthrough (0.196 to 1.626, an 8x improvement). However, NN gradient estimates proved unreliable for large steps and caused regressions in several functions.

- **Weeks 7–8 (TuRBO + multi-kernel ensembles):** Inspired by NeurIPS 2020 competition methods, we adopted trust-region Bayesian optimisation (TuRBO) and multi-kernel GP ensembles. Week 8 was the strongest single week: 5 out of 8 new best values. The multi-kernel ensemble was the key — it automatically selects appropriate smoothness rather than forcing a single kernel on all functions.

- **Weeks 9–10 (Bug fixes + coordinate-wise search):** A rigorous audit uncovered critical bugs: domain clipping at [0.01, 0.99] instead of [0, 1] had silently destroyed F5's results for 7 weeks; a Thompson Sampling override was selecting exploratory points during intended exploitation; and a pinning-order bug in boundary-aware optimisation was overriding our boundary corrections. Fixing these plus a coordinate-wise line search on F1 delivered a further +12.4% improvement.

- **Weeks 11–12 (HEBO-inspired refinements):** We adopted three techniques from HEBO: output warping (Yeo-Johnson), Noisy EI for the confirmed-noisy F2, and multi-acquisition ensembles. Week 11 delivered 5 out of 8 new best values — matching Week 8 as the best single week.

**What influenced these changes?**

Primarily data trends and post-hoc analysis of regressions. Every time a query regressed, we investigated why — was the trust region too large? Did the acquisition function select an exploratory point when we needed exploitation? Was a software bug silently corrupting the search? This forensic approach to failures drove most of the methodological improvements.

**Guiding principles that emerged:**

- **Trust region radius must match peak width.** Too large and you overshoot; too small and you don't explore enough. Functions like F1 and F4 have extremely narrow peaks (radius must be 0.002–0.003), while F7 and F8 tolerate 0.005–0.008.
- **PI over EI for exploitation.** Expected Improvement rewards uncertainty, which can steer queries away from the best-known region. Probability of Improvement asks the right question: "will this beat what we have?"
- **Function-specific strategies are essential.** A single configuration cannot serve 8 functions with different dimensionalities, noise levels, and landscape shapes.
- **Never trust a single acquisition function.** The multi-acquisition ensemble (PI + EI + UCB) is more robust than any individual method.

---

## 3. Patterns, data and insights

**Most meaningful trends observed:**

- **Kernel smoothness divides the functions into two groups.** F1–F6 favour rough kernels (Matern v=0.5), meaning their landscapes have sharp, discontinuous features. F7–F8 favour smooth kernels (Matern v=2.5), indicating gentler, more continuous surfaces. This distinction matters enormously for how aggressively we can extrapolate from existing data.

- **Some functions have extremely narrow peaks.** F1's peak is roughly 0.02 units wide — shifting just 0.003 in one coordinate caused a 10.7% performance drop. F4 is similarly peaked: a trust region radius of 0.010 (seemingly small) caused a 15% regression. These functions demand ultra-conservative exploitation.

- **F2 is definitively noisy.** Three evaluations of the exact same input returned 0.611, 0.667, and 0.590. This confirmed that standard acquisition functions (which assume noise-free observations) are inappropriate. Switching to Noisy EI — which uses the GP-predicted best rather than the observed best as the reference — led to a new best in Week 11.

- **F5 has a boundary optimum.** Two of its four inputs should be pinned at exactly 1.0 (the upper boundary of the domain). The domain clipping bug at [0.01, 0.99] had prevented us from discovering this for 7 weeks — fixing it was the single largest point gain of the project (+55.7 points).

- **F7 and F8 show consistent directional trends.** Both achieved 4 consecutive weekly improvements by following identified gradient directions (e.g., x3 increasing for F7; x5 and x7 increasing for F8). These are the only functions where momentum-based strategies have reliably worked.

**Variables that influence results most:**

For peaked functions (F1, F3, F4, F6), the trust region radius is the single most important hyperparameter — get it wrong and you regress regardless of your surrogate model quality. For noisy functions (F2), the choice of acquisition function matters more than the model. For boundary functions (F5), getting the domain constraints right was more impactful than any modelling improvement.

**How these observations shaped understanding:**

The overarching lesson is that in extreme low-budget optimisation, understanding the function's character — peaked vs. flat, noisy vs. deterministic, interior vs. boundary optimum — matters more than having a sophisticated surrogate model. The best model in the world will not help if you are searching in the wrong region or using the wrong exploitation radius.

---

## 4. Decision-making and iteration

**Balancing exploration and exploitation**

In the early weeks (1–6), we leaned towards exploration — trying diverse regions of the input space to build a rough global picture. From Week 7 onwards, as we accumulated enough data for reliable GP fits (18+ points per function), we shifted heavily towards exploitation using TuRBO trust regions. The final two weeks are pure exploitation: Week 12 uses ultra-tight radii (0.002–0.008), and Week 13 will simply resubmit the best-known points (EXACT_RETURN).

The transition was not uniform across functions. F7 and F8 still received moderate trust region radii (0.005–0.008) because they showed consistent directional improvement, suggesting we hadn't yet reached their peaks. F1, F3, F4, and F6 received the tightest radii (0.002–0.003) because any larger step caused regression.

**Example 1: The F1 coordinate-wise breakthrough (what worked)**

By Week 9, F1's best value was 1.773 at approximately [0.631, 0.622]. Rather than searching a 2D trust region, we ran a coordinate-wise line search — holding x0 fixed and varying x1, then vice versa. This revealed the peak was actually near [0.630, 0.629], not [0.631, 0.622]. A single well-placed query in Week 10 jumped to 1.993 — a 12.4% improvement, the largest single-week gain of the project. The lesson: when the landscape is extremely peaked, reducing the search to one dimension at a time can be more effective than searching all dimensions simultaneously.

**Example 2: The F6 trajectory hypothesis (what didn't work)**

After Week 8's success on F6, we noticed a directional trend: the best values improved as the query moved in a particular direction across weeks 7 and 8. We hypothesised that continuing along this trajectory would yield further improvement. Week 9 took a full step in that direction — it regressed. Week 10 took a half step — it also regressed. Week 11 tried a GP-guided perpendicular search — still regressed. The trajectory hypothesis was falsified after three consecutive failures. The lesson: patterns in 2–3 data points can be coincidental, especially with peaked functions where small shifts cause large output changes.

**Handling uncertainty and unexpected results**

When a query regresses, we treat it as a diagnostic opportunity rather than a setback. Each regression generates a data point that constrains the function's landscape — it tells us where the peak is *not*. We systematically reduce trust region radii after regressions and investigate whether the regression was caused by a strategic error (wrong direction, wrong radius) or a technical bug (domain clipping, acquisition function misbehaviour). This forensic approach led to the discovery of three critical bugs in Week 9, which collectively were more valuable than any single modelling improvement.

---

## 5. Next steps and reflection

**Planned actions**

- **Week 13 (final week): EXACT_RETURN.** Submit the best-known point for each function to lock in our results. For functions where Week 12 queries may have improved (we're awaiting results), we'll submit the better of the two.
- If we had additional weeks, the highest-value next steps would be: (a) implementing input warping (Kumaraswamy transform) as used by HEBO, to handle non-stationary landscapes; (b) a heteroscedastic noise model for F2, where different regions of the input space likely have different noise levels; and (c) evolutionary acquisition optimisation (NSGA-II) instead of our current Sobol + argmax approach.

**Connection to the broader ML landscape**

This project is a distilled version of the hyperparameter optimisation problem that underpins all of modern machine learning. Every time a practitioner tunes learning rates, regularisation strengths, or architectural choices, they face the same trade-off: each training run is expensive, the objective landscape is unknown, and you need principled methods to decide what to try next. The techniques we've used — Gaussian processes, trust regions, acquisition function ensembles, and output warping — are the same ones powering tools like Vizier, Optuna, and HEBO that automate this process at scale. The extreme budget constraint (1 query per week) forced us to think carefully about every decision, which is precisely the regime where Bayesian optimisation offers the most value over random or grid search.

**Communicating to a non-technical audience**

Imagine you're trying to find the highest point on a mountain range, but you're blindfolded and can only take one step per day. Each step, you feel the ground under your feet and get a height reading. Our approach builds a mental map of the terrain from these readings — predicting where the peaks likely are and where we're most uncertain — then chooses each step to either refine what we know about a promising peak or explore an uncharted area. Over 12 weeks, this systematic approach has found strong results across all 8 "mountain ranges," with the biggest wins coming from carefully analysing our missteps to refine the map.
