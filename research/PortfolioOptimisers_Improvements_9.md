# PortfolioOptimisers.jl — every novel idea from reports 1 to 8, consolidated and prototyped

**Date:** 2026-08-16 · **Base:** branch `dev`, commit `0ddb6d671`, v0.28.0 · **Author:** Claude (Opus 5)

---

## 1. What this report is

I read reports 1 to 8 in full, extracted every distinct idea, removed the duplicates and the
already-implemented, and built a working prototype for each survivor. **Sixteen new prototypes**
join the six from report 8. All twenty-two are documented, and all twenty-two ran.

```bash
julia -t 1 --project=. research/prototypes/run_all.jl     # prototypes 01-06, report 8
julia -t 1 --project=. research/prototypes/run_novel.jl   # prototypes 07-22, this report
```

Both exit zero. Every number quoted below comes from that output.

The code is **10,353 lines** across 22 modules plus two drivers. Each module states its notation,
defines every variable, and documents every function with its arguments, returns, mathematics,
validation and known limits. None loads PortfolioOptimisers.jl, so each can be adapted into `src/`
on its own.

### What is different about this report

Reports 1 to 7 propose. Report 8 checked its claims against the code. **This one also checks the
mathematics by running it**, and the running found errors. Six claims I wrote had to be corrected
after a test disagreed with them:

| I claimed | The test said | Where |
| :--- | :--- | :--- |
| Effective number of models is capped at `M` | Capped at `M - 1`; centring costs a degree of freedom | Prototype 3 |
| Robustness lies in `[0, 1]` | It goes to `-0.82`; active share is unbounded for levered portfolios | Prototype 13 |
| Volatility clustering breaks conformal coverage | Marginal coverage survives. **Conditional** coverage splits 0.944 against 0.841 | Prototype 12 |
| A cost-aware rule that amortises always trades | A rule that does *not* amortise **never** trades. Zero trades in 1248 periods | Prototype 16 |
| The empirical tail-dependence estimator is merely noisy | It is **biased**, and a near-Gaussian copula measures `0.31` at `q = 0.05` when the truth is `0` | Prototype 10 |
| `best_crp` converged | It did not, and the best single asset beat it, which is impossible | Prototype 9 |

Each is now documented with the measurement that produced it. **That is the value of building the
thing rather than describing it.**

---

## 2. The consolidation

Every distinct idea in reports 1 to 8, and where it went.

### Built (22 prototypes)

| Idea | Raised in | Prototype |
| :--- | :--- | :--- |
| Sampling from a fitted prior | 4, 8 | `01_scenario_generation.jl` |
| Wasserstein ambiguity set | 1, 3, 4, 5, 8 | `02_wasserstein_ambiguity.jl` |
| Model disagreement, consensus, ensembles | 4, 8 | `03_model_population.jl` |
| Calibration against a known truth | 8 | `04_simulated_truth_calibration.jl` |
| The breakeven view | 8 | `05_breakeven_view.jl` |
| Numeric performance summary | 5, 6, 8 | `06_performance_summary.jl` |
| Differentiable layer, sensitivity analysis | 1, 3, 7 | `07_optimiser_sensitivity.jl` |
| Critical line algorithm | 2 | `08_critical_line.jl` |
| Online portfolio selection | 2, 5 | `09_online_portfolio_selection.jl` |
| Conditional stress, scenario transforms, t-copula | 2, 3, 4, 6 | `10_conditional_stress.jl` |
| Gelbrich and divergence ambiguity sets | 3, 4 | `11_moment_and_divergence_ambiguity.jl` |
| Bootstrap weight intervals, conformal prediction | 3, 4 | `12_weight_uncertainty.jl` |
| Robustness analysis and the robustness frontier | 4 | `13_robustness_frontier.jl` |
| Constraint attribution, Shapley, rebalance decomposition | 2, 4, 7 | `14_attribution.jl` |
| Portfolio state, frozen positions, drift, NaN awareness | 4 | `15_portfolio_state.jl` |
| Rebalancing policy layer | 3, 5 | `16_rebalancing_policies.jl` |
| Market impact and optimal execution | 1, 2, 3, 4, 7 | `17_market_impact.jl` |
| Tax lots, wash sales, after-tax rebalancing | 2, 3 | `18_tax_aware.jl` |
| Online and incremental estimators, `partial_fit` | 3, 4 | `19_online_moments.jl` |
| Laplacian regularisation, neighbourhood constraints | 3 | `20_graph_structure.jl` |
| Regime models, worst-case regime | 2, 4, 7 | `21_regime_models.jl` |
| Provenance, capability and compatibility checking | 4 | `22_provenance_and_capability.jl` |

### Already implemented, so not built

Report 8, section 4 lists these with file and line evidence: the backtest engine, walk-forward and
combinatorial cross-validation, search cross-validation, pre-selection, higher moments, Brinson and
factor attribution, graph and phylogeny constraints, custom objectives and constraints, turnover,
fees, tracking, regime-adjusted moments, discrete allocation, and mixed-integer constraints.

Two more, found while consolidating:

- **ESG constraints** (reports 2, 5, 6). A linear constraint on a per-asset score is already
  expressible through `UniverseSets` and `LinearConstraintGeneration`. A dedicated type would be
  sugar over existing machinery, not a capability.
- **Batteries-included pipelines** (report 3). The `Pipeline` estimator is exactly this.

### Deliberately not built

| Idea | Raised in | Why not |
| :--- | :--- | :--- |
| Reinforcement learning environments | 2, 6 | A research topic, not a library feature. It needs an environment, a reward and a training loop, none of which belongs beside an estimator. |
| Large-language-model views | 6 | A network call inside a Prior destroys reproducibility, which is the property the Pipeline exists to protect. |
| Quantum and quantum-inspired solvers | 2 | No available solver beats Clarabel on these problems. JuMP would carry one if it existed. |
| Rough volatility, fractional models | 2 | A univariate research topic with no seam in a cross-sectional allocation library. |
| Causal portfolio optimisation | 2 | The identification problem is unsolved for asset returns. A discovered graph on returns is a correlation graph with a stronger claim attached. |
| GPU acceleration | 1, 2, 5 | The bottleneck is the conic solver, which is not the library's code. Profile first. |
| Broker connectivity, Python wrapper | 5 | Separate packages. Report 5 says so itself. |
| Multi-stage stochastic programming | 2 | See below. |
| Multi-period optimisation in the core | 1, 3, 7 | See below. |
| Mixed-frequency data | 3 | Real, but it is a data-alignment problem, not a portfolio one. It belongs upstream of `ReturnsResult`. |
| scikit-learn regressor interoperability | 4 | An interoperability layer, not mathematics. It needs a decision about extension boundaries that is the maintainer's, not mine. |

**Multi-period deserves its paragraph.** It is the most valuable item on this list, and cvxportfolio
does it well. I still keep it out of the core, for a structural reason: `w` is one vector in every
file in `src/20_Optimisation/09_JuMPConstraints/`. A horizon index breaks all thirteen, plus every
risk-measure constraint file. That is a rewrite, not a feature. The design note in
`research/NOTRACK_AlgoTrader.jl_README.md` already places the loop in the companion package and
treats `optimise` as a pure function called inside it. **I agree with that split.** Prototypes 15,
16 and 17 supply the pieces that companion would need — state, policy and impact — without touching
the single-period core.

---

## 3. The sixteen new prototypes

Each section gives the gap, the mathematics, and the measured result. **The full argument lists,
validation rules and known limits are in the module docstrings**, which are the deliverable; this is
the map.

---

### 3.1 Optimiser sensitivity — `07_optimiser_sensitivity.jl`

Reports 1, 3 and 7 ask for a "differentiable layer", "automatic differentiation through the
pipeline" and "sensitivity analysis". **These are one object.**

#### Mathematics

For the equality-constrained problem

$$
\min_w \; \tfrac{1}{2} w'\Sigma w - \tfrac{1}{\gamma}\mu'w \quad \text{s.t.} \quad Aw = b
$$

stationarity is the saddle-point system

$$
\begin{bmatrix} \Sigma & A' \\ A & 0 \end{bmatrix}
\begin{bmatrix} w \\ \nu \end{bmatrix} =
\begin{bmatrix} \mu/\gamma \\ b \end{bmatrix},
\qquad
\begin{bmatrix} K & G \\ \cdot & \cdot \end{bmatrix} = \begin{bmatrix} \Sigma & A' \\ A & 0 \end{bmatrix}^{-1}
$$

Every derivative is a product with one block:

$$
\frac{\partial w}{\partial \mu} = \frac{K}{\gamma}, \qquad
\frac{\partial w}{\partial b} = G, \qquad
\frac{\partial w}{\partial t}\bigg|_{\Sigma + tE} = -K E w, \qquad
\frac{\partial L}{\partial \mu} = \frac{K}{\gamma}\frac{\partial L}{\partial w}
$$

The last is the vector-Jacobian product. **It makes the optimiser a trainable layer at the cost of
one matrix-vector product**, with no solver re-run and no unrolled iterations.

#### Verified

```
dw/dmu     max |analytic - finite difference| = 1.29e-08
dw/dsigma  max |analytic - finite difference| = 3.35e-06
K null space contains A': 1.1e-14   pullback == J'v: 5.9e-15
constraint price: d(obj)/db = 0.020824, -nu = 0.020824
```

#### The finding

`opnorm(K/γ)` is a **usable diagnostic available before any backtest**: it is the factor by which
an error in the expected returns is magnified into an error in the weights.

| ridge added to `Σ` | μ-amplification |
| :--- | ---: |
| `0.5` | 0.67 |
| `0.05` | 6.54 |
| `0.005` | 55.56 |
| `0.0005` | **222.37** |

That is the Markowitz enigma of Best and Grauer (1991), as a number a caller can compute in one line
and act on.

---

### 3.2 Critical line algorithm — `08_critical_line.jl`

Report 2 asks for it. The library solves every mean-variance problem with a generic conic solver, so
a frontier costs one solve per point and the points between are interpolated by guess.

#### Mathematics

For a free set $F$ and a bounded set $B$ held at $w_B$, with $C = \Sigma_{FF}^{-1}$,

$$
w_F(\lambda) = \alpha + \lambda\beta, \qquad
\beta = C\mu_F - \frac{a_0}{a_1}C\mathbf{1}, \qquad
\alpha = -C\Sigma_{FB}w_B + \frac{s + a_2}{a_1}C\mathbf{1}
$$

with $a_0 = \mathbf{1}'C\mu_F$, $a_1 = \mathbf{1}'C\mathbf{1}$, $a_2 = \mathbf{1}'C\Sigma_{FB}w_B$.
**Both coefficients are constant while the free set is constant**, so the frontier is piecewise
affine and a finite list of turning points describes it exactly.

#### Verified

```
15 turning points, lambda from 691.5 down to 0
KKT stationarity on the free set: 1.39e-15 ; sign condition on bounds: 5.72e-17
max Sharpe 0.1352 at lambda 0.3459, found by interpolation with no solver
```

#### The finding

Against Clarabel at default tolerance the agreement was `1.9e-4`. **That gap is the solver's, not
the algorithm's.** At tightened tolerance it fell to `2.4e-6`, and the CLA objective was never worse
than the solver's, by `-1.4e-14`. The KKT residual of `1.4e-15` is the real accuracy. The frontier
also carries a free audit trail: the active set at every `λ`, which answers "at which risk aversion
does asset seven enter?" — a question no repeated solve can answer.

---

### 3.3 Online portfolio selection — `09_online_portfolio_selection.jl`

Report 5 notes that no Julia package implements these. They need no prior, no covariance and no
solver, and they carry worst-case guarantees that hold for **every** price sequence.

#### Mathematics

With price relatives $x_t$ and wealth $S_T = \prod_t \langle w_t, x_t\rangle$:

- **Universal portfolio** (Cover 1991): $w_{t+1} = \int b\,S_t(b)\,d\mu(b) \big/ \int S_t(b)\,d\mu(b)$.
- **Exponentiated gradient**: $w_{t+1,i} \propto w_{t,i}\exp\!\big(\eta x_{t,i}/\langle w_t,x_t\rangle\big)$.
- **Online Newton step**: $w_{t+1} = \Pi_\Delta\big(\delta A_t^{-1}b_t\big)$, with logarithmic regret.
- **PAMR**: $\min_w \tfrac12\|w-w_t\|^2$ subject to $\langle w,x_t\rangle \le \varepsilon$.
- **OLMAR / RMR**: revert to a moving average, or to the $L_1$ (spatial) median found by Weiszfeld's
  iteration.

All use the simplex projection of Duchi and co-authors (2008): $w_i = \max(v_i - \theta, 0)$ with
one scalar $\theta$ set by the budget.

#### Verified

| strategy | reverting market | trending market |
| :--- | ---: | ---: |
| uniform constant rebalanced | 1.82 | 4.81 |
| universal portfolio | 1.70 | 5.89 |
| exponentiated gradient | 1.77 | 5.31 |
| online Newton step | 1.82 | 4.82 |
| PAMR | 46.6 | 0.00 |
| OLMAR | 195.3 | 0.00 |
| **RMR** | **212.2** | **0.00** |
| best constant portfolio, in hindsight | 1.83 | 144.7 |

#### The finding

The reversion family is **not a strategy, it is a bet on a market property**, and the bet is total.
On a reverting market RMR turns 1.83 into 212. On a trending one it goes to zero. Running a momentum
rule and a reversion rule side by side is a cheap, assumption-free test of which regime a market is
in, and it costs no estimation at all. Negative regret against the best constant portfolio is
expected and is not a bug: the bound only says a strategy cannot fall far behind.

---

### 3.4 Conditional stress — `10_conditional_stress.jl`

A stress test that shocks equities and leaves credit at its unconditional mean is not a stress test.

#### Mathematics

Partition into the stressed block 1 and the rest:

$$
\mu_{2\mid 1} = \mu_2 + \Sigma_{21}\Sigma_{11}^{-1}(v - \mu_1), \qquad
\Sigma_{2\mid 1} = \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}
$$

The t-copula's lower tail dependence has a closed form (Demarta and McNeil 2005):

$$
\lambda = 2\,t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)
$$

#### Verified

```
conditional mean error 4.6e-05, conditional covariance error 4.3e-07  (200k scenarios)
a -5% shock in asset 1 moves asset 2 to -0.0172, against 0.0002 unconditionally
```

#### The finding

**The empirical tail-dependence estimator is badly biased upward, and the bias dies slowly.**
Measured on two million scenarios at copula correlation 0.595:

| `q` | `ν=3` | `ν=5` | `ν=12` | `ν=500` |
| :--- | ---: | ---: | ---: | ---: |
| `0.05` | 0.426 | 0.383 | 0.341 | **0.307** |
| `0.005` | 0.384 | 0.298 | 0.222 | 0.150 |
| `0.0005` | 0.385 | 0.282 | 0.170 | 0.083 |
| theory | 0.370 | 0.263 | 0.092 | **0.000** |

Read the last column. A **near-Gaussian** copula, whose true tail dependence is exactly zero,
measures `0.307` at a five per cent tail. A practitioner estimating tail dependence from a five per
cent tail will conclude that a Gaussian model has strong tail dependence, which is the exact
opposite of the truth, and it is the error that makes Gaussian stress tests look adequate.

---

### 3.5 Moment and divergence ambiguity — `11_moment_and_divergence_ambiguity.jl`

Three ambiguity families answer three different questions: what if the **data** moves (Wasserstein,
prototype 2), what if the **moments** are wrong (Gelbrich), what if the **probabilities** are wrong
(divergence). A caller should be able to say which they mean.

#### Mathematics

Gelbrich distance, with the Bures-Wasserstein metric on covariances:

$$
G^2 = \|\mu_1-\mu_2\|_2^2 + \operatorname{tr}\!\Big(\Sigma_1+\Sigma_2-2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\Big)
$$

Kullback-Leibler ball, by Donsker-Varadhan duality:

$$
\sup_{D(q\|p)\le\eta}\mathbb{E}_q[L] \;=\; \min_{\theta>0}\Big\{\theta\log\mathbb{E}_p\big[e^{L/\theta}\big] + \theta\eta\Big\},
\qquad q^\star_t \propto p_t e^{L_t/\theta^\star}
$$

**The whole problem collapses to a scalar minimisation, whatever `T` is** — the same collapse
prototype 5 exploits, for the same reason.

#### Verified

| `η` | dual value | `E_q*[L]` | `D(q*‖p)` | `θ*` |
| ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.00799 | 0.00799 | 0.00000 | ∞ |
| 0.010 | 0.01263 | 0.01263 | 0.01000 | 0.246 |
| 0.200 | 0.03287 | 0.03287 | 0.20000 | 0.075 |
| 1.000 | 0.07596 | 0.07596 | 1.00000 | 0.044 |

The dual value equals the tilt's expectation exactly, and the divergence binds at exactly `η`. A
brute-force search over two million random measures agreed with the dual to `1.8e-4`. Bures-Wasserstein
reduces to `‖√D₁ − √D₂‖_F` in the commuting case to `8.9e-16`.

#### The finding

**A divergence ball can only reweight scenarios that exist.** If every scenario is mild, no `η`
produces a severe worst case — the KL value is bounded above by the worst observed loss
(`0.172 ≤ 0.174` at `η = 5`). A Wasserstein ball can move a scenario to a value never observed.
That is the structural difference, and it decides which set to use.

---

### 3.6 Weight uncertainty and conformal prediction — `12_weight_uncertainty.jl`

Two different error bars: on the **weights** (bootstrap), and on the **outcome** (conformal, with a
distribution-free finite-sample guarantee).

#### Mathematics

Split conformal: with `n` exchangeable calibration scores,

$$
\hat{s} = s_{(k)}, \quad k = \lceil (n+1)(1-\alpha)\rceil
\quad\Longrightarrow\quad
1-\alpha \;\le\; \mathbb{P}(s_{n+1}\le \hat{s}) \;\le\; 1-\alpha+\tfrac{1}{n+1}
$$

#### Verified

| `α` | coverage | target | upper bound | inside |
| ---: | ---: | ---: | ---: | :--- |
| 0.20 | 0.8000 | 0.8000 | 0.8050 | yes |
| 0.10 | 0.8998 | 0.9000 | 0.9050 | yes |
| 0.05 | 0.9504 | 0.9500 | 0.9550 | yes |

The theorem, verified. Also: 18 calibration points cannot support a 95 per cent interval, and the
function returns `Inf` rather than pretending.

#### The finding

**Marginal coverage is weaker than it sounds.** On a simulated GARCH series with contiguous
calibration windows:

| quantity | value |
| :--- | ---: |
| marginal coverage | 0.893 |
| coverage in the calm half | 0.944 |
| coverage in the turbulent half | 0.841 |

The marginal number is on target and the conditional numbers are ten points apart. **The interval is
too wide when nothing is happening and too narrow exactly when it is needed.** Clustering does not
break the theorem; it breaks the interpretation. The fix is a volatility-adjusted score, and the
library already has the estimator for it in `src/08_Moments/36_RegimeAdjustedExpWeightedVariance.jl`.

Separately, the bootstrap makes the difference between procedures stark: mean weight standard
deviation `0.022` for minimum variance against **`8.16`** for tangency, with sign stability `0.99`
against `0.73`.

---

### 3.7 The robustness frontier — `13_robustness_frontier.jl`

Report 4 calls this its strongest original idea and does not define it. Return and risk are
properties of a **portfolio**. Robustness is a property of the **procedure**.

#### Mathematics

$$
R(P; X, \Pi) = 1 - \frac{1}{|\Pi|}\sum_{\pi\in\Pi} \mathrm{AS}\big(P(X),\, P(\pi(X))\big),
\qquad \mathrm{AS}(w_1,w_2)=\tfrac12\|w_1-w_2\|_1
$$

#### Verified

`R(equal weight) = 1.000000` exactly, as it must for a constant procedure. And monotone in
shrinkage, which is the property that validates the measure:

| shrinkage `λ` | `R` | mean shift |
| ---: | ---: | ---: |
| 0.0 | **-0.820** | 1.820 |
| 0.4 | -0.054 | 1.054 |
| 1.0 | 0.593 | 0.407 |

#### The finding

**The range is `(-∞, 1]`, not `[0, 1]`**, and that is informative rather than a defect. Active share
is bounded by one only for long-only fully-invested portfolios. A robustness of `-0.82` says the
average resample rewrites **more than 1.8 times the whole portfolio**. The measure is left
unnormalised so that number survives.

The three-objective front also behaves as predicted: adding robustness surfaces equal weight, which
a return-risk view discards, while correctly excluding the unshrunk tangency variants as dominated.

---

### 3.8 Attribution — `14_attribution.jl`

Three questions with one answer: what did this constraint cost, how is credit shared between
overlapping causes, and why did risk change at this rebalance.

#### Mathematics

$$
\phi_i = \sum_{S\subseteq N\setminus\{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\big[v(S\cup\{i\}) - v(S)\big]
$$

with efficiency $\sum_i \phi_i = v(N) - v(\emptyset)$, symmetry, the null-player property and
linearity. Young (1985) proves it is the **unique** such rule. For the rebalance decomposition with
two causes it reduces to averaging the two orderings, which is exactly Brinson's interaction term
split in half.

#### Verified

Efficiency `1.4e-14`, null player `0.0`, symmetry `0.0`. Euler's identity for risk contributions to
`8.7e-19`. The rebalance decomposition residual `0.0`.

#### The finding

**Leave-one-out attribution is not merely imprecise; on overlapping constraints it reports nothing
at all.** Two identical binding caps plus one that never binds:

| constraint | Shapley | leave-one-out |
| :--- | ---: | ---: |
| cap A | `-4.84e-5` | `0.0` |
| cap B (exact duplicate) | `-4.84e-5` | `0.0` |
| global cap, never binds | `0.0` | `0.0` |
| **sum** | `-9.67e-5` | **`0.0`** |
| **true total cost** | `-9.67e-5` | |

Leave-one-out **misses 100 per cent of the cost**, because removing either duplicate leaves the
other binding. Shapley splits it equally, gives the non-binding constraint exactly zero, and sums to
the true total with error `1.4e-20`.

---

### 3.9 Portfolio state — `15_portfolio_state.jl`

Report 4's strongest practical point: optimisation is not a stateless function of returns.

#### Mathematics

$$
w^{\text{drift}}_i = \frac{w_i(1+r_i)}{\sum_j w_j(1+r_j)}\sum_j w_j,
\qquad
\tau = \tfrac12\big\|w^{\text{target}} - w^{\text{drift}}\big\|_1
$$

Frozen positions reduce the budget rather than the universe: solve the tradable block to sum to
`budget - held_fixed`, and the reassembled portfolio satisfies the original budget with **no
post-hoc renormalisation**, which would silently rescale the frozen positions.

#### Verified

```
drift preserves the budget to 0.0e+00
frozen holds 0.1870, sub-budget 0.8130, restored budget 1.000000, frozen preserved exactly true
```

#### The finding

**Turnover measured against the last target instead of the drifted holdings understated the trade by
54.2 per cent** — 0.0667 against 0.1455. That is real money, invisible to any calculation that never
sees the holdings. The library's `src/15_Turnover.jl` is *capable* of the correct calculation; what
is absent is the state object that makes the drifted vector the natural argument.

---

### 3.10 Rebalancing policies — `16_rebalancing_policies.jl`

#### Verified

| policy | trades | cost | gross | **net** |
| :--- | ---: | ---: | ---: | ---: |
| always | 1248 | 0.0493 | 1.7466 | 1.6627 |
| never | 0 | 0.0000 | 1.7102 | 1.7102 |
| calendar, 21 days | 60 | 0.0126 | 1.7597 | 1.7376 |
| threshold 0.05 | 55 | 0.0127 | 1.7440 | 1.7221 |
| cost aware | **0** | 0.0000 | 1.7102 | 1.7102 |
| **hybrid** | 20 | 0.0069 | 1.7581 | **1.7460** |

#### The finding

**The cost-aware rule never traded, and that is a horizon bug in the idea, not in the code.** The
utility gain is per period; the cost is paid once. Comparing them directly means the rule almost
never fires. The fix is to compare like with like:

$$
\text{trade when}\quad h\big(U(w^{\text{target}}) - U(w^{\text{held}})\big) - c > \text{threshold}
$$

where `h` is the expected holding period. **A rule with `h = 1` never trades; a rule with `h = ∞`
always trades.** Neither extreme is right, and the prototype deliberately exposes the `h = 1` end so
the failure is visible rather than buried in a tuned constant. `HybridPolicy` sidesteps it with a
calendar ceiling, which is why it wins on the net figure.

---

### 3.11 Market impact — `17_market_impact.jl`

A fee is proportional to size. Impact is not, and the difference changes the answer rather than
just the bill.

#### Mathematics

Square-root law: price move $\propto \sigma\sqrt{Q/V}$, so **total** cost $\propto Q^{3/2}$.
Almgren-Chriss optimal liquidation, with $\tilde\eta = \eta - \gamma\tau/2$ and
$\tilde\kappa^2 = \lambda\sigma^2/\tilde\eta$:

$$
x_k = X_0\,\frac{\sinh\big(\kappa(T-t_k)\big)}{\sinh(\kappa T)},
\qquad
\mathbb{E}[C] = \frac{\gamma X_0^2}{2} + \frac{\tilde\eta}{\tau}\sum_k n_k^2,
\qquad
\mathrm{Var}[C] = \sigma^2\tau\sum_{k=1}^{n-1} x_k^2
$$

#### Verified

```
doubling the order multiplies cost by 2.8284, theory 2^1.5 = 2.8284
marginal cost divided by average cost = 1.5000, theory 1.5
lambda = 0 is exactly uniform (TWAP): max deviation 8.7e-11
the closed-form path is optimal: 400 of 400 perturbations were no better
```

| `λ` | half-life | `E[cost]` | `sd[cost]` |
| ---: | ---: | ---: | ---: |
| 0 | ∞ | 243,750 | 745,486 |
| 1e-7 | 11.26 | 248,374 | 675,771 |
| 1e-6 | 3.57 | 356,786 | 433,728 |
| 1e-5 | 1.14 | 823,664 | 195,203 |

#### The findings

Two, both counter-intuitive and both worth documenting in any adaptation.

1. **The marginal cost of the last share is 1.5 times the average cost per share.** A trader who
   prices a block at the average square-root cost under-charges the marginal decision by fifty per
   cent.
2. **Permanent impact drops out of the optimal path entirely.** Gatheral (2010) shows permanent
   impact must be linear to exclude dynamic arbitrage, and a linear permanent cost depends only on
   the total traded, not on the schedule. It changes the bill and not the plan.

And a tractability note: a `Q^(3/2)` cost is two rotated second-order cones, and a general exponent
`1+δ` is one `MOI.PowerCone`. The library already uses that cone in
`src/20_Optimisation/09_JuMPConstraints/13_WeightNormConstraints.jl`. **A convex impact model costs
nothing in tractability.**

---

### 3.12 Tax-aware lot selection — `18_tax_aware.jl`

Report 2 is right that no open-source portfolio library does this well. A position is not a number;
it is a stack of lots.

#### Mathematics

Tax per share for a lot is $\text{rate}(\text{lot})\cdot(\text{price} - \text{basis})$, and it does
not depend on which other lots are chosen. **So sorting by it is exactly optimal, not a heuristic** —
the problem has matroid structure and greedy solves it.

#### Verified

| method | tax | net | short gain | long gain |
| :--- | ---: | ---: | ---: | ---: |
| FIFO | 3755.00 | 33745.00 | 1500.00 | 16000.00 |
| LIFO | -140.00 | 37640.00 | -2000.00 | 3000.00 |
| HIFO | -140.00 | 37640.00 | -2000.00 | 3000.00 |
| LOFO | 3755.00 | 33745.00 | 1500.00 | 16000.00 |
| tax-optimal | -140.00 | 37640.00 | -2000.00 | 3000.00 |

#### The finding

**On a $37,500 sale the choice of lot rule moved the tax bill by $3,895 — 10.4 per cent of
proceeds.** That is larger than most alpha. It is invisible to every optimiser that models a position
as a scalar weight.

A second finding, and the reason `:tax_optimal` exists separately from `HIFO`: **HIFO minimises the
realised gain, not the tax.** When short-term and long-term rates differ, selling a high-basis
short-term lot can cost more than a lower-basis long-term one. The two rules coincide only when the
rates are equal.

The wash-sale check is symmetric, and the "before" half is the one people forget: a purchase 16 days
*before* a loss sale triggers the rule just as a repurchase after it does.

**This models mechanics, not advice.** The rules encoded are a simplified reading of the United
States federal regime. Several jurisdictions mandate an averaged basis and have no lot choice at all.

---

### 3.13 Online moments — `19_online_moments.jl`

#### Mathematics

Welford's update, whose asymmetry is the whole point:

$$
n \leftarrow n+1, \quad
d = x - \mu_{\text{old}}, \quad
\mu \leftarrow \mu_{\text{old}} + d/n, \quad
M \leftarrow M + d\,(x-\mu_{\text{new}})'
$$

Chan's merge, which makes it parallel and associative:

$$
\mu = \mu_A + \delta\frac{n_B}{n}, \qquad
M = M_A + M_B + \delta\delta'\frac{n_A n_B}{n}
$$

#### The finding

On data with mean 1000 and unit-scale spread — the case that occurs whenever prices are used instead
of returns:

```
Welford covariance error  7.37e-14
textbook covariance error 6.38e-10, which is 8653 times worse
parallel merge equals the sequential answer to 1.15e-13
```

**The first factor uses the old mean and the second uses the new one.** Using the same mean in both
gives the textbook formula, which Chan and co-authors (1983) show can return a negative variance on
real data. Welford's cannot. The speed gain is the advertised benefit; the accuracy is the real one.

---

### 3.14 Graph structure — `20_graph_structure.jl`

The library turns correlations into graphs and uses them for **constraints**. A constraint is a
wall; a penalty is a slope, and a slope is usually what is meant.

#### Mathematics

The identity the whole idea rests on:

$$
w'Lw = \tfrac{1}{2}\sum_{i,j} A_{ij}\,(w_i - w_j)^2, \qquad L = D - A
$$

The left side is a quadratic form a solver handles trivially. The right side is "penalise weight
differences between connected assets in proportion to how strongly they are connected".

#### Verified

```
w'Lw equals the pairwise sum exactly: 0.0e+00
L*1 = 0 to 1.1e-16 ; positive semi-definite ; 2 connected components (truth 2)
normalised spectrum lies in [0, 2]: true
```

| penalty `λ` | spread inside component 1 | component 2 |
| ---: | ---: | ---: |
| 0 | 0.1807 | 0.2418 |
| 100 | 0.0014 | 0.0034 |
| 1e6 | 0.0000 | 0.0000 |

#### The finding

**A Laplacian penalty is a covariance modification, not a new kind of objective**: the problem is the
ordinary one with $\Sigma_{\text{eff}} = \Sigma + 2\lambda L$, which is still positive semi-definite.
So the library needs no new optimiser, only a way to add a matrix to the covariance. And the limit
is exactly what it should be: as `λ` grows the solution becomes equal-weight **within each connected
component**, because only the null space of `L` survives.

Use the **normalised** Laplacian whenever `λ` is tuned on one universe and applied to another. Its
eigenvalues lie in `[0, 2]`; the combinatorial one's do not, so the same `λ` means something
different on a denser graph.

---

### 3.15 Regime models — `21_regime_models.jl`

The library has regime-*adjusted* moments. It has no regime **model**.

#### The design, which is the main idea

Fitting a `K`-state model directly on `N` assets needs `K·N(N+3)/2` parameters and never converges to
anything meaningful. So the design is asymmetric:

- The **state** is inferred by a hidden Markov model on **one** driver series: `3K + K²` parameters.
- The **moments** are then estimated for the whole panel, weighted by the smoothed state
  probabilities.

#### Mathematics

Forward-backward in log space, then Baum-Welch. The law of total covariance for the mixture:

$$
\Sigma = \underbrace{\sum_k p_k \Sigma_k}_{\text{within}} + \underbrace{\sum_k p_k (\mu_k - \bar\mu)(\mu_k-\bar\mu)'}_{\text{between}}
$$

**The between term is the one that gets forgotten.** It is positive semi-definite, so dropping it
always understates the risk, and it understates most exactly when the regimes differ most in mean —
which is when it matters.

#### Verified

```
log likelihood non-decreasing at every step: true (14 iterations)
true   mean [-0.0015, 0.0008]  sd [0.02, 0.006]
fitted mean [-0.00129, 0.00088]  sd [0.01965, 0.00607]
state classification accuracy 0.973
expected durations [18.6, 58.9] periods ; stationary distribution [0.24, 0.76]
```

#### The finding

**Report the expected duration, never the transition probability.** `P[1,1] = 0.98` means nothing to
a reader; "the bad state lasts 19 days on average" means something, and it is the number that decides
whether a regime model is usable for allocation at all. A state that lasts two days cannot be traded.

Also: the effective sample size of a regime is its probability mass, not `T`. The bad state here had
709 effective observations out of 3000, so its covariance is four times noisier than the good state's.
**Report the weight beside the moments and shrink hard in the rare state.**

---

### 3.16 Provenance and capability — `22_provenance_and_capability.jl`

Two ideas from report 4 that look like housekeeping. The second has teeth.

#### The finding

The library has 229 abstract types and a great many valid combinations of them. It also has invalid
ones, and today an invalid one surfaces as a `MethodError` from deep inside a solve. A declared
capability layer turns that into a sentence, before any work happens:

```
compatible: false
  ! NegativeSkewness requires sk, which nothing upstream provides. Available: X, mu, sigma.
with a high-order prior: true
```

**The value is the message, not the boolean.** The cost is one `provides` or `requires` method per
concrete type, declared by its author. It must be a **per-type declaration and never a walk over the
fields**, for exactly the reason `CONTEXT.md` gives for `deferred_slots`: the presence of a field does
not imply the semantics.

The check is **necessary and not sufficient**. It catches a missing quantity. It cannot catch one of
the wrong shape or computed on a different universe; the dimension assertions already in the library
do that.

For provenance, the three fields that matter are `data_hash`, `seeds` and `versions`. Everything else
is documentation. The timestamp is deliberately excluded from the fingerprint: **a record is
identified by what it did, not by when.**

---

## 4. Recommended sequence

Ordered by value divided by effort, using the report 8 items as phase 0.

| Phase | Work | Prototype | Size |
| :--- | :--- | :--- | :--- |
| 0 | Performance summary into `src/`; fix the DR-CVaR docstring | 06 | hours |
| 1 | Capability and compatibility layer | 22 | small |
| 1 | Welford and EWMA `partial_fit` | 19 | small |
| 1 | Wasserstein and Gelbrich ambiguity sets, calibrated | 02, 11 | small |
| 2 | Portfolio state, drift-aware turnover, frozen positions | 15 | medium |
| 2 | Rebalancing policies | 16 | small |
| 2 | Market impact as a convex cost term | 17 | medium |
| 3 | Prior sampling seam | 01 | medium |
| 3 | Conditional stress and scenario transforms | 10 | small |
| 3 | Regime models | 21 | medium |
| 4 | Optimiser sensitivity and the differentiable layer | 07 | medium |
| 4 | Shapley constraint attribution | 14 | small |
| 4 | Model population and robustness frontier | 03, 13 | small |
| 4 | Bootstrap and conformal intervals | 12 | small |
| 5 | Critical line algorithm | 08 | medium |
| 5 | Online portfolio selection | 09 | small |
| 5 | Laplacian regularisation | 20 | small |
| 5 | The breakeven view | 05 | small |
| 6 | Tax-aware lots | 18 | large |
| 6 | Calibration protocol | 04 | medium |

Phases 1 and 2 are the ones with the best ratio. Phase 6 is the one with the most differentiation.

---

## 5. One general rule, restated

`src/` is 109,665 lines and 229 abstract types. The risk to the library now is surface area, not
capability.

**Prefer an addition that is a seam over an addition that is a leaf.** A seam is one verb many types
answer: `simulate`, `partial_fit`, `provides`, `port_opt_view`, `factory`. A leaf is one more
estimator beside the forty that exist.

Counted against that rule, of the 22 prototypes: **14 are seams**, 5 are self-contained algorithms
that add no surface to existing types (08, 09, 17, 18, 21), and 3 are Results (03, 06, 13). None is a
new leaf estimator.

---

## 6. Sources

Named in prose, per the repository convention. Report 8 lists 30 sources for prototypes 1 to 6.
These are the additional ones for prototypes 7 to 22.

1. Agarwal, A., Hazan, E., Kale, S. and Schapire, R. E. (2006). Algorithms for portfolio management
   based on the Newton method. *Proceedings of the 23rd International Conference on Machine
   Learning*, 9–16.
2. Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S. and Kolter, J. Z. (2019).
   Differentiable convex optimization layers. *Advances in Neural Information Processing Systems*
   32. arXiv:1910.12430.
3. Almgren, R. and Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*
   3(2), 5–39.
4. Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005). Direct estimation of equity market
   impact. *Risk* 18(7), 58–62.
5. Amos, B. and Kolter, J. Z. (2017). OptNet: differentiable optimization as a layer in neural
   networks. *Proceedings of the 34th International Conference on Machine Learning*, 136–145.
   arXiv:1703.00443.
6. Ando, R. K. and Zhang, T. (2007). Learning on graph with Laplacian regularization. *Advances in
   Neural Information Processing Systems* 19, 25–32.
7. Ang, A. and Bekaert, G. (2002). International asset allocation with regime shifts. *Review of
   Financial Studies* 15(4), 1137–1187.
8. Bailey, D. H. and López de Prado, M. (2013). An open-source implementation of the critical-line
   algorithm for portfolio optimization. *Algorithms* 6(1), 169–196.
9. Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2014). Pseudo-mathematics and
   financial charlatanism. *Notices of the American Mathematical Society* 61(5), 458–471.
10. Barber, R. F., Candès, E. J., Ramdas, A. and Tibshirani, R. J. (2023). Conformal prediction
    beyond exchangeability. *Annals of Statistics* 51(2), 816–845.
11. Baum, L. E., Petrie, T., Soules, G. and Weiss, N. (1970). A maximization technique occurring in
    the statistical analysis of probabilistic functions of Markov chains. *Annals of Mathematical
    Statistics* 41(1), 164–171.
12. Ben-Tal, A., den Hertog, D., De Waegenaere, A., Melenberg, B. and Rennen, G. (2013). Robust
    solutions of optimization problems affected by uncertain probabilities. *Management Science*
    59(2), 341–357.
13. Berkin, A. L. and Ye, J. (2003). Tax management, loss harvesting, and HIFO accounting.
    *Financial Analysts Journal* 59(4), 91–102.
14. Best, M. J. and Grauer, R. R. (1991). On the sensitivity of mean-variance efficient portfolios
    to changes in asset means. *Review of Financial Studies* 4(2), 315–342.
15. Bhatia, R., Jain, T. and Lim, Y. (2019). On the Bures-Wasserstein distance between positive
    definite matrices. *Expositiones Mathematicae* 37(2), 165–191.
16. Borodin, A., El-Yaniv, R. and Gogan, V. (2004). Can we learn to beat the best stock. *Journal of
    Artificial Intelligence Research* 21, 579–594. arXiv:1107.0036.
17. Castro, J., Gómez, D. and Tejada, J. (2009). Polynomial calculation of the Shapley value based
    on sampling. *Computers and Operations Research* 36(5), 1726–1730.
18. Chan, T. F., Golub, G. H. and LeVeque, R. J. (1983). Algorithms for computing the sample
    variance: analysis and recommendations. *The American Statistician* 37(3), 242–247.
19. Chaudhuri, S. E., Burnham, T. C. and Lo, A. W. (2020). An empirical evaluation of tax-loss
    harvesting alpha. *Financial Analysts Journal* 76(3), 99–108.
20. Chopra, V. K. and Ziemba, W. T. (1993). The effect of errors in means, variances, and
    covariances on optimal portfolio choice. *Journal of Portfolio Management* 19(2), 6–11.
21. Chung, F. R. K. (1997). *Spectral Graph Theory*. American Mathematical Society.
22. Constantinides, G. M. (1983). Capital market equilibrium with personal tax. *Econometrica*
    51(3), 611–636.
23. Constantinides, G. M. (1986). Capital market equilibrium with transaction costs. *Journal of
    Political Economy* 94(4), 842–862.
24. Cover, T. M. (1984). An algorithm for maximizing expected log investment return. *IEEE
    Transactions on Information Theory* 30(2), 369–373.
25. Cover, T. M. (1991). Universal portfolios. *Mathematical Finance* 1(1), 1–29.
26. Dammon, R. M., Spatt, C. S. and Zhang, H. H. (2001). Optimal consumption and investment with
    capital gains taxes. *Review of Financial Studies* 14(3), 583–616.
27. Davis, M. H. A. and Norman, A. R. (1990). Portfolio selection with transaction costs.
    *Mathematics of Operations Research* 15(4), 676–713.
28. Delage, E. and Ye, Y. (2010). Distributionally robust optimization under moment uncertainty.
    *Operations Research* 58(3), 595–612.
29. Demarta, S. and McNeil, A. J. (2005). The t copula and related copulas. *International
    Statistical Review* 73(1), 111–129.
30. Donohue, C. and Yip, K. (2003). Optimal portfolio rebalancing with transaction costs. *Journal
    of Portfolio Management* 29(4), 49–63.
31. Donti, P., Amos, B. and Kolter, J. Z. (2017). Task-based end-to-end model learning in stochastic
    optimization. *Advances in Neural Information Processing Systems* 30. arXiv:1703.04529.
32. Duchi, J., Shalev-Shwartz, S., Singer, Y. and Chandra, T. (2008). Efficient projections onto the
    l1-ball for learning in high dimensions. *Proceedings of the 25th International Conference on
    Machine Learning*, 272–279.
33. Embrechts, P., McNeil, A. J. and Straumann, D. (2002). Correlation and dependence in risk
    management: properties and pitfalls. In: *Risk Management: Value at Risk and Beyond*, Cambridge
    University Press, 176–223.
34. Fiacco, A. V. (1976). Sensitivity analysis for nonlinear programming using penalty methods.
    *Mathematical Programming* 10(1), 287–311.
35. Fleming, J., Kirby, C. and Ostdiek, B. (2001). The economic value of volatility timing. *Journal
    of Finance* 56(1), 329–352.
36. Föllmer, H. and Schied, A. (2011). *Stochastic Finance: An Introduction in Discrete Time*, 3rd
    edition. De Gruyter.
37. Frazzini, A., Israel, R. and Moskowitz, T. J. (2018). Trading costs. Working paper, AQR Capital
    Management.
38. Gatheral, J. (2010). No-dynamic-arbitrage and market impact. *Quantitative Finance* 10(7),
    749–759.
39. Gelbrich, M. (1990). On a formula for the L2 Wasserstein metric between measures on Euclidean
    and Hilbert spaces. *Mathematische Nachrichten* 147(1), 185–203.
40. Grinold, R. C. and Kahn, R. N. (1999). *Active Portfolio Management*, 2nd edition. McGraw-Hill.
41. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and
    the business cycle. *Econometrica* 57(2), 357–384.
42. Helmbold, D. P., Schapire, R. E., Singer, Y. and Warmuth, M. K. (1998). On-line portfolio
    selection using multiplicative updates. *Mathematical Finance* 8(4), 325–347.
43. Huang, D., Zhou, J., Li, B., Hoi, S. C. H. and Zhou, S. (2016). Robust median reversion strategy
    for online portfolio selection. *IEEE Transactions on Knowledge and Data Engineering* 28(9),
    2480–2493.
44. Kupiec, P. H. (1998). Stress testing in a value at risk framework. *Journal of Derivatives*
    6(1), 7–24.
45. Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica* 53(6), 1315–1335.
46. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L. (2018). Distribution-free
    predictive inference for regression. *Journal of the American Statistical Association* 113(523),
    1094–1111.
47. Leland, H. E. (1999). Optimal portfolio management with transactions costs and capital gains
    taxes. Working paper RPF-290, University of California Berkeley.
48. Li, B. and Hoi, S. C. H. (2012). On-line portfolio selection with moving average reversion.
    *Proceedings of the 29th International Conference on Machine Learning*. arXiv:1206.4626.
49. Li, B. and Hoi, S. C. H. (2014). Online portfolio selection: a survey. *ACM Computing Surveys*
    46(3), article 35.
50. Li, B., Zhao, P., Hoi, S. C. H. and Gopalkrishnan, V. (2012). PAMR: passive aggressive mean
    reversion strategy for portfolio selection. *Machine Learning* 87(2), 221–258.
51. Lundberg, S. M. and Lee, S. I. (2017). A unified approach to interpreting model predictions.
    *Advances in Neural Information Processing Systems* 30. arXiv:1705.07874.
52. Mantegna, R. N. (1999). Hierarchical structure in financial markets. *European Physical Journal
    B* 11(1), 193–197.
53. Markowitz, H. M. (1956). The optimization of a quadratic function subject to linear constraints.
    *Naval Research Logistics Quarterly* 3(1–2), 111–133.
54. Martin, P. G. and McCann, B. B. (1989). *The Investor's Guide to Fidelity Funds*. Wiley.
55. Meucci, A. (2005). *Risk and Asset Allocation*. Springer.
56. Niedermayer, A. and Niedermayer, D. (2010). Applying Markowitz's critical line algorithm. In:
    *Handbook of Portfolio Construction*, Springer, 383–400.
57. Nguyen, V. A., Kuhn, D. and Mohajerin Esfahani, P. (2022). Distributionally robust inverse
    covariance estimation: the Wasserstein shrinkage estimator. *Operations Research* 70(1),
    490–515.
58. Nystrup, P., Madsen, H. and Lindström, E. (2018). Dynamic portfolio optimization across hidden
    market regimes. *Quantitative Finance* 18(1), 83–95.
59. Peng, R. D. (2011). Reproducible research in computational science. *Science* 334(6060),
    1226–1227.
60. Perold, A. F. (1988). The implementation shortfall: paper versus reality. *Journal of Portfolio
    Management* 14(3), 4–9.
61. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech
    recognition. *Proceedings of the IEEE* 77(2), 257–286.
62. RiskMetrics Group (1996). *RiskMetrics Technical Document*, 4th edition. J. P. Morgan.
63. Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M. and
    Tarantola, S. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
64. Sandve, G. K., Nekrutenko, A., Taylor, J. and Hovig, E. (2013). Ten simple rules for
    reproducible computational research. *PLoS Computational Biology* 9(10), e1003285.
65. Shapley, L. S. (1953). A value for n-person games. In: *Contributions to the Theory of Games II*,
    Princeton University Press, 307–317.
66. Stein, D. M. and Narasimhan, P. (1999). Of passive and active equity portfolios in the presence
    of taxes. *Journal of Wealth Management* 2(2), 55–63.
67. Torre, N. and Ferrari, M. (1997). *Market Impact Model Handbook*. BARRA.
68. Tumminello, M., Aste, T., Di Matteo, T. and Mantegna, R. N. (2005). A tool for filtering
    information in complex systems. *Proceedings of the National Academy of Sciences* 102(30),
    10421–10426.
69. von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and Computing* 17(4),
    395–416.
70. Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World*.
    Springer.
71. Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n points donnés est
    minimum. *Tôhoku Mathematical Journal* 43, 355–386.
72. Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products.
    *Technometrics* 4(3), 419–420.
73. Young, H. P. (1985). Monotonic solutions of cooperative games. *International Journal of Game
    Theory* 14(2), 65–72.
74. Young, T. W. (1991). Calmar ratio: a smoother tool. *Futures* 20(1), 40.

---

## 7. The prototype index

| File | Lines | Central verified claim |
| :--- | --: | :--- |
| `01_scenario_generation.jl` | 424 | Each generator reproduces its target moments, rank correlation and marginals. |
| `02_wasserstein_ambiguity.jl` | 526 | The closed form is attained exactly by the worst-case measure, for three ground metrics. |
| `03_model_population.jl` | 465 | The ambiguity decomposition holds to `7.6e-21`. |
| `04_simulated_truth_calibration.jl` | 354 | Reproduces DeMiguel–Garlappi–Uppal (2009). No estimator beats the oracle. |
| `05_breakeven_view.jl` | 434 | `D = -L(λ*)`; agrees with an independent root-find to `1.0e-11`. |
| `06_performance_summary.jl` | 287 | Matches the plot extension's five formulas; the Sharpe standard error reduces correctly. |
| `07_optimiser_sensitivity.jl` | 391 | Three analytic Jacobians match finite differences; the constraint price is exact. |
| `08_critical_line.jl` | 471 | KKT residual `1.4e-15`; never worse than a high-accuracy solver. |
| `09_online_portfolio_selection.jl` | 745 | The best constant portfolio dominates the best single asset; every strategy stays on the simplex with no look-ahead. |
| `10_conditional_stress.jl` | 593 | Conditional moments match 200k Monte Carlo; t-copula tail dependence converges to theory. |
| `11_moment_and_divergence_ambiguity.jl` | 482 | The Kullback-Leibler dual binds at exactly `η`; brute force agrees to `1.8e-4`. |
| `12_weight_uncertainty.jl` | 390 | Conformal coverage lands inside `[1-α, 1-α+1/(n+1)]` for every `α`. |
| `13_robustness_frontier.jl` | 355 | `R(1/N) = 1` exactly, and `R` rises monotonically with shrinkage. |
| `14_attribution.jl` | 385 | Shapley efficiency to `1.4e-20`; Euler identity to `8.7e-19`. |
| `15_portfolio_state.jl` | 368 | Budget arithmetic exact; frozen weights preserved bit for bit. |
| `16_rebalancing_policies.jl` | 436 | `threshold = 0` reproduces `always`; the hybrid rule wins on the net figure. |
| `17_market_impact.jl` | 396 | `2^1.5` exactly; the Almgren-Chriss path beat 400 of 400 perturbations. |
| `18_tax_aware.jl` | 466 | Shares conserved, input never mutated, wash sales caught in both directions. |
| `19_online_moments.jl` | 344 | Welford matches batch to `7.4e-14`; the merge is exact and associative. |
| `20_graph_structure.jl` | 334 | The Laplacian identity is exact at `0.0e+00`. |
| `21_regime_models.jl` | 496 | The log likelihood never decreases; 97.3 per cent state accuracy on simulated truth. |
| `22_provenance_and_capability.jl` | 350 | Fingerprints stable, sensitive and transpose-aware; the check yields an actionable sentence. |
| `run_all.jl` | 241 | Reproduces every number in report 8. |
| `run_novel.jl` | 620 | Reproduces every number in this report. |

**Total: 10,353 lines.**
