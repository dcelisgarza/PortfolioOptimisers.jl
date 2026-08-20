# PortfolioOptimisers.jl — Suggested Issues

Notes from reading the source (v0.24–0.26, cloned from `dcelisgarza/PortfolioOptimisers.jl`)
and the project's Discourse announcement thread. Written as separate, independently-postable
issues. Priority ordering reflects effort vs. value, not importance.

---

## Issue 1 — DR-CVaR family: docstring/implementation mismatch, missing citation, and a

reusable Wasserstein ambiguity primitive

**Priority: high (docs), medium (refactor).**

### Verdict on the existing implementation

`DistributionallyRobustConditionalValueatRisk` (and its Range/CDaR siblings) are **not** a toy
closed-form bound — the JuMP constraint builder in
`src/20_Optimisation/20_RiskMeasureConstraints/07_ConditionalXatRiskConstraints.jl` implements a
genuine tractable reformulation of the Wasserstein-ball worst-case CVaR: a dual multiplier `lb`
(λ), per-sample epigraph slacks `s`, and per-sample dual variables `u`/`v` constrained via
`JuMP.MOI.NormInfinityCone`. Structurally this matches the Mohajerin Esfahani & Kuhn (2018)
piecewise-affine-loss reformulation — the same one skfolio's
[`DistributionallyRobustCVaR`](https://skfolio.org/_modules/skfolio/optimization/convex/_distributionally_robust.html)
([example](https://skfolio.org/auto_examples/distributionally_robust_cvar/plot_1_distributionally_robust_cvar.html))
and MOSEK's Fusion tutorial notebook
`Data-driven_distributionally_robust_portfolio.ipynb` (in MOSEK's own `Tutorials` repository)
implement — extended here to also cover the two-sided range and CDaR, which neither of those does.
This is a solid, correct-looking implementation.

Three gaps found while reading it closely:

### 1. Docstring math doesn't match the code

`src/19_RiskMeasures/07_ConditionalXatRisk.jl` documents the "Mathematical definition" of DR-CVaR
as the closed form:

```text
DR-CVaR = CVaR + l·r
```

This is not what the JuMP builder computes — it solves an LP with `T×N` auxiliary variables and
infinity-norm cone constraints (see lines ~195–297 of the constraints file). Either:

- the docstring should show the actual dual LP, ideally using the paper's notation (`a_k`, `b_k`,
  λ) so `l`'s role as a piecewise-affine-loss coefficient is clear instead of reading like a free,
  user-chosen Lipschitz constant, or
- if `CVaR + l·r` is a valid special-case bound of the LP, the docstring should say when it
  applies and how it relates to what's actually solved.

### 2. No citation

Neither `07_ConditionalXatRisk.jl` nor `07_ConditionalXatRiskConstraints.jl` cites Mohajerin
Esfahani & Kuhn (2018), *"Data-driven distributionally robust optimization using the Wasserstein
metric: performance guarantees and tractable reformulations"* (Mathematical Programming), which
appears to be the source of the reformulation. The `@cite` machinery is already used elsewhere in
the codebase (e.g. `05_L1UncertaintySets.jl`) — this would just be applying the existing
convention.

For a worked reference implementation to cross-check the docstring rewrite against, MOSEK's Fusion
tutorial notebook implements the same paper's numerical experiments end-to-end, including the
exact piecewise-linear CVaR encoding (`a_k`, `b_k` coefficient lists) and Wasserstein-radius
parameterization that the JuMP code here mirrors:
`Data-driven_distributionally_robust_portfolio.ipynb`, in MOSEK's own `Tutorials` repository
(companion [blog post](https://themosekblog.blogspot.com/2021/04/data-driven-distributionally-robust.html)
with the paper's authors walking through it).

### 3. Wasserstein ambiguity isn't a reusable `UncertaintySet`

`src/14_UncertaintySets/` has a clean, composable `Box`/`Ellipse` × `Delta`/`Normal`/
`ARCH-bootstrap` hierarchy (plus a cross-polytope ℓ1 set), usable by any optimizer that accepts
mean/covariance uncertainty. Wasserstein ambiguity, by contrast, is hard-coded into three specific
risk-measure structs, each carrying its own bespoke LP encoding. There's no `WassersteinUncertaintySet`
that plugs into robust mean/covariance the way Blanchet, Chen & Zhou (2022, *Management Science*)
do for mean-variance.

**Feature request**: factor the Wasserstein-ball machinery into a reusable uncertainty-set type
that plugs into the existing uncertainty-set-consuming optimizers, with the CVaR/CVaR-Range/CDaR
risk measures becoming thin consumers of it rather than each carrying its own LP encoding. This
would also open the door to Wasserstein-robust variance / worst-case Sharpe without duplicating
the dual-cone plumbing per risk measure.

### 4. (Minor) Evaluation functor silently drops DR semantics

`RMCVaR{T}` unions `ConditionalValueatRisk` and `DistributionallyRobustConditionalValueatRisk`
onto the same evaluation functor, which only uses `alpha` — `l` and `r` are ignored when the risk
measure is called directly on a realized returns vector (as opposed to inside a JuMP model). This
may be intentional (DR-CVaR isn't well-defined pointwise without the dual formulation), but it's a
sharp edge worth a docstring note either way — someone using `DistributionallyRobustConditionalValueatRisk`
to compute post-hoc portfolio statistics or in `plot_risk_contribution` will silently get plain
CVaR back.

**Suggested priority**: #1 and #2 are quick documentation fixes. #3 is a larger but valuable
architectural change given how much of the library is already built around swappable
`UncertaintySet` types. #4 is a one-line clarification either way.

---

## Issue 2 — Backtest engine + performance tearsheet

**Priority: high — matches the maintainer's own stated roadmap ("Report generation," "better
plots," item 3 in the v0.14.2 release notes)**

Cross-validation (`WalkForwardEstimator`, `CombinatorialCrossValidation`,
`MultipleRandomisedEstimator`) and time-dependent constraints/optimisers already give the library
genuine out-of-sample validation machinery — this is not a request to add that. What's still
missing is the layer *on top* of a completed walk-forward run: a rebalancing-calendar-aware
simulation object that turns a sequence of `optimise()` calls into a single wealth path with
realistic frictions, plus a standard report generated from it.

For reference, the shape of this layer in the Python ecosystem: [cvxgrp/cvxportfolio](https://github.com/cvxgrp/cvxportfolio)
(multi-period trading with transaction costs, from the Stanford Boyd group), and the general
event-driven backtest architecture of [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded)
and [mementum/backtrader](https://github.com/mementum/backtrader) — a rebalance calendar, order
execution against next-bar prices, slippage/impact model, and a results object independent of the
optimisation loop itself.

Concretely:

- A `Backtest`/`Simulation` result type that consumes a walk-forward prediction (the library
  already produces a concatenated portfolio-returns series `R` from pipelines) and adds:
  transaction costs and slippage per rebalance, a fixed or calendar-based rebalancing schedule
  independent of the CV fold boundaries, and benchmark-relative tracking over the full simulated
  history.
- A one-call tearsheet: Sharpe/Sortino/Calmar, rolling volatility and Sharpe, drawdown table,
  turnover, and the existing cumulative-returns/composition/drawdown plots assembled into a single
  report (HTML or a Plots.jl multi-panel figure). This is explicitly on the roadmap already
  ("Report generation," "Perhaps excel output") — this issue is really just asking for it to stay
  prioritized and suggesting the shape (reuse the existing plotting functions as panels rather than
  building new ones). [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) is the
  reference point — `quantstats.reports` generates exactly this kind of one-call HTML tear sheet
  from a returns series. [robertmartin8/PyPortfolioOpt's cookbook](https://github.com/robertmartin8/PyPortfolioOpt/tree/main/cookbook)
  is also a good model for presentation, not of the report itself but of documenting the full
  pipeline (data → estimation → optimisation → evaluation) as runnable notebooks rather than
  per-feature API docs — worth considering alongside the tearsheet as a documentation format.

Given the release notes mention plotting has been the sore point (Makie's dendrogram support,
GraphRecipes/StatsPlots as an extension dependency, wanting Plotly-style mouseover but not wanting
to lock into it) — for a *tearsheet* specifically, a static, non-interactive HTML/PNG panel export
would sidestep that whole fight and could ship well before the general interactive-plots question
is resolved.

---

## Issue 3 — Online portfolio selection module

**Priority: medium — the maintainer raised this themselves ("online statistical methods maybe?")
and talked themselves out of it; worth revisiting**

The v0.19 release notes ask "I'm not sure how to make most of the current [estimators] updated" —
that's the right instinct for retrofitting the existing batch estimators, but online portfolio
selection (OLPS) is usually a separate family of algorithms, not an online update of the existing
moment/covariance estimators. A small, self-contained module implementing the standard OLPS
benchmarks would sit next to the existing `Naive` optimisers (inverse volatility, equal-weighted,
random-weighted) rather than requiring any change to priors/moments:

- **Universal Portfolio** — Cover, T. (1991), "Universal Portfolios," *Mathematical Finance* 1(1).
  Pre-arXiv; standard journal citation only.
- **Exponentiated Gradient** — Helmbold, D.P., Schapire, R.E., Singer, Y. & Warmuth, M.K. (1998),
  "On-Line Portfolio Selection Using Multiplicative Updates," *Mathematical Finance* 8(4).
  Pre-arXiv; standard journal citation only.
- **OLMAR** — Li, B. & Hoi, S.C.H. (2012), "On-Line Portfolio Selection with Moving Average
  Reversion," ICML. [arxiv.org/abs/1206.4626](https://arxiv.org/abs/1206.4626)
- **RMR** (Robust Median Reversion) — Huang, D., Zhou, J., Li, B., Hoi, S.C.H. & Zhou, S.
  (2013/2016), "Robust Median Reversion Strategy for On-Line Portfolio Selection," IJCAI '13 /
  *IEEE TKDE* 28(9). [ijcai.org/Abstract/13/296](https://www.ijcai.org/Abstract/13/296)
- **Anticor** — Borodin, A., El-Yaniv, R. & Gogan, V. (2004), "Can We Learn to Beat the Best
  Stock," *JAIR* 21. [arxiv.org/abs/1107.0036](https://arxiv.org/abs/1107.0036)

These are sequential, no-forecast-needed strategies that are standard reference points in the OLPS
literature and, as far as I can tell, aren't implemented in any Julia package. They'd be genuinely
cheap to add relative to the JuMP-based machinery already in the library, and would give the
walk-forward CV infrastructure a natural, very-fast baseline to compare against.

For general background and comparison tables across the OLPS family, see also: Li, B. & Hoi,
S.C.H. (2014), "Online Portfolio Selection: A Survey," *ACM Computing Surveys* 46(3),
[doi.org/10.1145/2512962](https://doi.org/10.1145/2512962) — a widely cited survey covering
Universal Portfolio, EG, OLMAR, RMR, Anticor, PAMR, CWMR, and CORN in one place, useful as a single
citation anchor for the whole module rather than five separate papers. The same authors maintain a
reference MATLAB/Octave implementation, [OLPS/OLPS](https://github.com/OLPS/OLPS), which is worth
checking against for numerical correctness when porting any of these algorithms.

---

## Issue 4 — Wasserstein radius / DR-CVaR hyperparameter guidance

**Priority: low, small — pairs naturally with Issue 1.**

`r` (Wasserstein radius) and `l` in `DistributionallyRobustConditionalValueatRisk` are free
user-supplied constants with no calibration guidance, unlike the ℓ1 uncertainty set's
`ActiveAssetsUncertaintyAlgorithm`, which converts an interpretable quantity (target number of
active assets) into the right radius. Given the hyperparameter-tuning machinery
(`GridSearchCrossValidation`/`RandomisedSearchCrossValidation`) already exists, this may just be a
documentation/example gap rather than a code gap — a worked example showing `r` swept via the
existing CV tooling would likely resolve most of the need. Worth confirming whether that combination
is already possible end-to-end before treating it as a feature request.

---

## Issue 5 — Broker / paper-trading connectivity

**Priority: low — larger scope change, only worth it if the library wants to grow beyond a
research tool**

Every optimiser currently ends at portfolio weights or (via `DiscreteAllocation`/
`GreedyAllocation`) a discrete share allocation. A thin adapter layer for order routing — even just
paper trading via a broker API (Alpaca, Interactive Brokers) — would let people run the output of
an optimisation without hand-wiring the execution step themselves. This is a genuinely different
kind of dependency (network/auth/live-market-state) from everything else in the library, so it may
be better suited as a separate package (`PortfolioOptimisersLive.jl` or similar) that depends on
this one, rather than living in the core repo.

---

## Issue 6 — Python interoperability wrapper

**Priority: low, high-leverage if pursued.**

The Julia quant-finance community is much smaller than Python's. A thin `PythonCall`/`juliacall`-based
wrapper package exposing the pipeline API (`fit`, `optimise`, cross-validation, prediction) to
Python would meaningfully expand the user base without touching the core codebase — similar to how
`diffrax`/`Symbolics.py`-style wrappers expose Julia packages to Python users. Given v0.24 just
removed the *inbound* Python dependency (`astropy`/`arch` via `PythonCall`/`CondaPkg`), this would
be the mirror image: making the library callable *from* Python instead of calling *into* it. Best
suited as a separate thin package, not a core-repo change.

---

## Issue 7 — GPU / sparse solver support for large universes

**Priority: low — matters only past a few hundred assets.**

Most examples in the docs run at ~20–25 assets. For institutional-scale universes (thousands of
names), sparse covariance handling and a GPU-accelerated conic solver path (e.g. via
[exanauts/CUDSS.jl](https://github.com/exanauts/CUDSS.jl), a Julia interface to NVIDIA's cuDSS
sparse direct solver, through the existing `Solver`/JuMP abstraction) would matter. This is an area
where Julia has a structural advantage over the Python competition if exploited, since the JuMP
layer already abstracts over solver choice — this would mainly be about validating/documenting
that the existing covariance-denoising and clustering machinery scales, rather than new modelling
code.

---

## Issue 8 — Dedicated ESG / factor-tilt constraint helpers

**Priority: low — mostly a convenience layer over what already exists.**

The existing linear/group constraint machinery (`AssetSets`, group cardinality, equation parsing)
can already express ESG-style scoring constraints, but a purpose-built module — named constraint
helpers plus a common data schema for per-asset ESG/factor scores — would lower the barrier for
that increasingly common use case without requiring users to hand-roll the equation strings
themselves.

---

## Appendix — Repos, papers, and tools referenced above

### Comparison libraries

- [robertmartin8/PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) — cookbook at
  [/cookbook](https://github.com/robertmartin8/PyPortfolioOpt/tree/main/cookbook)
- [dcajasn/Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)
- [dcajasn/Riskfolio.jl](https://github.com/dcajasn) — same author's in-development Julia port of
  Riskfolio-Lib; listed on his GitHub profile as "now in development." Worth being aware of as
  either prior art or a potential point of overlap/collaboration in the Julia ecosystem.
- [skfolio/skfolio](https://github.com/skfolio/skfolio) — DR-CVaR source at
  [`_distributionally_robust.py`](https://skfolio.org/_modules/skfolio/optimization/convex/_distributionally_robust.html)

### Multi-period optimisation / backtesting

- [cvxgrp/cvxportfolio](https://github.com/cvxgrp/cvxportfolio)
- [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded)
- [mementum/backtrader](https://github.com/mementum/backtrader)

### Reporting

- [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats)

### GPU / sparse solvers

- [exanauts/CUDSS.jl](https://github.com/exanauts/CUDSS.jl)

### Wasserstein DRO — papers and reference implementations

- Mohajerin Esfahani, P. & Kuhn, D. (2018), "Data-driven distributionally robust optimization
  using the Wasserstein metric: performance guarantees and tractable reformulations,"
  *Mathematical Programming* 171(1–2), 115–166.
- Rockafellar, R.T. & Uryasev, S. (2000), "Optimization of Conditional Value-at-Risk," *Journal of
  Risk* 2, 21–42.
- Blanchet, J., Chen, L. & Zhou, X.Y. (2022), "Distributionally Robust Mean-Variance Portfolio
  Selection with Wasserstein Distances," *Management Science* 68(9).
- `Data-driven_distributionally_robust_portfolio.ipynb`, in MOSEK's own `Tutorials` repository —
  worked Fusion-API implementation of the Esfahani-Kuhn paper's numerical experiments, using the
  same `a_k`/`b_k` piecewise-affine-loss/λ/dual-cone structure as PortfolioOptimisers.jl's JuMP
  code. Companion [blog post](https://themosekblog.blogspot.com/2021/04/data-driven-distributionally-robust.html)
  with the paper's authors.

### Online Portfolio Selection — papers and reference implementation

- Cover, T. (1991), "Universal Portfolios," *Mathematical Finance* 1(1). (pre-arXiv, journal only)
- Helmbold, D.P., Schapire, R.E., Singer, Y. & Warmuth, M.K. (1998), "On-Line Portfolio Selection
  Using Multiplicative Updates," *Mathematical Finance* 8(4). (pre-arXiv, journal only)
- Li, B. & Hoi, S.C.H. (2012), "On-Line Portfolio Selection with Moving Average Reversion," ICML.
  [arxiv.org/abs/1206.4626](https://arxiv.org/abs/1206.4626)
- Huang, D., Zhou, J., Li, B., Hoi, S.C.H. & Zhou, S. (2013/2016), "Robust Median Reversion
  Strategy for On-Line Portfolio Selection," IJCAI '13 / *IEEE TKDE* 28(9).
  [ijcai.org/Abstract/13/296](https://www.ijcai.org/Abstract/13/296)
- Borodin, A., El-Yaniv, R. & Gogan, V. (2004), "Can We Learn to Beat the Best Stock," *JAIR* 21.
  [arxiv.org/abs/1107.0036](https://arxiv.org/abs/1107.0036)
- Li, B. & Hoi, S.C.H. (2014), "Online Portfolio Selection: A Survey," *ACM Computing Surveys*
  46(3). [doi.org/10.1145/2512962](https://doi.org/10.1145/2512962)
- [OLPS/OLPS](https://github.com/OLPS/OLPS) — the authors' own MATLAB/Octave reference toolbox,
  useful for numerical cross-checking when porting these algorithms.

### Primary source (this repo)

- [dcelisgarza/PortfolioOptimisers.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl)
- [Julia Discourse announcement thread](https://discourse.julialang.org/t/ann-portfoliooptimisers-jl-ape-together-strong/133099)
