# PortfolioOptimisers.jl — Suggested Issues

Notes from reading the source (v0.24–0.26, cloned from `dcelisgarza/PortfolioOptimisers.jl`)
and the project's Discourse announcement thread. Written as separate, independently-postable
issues. Priority ordering reflects effort vs. value, not importance.

---

## Issue 1 — DR-CVaR family: docstring/implementation mismatch, missing citation, and a

reusable Wasserstein ambiguity primitive

### 3. Wasserstein ambiguity isn't a reusable `UncertaintySet`

`src/11_UncertaintySets/` has a clean, composable `Box`/`Ellipse` × `Delta`/`Normal`/
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

**Suggested priority**: #3 is a larger but valuable architectural change given how much of the
library is already built around swappable `UncertaintySet` types.

---

## Issue 2 — Backtest engine + performance tearsheet

**Priority: high — matches the maintainer's own stated roadmap ("Report generation," "better
plots," item 3 in the v0.14.2 release notes)**

Cross-validation (`WalkForwardEstimator`, `CombinatorialCrossValidation`,
`MultipleRandomisedEstimator`) and time-dependent constraints/optimisers already give the library
genuine out-of-sample validation machinery — this is not a request to add that. What's still
missing on top of a completed walk-forward run is a rebalancing schedule and slippage of its own,
plus a report export.

For reference, the shape of this layer in the Python ecosystem: cvxportfolio
(multi-period trading with transaction costs, from the Stanford Boyd group), and the general
event-driven backtest architecture of zipline-reloaded
and backtrader — a rebalance calendar, order
execution against next-bar prices, slippage/impact model, and a results object independent of the
optimisation loop itself.

Concretely:

- Slippage per rebalance, and a fixed or calendar-based rebalancing schedule independent of the
  CV fold boundaries.
- A tearsheet export: a drawdown table, and the existing summary and plots assembled into a
  single HTML report. quantstats is the
  reference point — `quantstats.reports` generates exactly this kind of one-call HTML tear sheet
  from a returns series.

Given the release notes mention plotting has been the sore point (Makie's dendrogram support,
GraphRecipes/StatsPlots as an extension dependency, wanting Plotly-style mouseover but not wanting
to lock into it) — for a *tearsheet* specifically, a static, non-interactive HTML/PNG panel export
would sidestep that whole fight and could ship well before the general interactive-plots question
is resolved.

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
CUDSS.jl, a Julia interface to NVIDIA's cuDSS
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

- oracle 2
- oracle 2's in-development Julia port. Worth being aware of as either prior art or a potential point
  of overlap/collaboration in the Julia ecosystem.

### Multi-period optimisation / backtesting

- cvxportfolio
- zipline-reloaded
- backtrader

### Reporting

- quantstats

### GPU / sparse solvers

- CUDSS.jl

### Wasserstein DRO — papers and reference implementations

- Blanchet, J., Chen, L. & Zhou, X.Y. (2022), "Distributionally Robust Mean-Variance Portfolio
  Selection with Wasserstein Distances," *Management Science* 68(9).

### Primary source (this repo)

- [dcelisgarza/PortfolioOptimisers.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl)
- [Julia Discourse announcement thread](https://discourse.julialang.org/t/ann-portfoliooptimisers-jl-ape-together-strong/133099)
