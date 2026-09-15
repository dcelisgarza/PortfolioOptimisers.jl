---
status: proposed
---

# The vine engine is a weak dependency, and the library owns the fit above the vine object

## Context

Map [#1080](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1080) builds a vine-copula
synthetic-data prior on VineCopulas.jl, with Copulas.jl underneath it for the pair-copula families.
Decision ticket [#1085](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1085) had to
settle two things before any build ticket could start: whether the two packages are direct or weak
dependencies, and which layer of the vine fit this library writes itself.

The facts the ruling rests on were measured on 2026-09-15 by
[#1082](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1082) and
[#1084](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1084), and re-read in the two
packages' sources during the ticket:

- **The released pair does not load.** VineCopulas.jl 0.1.2 declares `Copulas = "0.1.40"` with no
  upper cap below `0.2`, so the resolver takes Copulas.jl 0.1.43, whose `SurvivalCopula` lost a
  type parameter, and VineCopulas.jl fails to precompile. Copulas.jl 0.1.42 loads, but every fit
  whose family set holds the Student-t copula crashes in the t-quantile. Only Copulas.jl 0.1.41
  works. Upstream tracks the migration but has not capped the released version.
- **A direct dependency costs every user.** Fifteen packages join the manifest; `Optim = "2"`
  drops Optim 1.13 from the resolvable set; VineCopulas.jl's `LogExpFunctions = "0.3"` and
  `StatsFuns = "1.5.2"` downgrade two packages in this library's manifest; and Copulas.jl is
  rewriting its fitted-model API ahead of a 1.0.
- **VineCopulas.jl fits by sequential maximum likelihood only.** It has no Kendall-tau inversion,
  no observation weights, no automatic truncation, no threads, no conditional sampling, and its
  tree criterion is `:tau` or `:rho` on the plain rank correlation. At d = 20 and T = 2263 the
  maximum-likelihood fit with the Student-t family took 30–37 minutes; the tau-inversion route
  without it took 15 seconds. Copulas.jl's tau inversion for the Gaussian copula is a Nelder–Mead
  search 5e-5 off `sin(πτ/2)`, and it has none for the Student-t copula.
- **What the package delegates cleanly.** A supplied `RVineStructure` is honoured edge for edge.
  Any `Copulas.Copula{2}` that answers `logpdf` and `condition` is a valid edge by the package's
  own pair-copula contract. `hfunc1`, `hfunc2`, `hinv1`, `hinv2`, `rand`, `logpdf`, `rosenblatt`,
  `inverse_rosenblatt` and `simulate_qmc` run on any vine built from `RVineCopula(structure,
  edges)`. A truncated general R-vine samples through the package's execution-plan method, which
  bypasses a guard that says it cannot.
- **The reference implementation** owns its structure selection (a maximum spanning tree per tree
  on `|τ| + 1e-5` with a central-asset boost) and its pair estimation (tau inversion by default,
  maximum likelihood on request, an independence gate on the Kendall-tau p-value), and uses its
  copula families only for densities and h-functions.
- **The one extension precedent** is `PortfolioOptimisersPlotsExt` (ADR 0042 for the weak
  dependency, ADR 0072 for the complexity gate reading `ext/`): the core declares a bare verb with
  its docstring, and the extension defines the methods. No extension defines a type.

## Decision

**VineCopulas.jl and Copulas.jl are weak dependencies behind one extension,
`PortfolioOptimisersVineCopulasExt`, triggered by both.** The core's `[compat]` carries
`VineCopulas = "0.1.2"` and, until upstream caps or fixes the released pair,
`Copulas = "0.1.41 - 0.1.41"`; a `[compat]` entry on a weak dependency binds whenever the user's
environment holds the package, so this forces the working pair without depending on it. The bound
is widened in the commit that measures a fixed release.

**The library owns the fit above the vine object.** In this library's own code: the marginal
candidates and their fits (Distributions.jl and Optim.jl are direct dependencies; the
distributions the map names have no `fit_mle`), the pseudo-observations, the log transform and its
Jacobian, structure selection (the maximum spanning tree per tree, the central-asset boost, a
cluster constraint that builds tree 1 as within-cluster trees joined across clusters, and the
observation weights), pair estimation (tau inversion by default with the closed forms, per-edge
maximum likelihood on request), the independence gate, and truncation at a fixed depth or at a
depth the data selects. Delegated to VineCopulas.jl: the vine object, `hfunc1`/`hfunc2`/`hinv1`/
`hinv2` for the tree-by-tree pseudo-observations, `rand`, `logpdf`, the Rosenblatt pair and quasi-
Monte-Carlo sampling. The extension assembles `RVineCopula(structure, edges)` from what the library
selected.

**The estimator and its Results live in `src/`, and every field of theirs is bound on this
library's own types.** Pair-copula families, rotations, structure and truncation algorithms are the
library's vocabulary; marginal candidates are `Distributions.UnivariateDistribution` types. The
fitted vine sits on the Result behind one abstract type of the library's, and **the extension owns
exactly one concrete subtype of it**, wrapping the `RVineCopula` and the fitted marginals. The
estimator therefore constructs without the extension, and `prior` on it without the extension
throws the stub message of the ADR 0042 pattern. No docstring names the packages' types.

**Each owned piece is a loan.** When VineCopulas.jl gains the knob — tau inversion including the
Student-t, weights, a pluggable tree criterion with a grouping constraint, automatic truncation,
conditional sampling, threads — the extension swaps its own routine for the package's `fit` behind
the same seam. The library's implementation is the basis of the upstream proposal: once a piece
works here, the maintainer posts an issue upstream that carries the API, the implementation sketch
and an offer to contribute it. Three issues need no implementation and are drafted first: a
registry compat cap on the released pair, the stale truncated-R-vine guard and changelog line, and
a seedable `simulate_qmc`. The library never posts in, or links to, the upstream repositories
itself.

## Considered options

- **Direct dependencies.** Exact field bounds on the package types, at the cost above for every
  user who never draws a vine, plus a hard `Copulas` pin the whole library carries and the
  Copulas.jl 1.0 churn landing in the library's own resolve. Rejected: ADR 0042's default stands,
  and nothing in the ruling needs an exact foreign bound in the core.
- **Copulas.jl direct, VineCopulas.jl weak.** The pair families, `condition` and the
  goodness-of-fit test always available. Rejected: it carries most of the dependency closure for
  a surface only the vine prior reads.
- **Delegate the whole fit** to `fit(RVineCopula, U; structure = …)`. Rejected: maximum likelihood
  only, so 30–37 minutes at d = 20 with the Student-t family, no weights, no parity with the
  reference implementation's tau-inversion fixtures, no per-edge hooks for truncation selection.
- **Define the estimator inside the extension.** Exact foreign bounds, but no binding, no API page
  and no export until the package is loaded. Rejected.
- **Core estimator with `Any`-bounded slots** holding the packages' family types and structures.
  Rejected: no vocabulary of the library's own, and the estimator needs the package to be filled.
- **Wait for an upstream release** that loads with Copulas.jl 0.1.43 before merging. Rejected: the
  map would block on an external release date; the compat bound achieves the same today.

## Consequences

- The build tickets of map #1080 write the tree loop, the tau inversions, the weighted criteria,
  the independence gate and the truncation selector in `ext/PortfolioOptimisersVineCopulasExt.jl`,
  and the estimator, Results, marginal fits, transforms and structure vocabulary in `src/`.
- The test and docs environments add `VineCopulas` and `Copulas` with the same bound; the
  `sweep_manifest` gains rows for the new files; the complexity gate reads the extension.
- A future session that sees the compat bound and wonders why finds this ADR; the bound is not a
  preference but a measured precompile failure.
- Cluster-aware structure selection joins the central-asset boost as the second structure prior of
  the map; its defaults are the next decision ticket's.
