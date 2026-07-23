---
status: accepted
---

# The ucs / mu_ucs / sigma_ucs extraction shares post-sampling kernels, not `mu_half`/`sigma_half`

## Context

Every uncertainty-set estimator implements a triple: `ucs` (returns the μ and Σ sets as a
pair), `mu_ucs` (μ only), and `sigma_ucs` (Σ only). Each `ucs` body was literally its
`mu_ucs` body concatenated with its `sigma_ucs` body — the Wishart/bootstrap quantile loops
and the ellipsoid-fitting tail were re-encoded across
[02_Delta](../../src/14_UncertaintySets/02_DeltaUncertaintySets.jl),
[03_Normal](../../src/14_UncertaintySets/03_NormalUncertaintySets.jl), and
[04_Bootstrap](../../src/14_UncertaintySets/04_BootstrapUncertaintySets.jl). The copies had
drifted twice in eleven days (a shipped posdef bug on the Normal-box μ half, then a cosmetic
`eltype(sigma)`/`N` vs `eltype(sigma_mu)`/`size(pr.X,2)` divergence on the same μ lower bound).
The 2026-07-18 maintainability review flagged this and proposed extracting per-estimator
`mu_half(ue, pr)` / `sigma_half(ue, pr)` helpers so each triple member collapses to a
one-liner.

We rejected that specific shape because it is **not value-preserving**:

1. **Threaded rng.** For the Normal-ellipsoidal `NormalKUncertaintyAlgorithm` path, `ucs`
   draws `MvNormal` (μ) and then `Wishart` (Σ) from *one* seeded rng sequence, whereas
   standalone `sigma_ucs` re-seeds before its Wishart draw. So `ucs`'s Σ half already differs
   from standalone `sigma_ucs` (test_10 stores both separately and deliberately does not assert
   they are equal). A `mu_half`/`sigma_half` split where each half seeds itself would change
   `ucs`'s Σ numbers and force regenerating the seeded reference CSV.
2. **Double sampling.** For `ARCHUncertaintySet`, `ucs` calls `bootstrap_generator` once,
   producing μ and Σ from a single resample. A `mu_half`/`sigma_half` split would call
   `mu_bootstrap_generator` and `sigma_bootstrap_generator` — two full resample loops, ≈2× the
   `ucs` cost.

## Decision

Keep `prior`, seeding, and sampling inside each public `ucs` / `mu_ucs` / `sigma_ucs` body
(so `ucs` keeps its single combined sample and its μ-draw-then-Σ-draw order). Extract only the
**pure post-sampling construction** into shared helpers:

- `ellipsoidal_set(diagonal, method, q, samples, cov, class)` in
  [01_Base](../../src/14_UncertaintySets/01_Base_UncertaintySets.jl) — the diagonalise + fit-`k`
  - tag tail, shared by every ellipsoidal variant across Normal and ARCH (~13 sites). Works
  uniformly because `k_ucs` already absorbs unused trailing arguments, so `samples` is whatever
  the variant already passed (`X_mu`/`X_sigma`, a `1:n_sim` range, or `nothing`).
- `box_quantile_bounds(TE, get_ij, N, q, kwargs)` and `vec_quantile_bounds(mus, q, kwargs)` in
  01_Base — the symmetric / vector quantile fills; an accessor closure bridges the Wishart
  (`Vector`-of-matrices) and bootstrap (3-D array) containers, and positive-definite projection
  stays in the caller (Normal applies it, ARCH does not).
- `mu_normal_box_set` (03_Normal) and `mu_delta_box_set` / `sigma_delta_box_set` (02_Delta) for
  the per-estimator analytic μ/Σ formulas that genuinely differ between families.

The drifted μ lower bound is canonicalised by construction: both `ucs` and `mu_ucs` now call the
same `mu_normal_box_set`, so the family cannot drift again.

## Consequences

- The refactor is value-identical: test_10 stays green with zero edits and **no reference-CSV
  regeneration** (`BoxUncertaintySet.csv.gz`, `EllipsoidalUncertaintySet.csv.gz` unchanged).
- The seam is intentionally finer than the report's `mu_half`/`sigma_half`: `ucs` bodies stay
  multi-line (they own sampling), only the duplicated tails collapse. A future reader holding
  the maintainability report should read this ADR before "finishing the job" into halves — that
  shape trades away numeric preservation and ARCH performance.
- Sits inside [ADR 0016](0016-uncertainty-sets-own-their-posdef-and-moment-config.md) (each
  uncertainty set owns its posdef/moment config); it does not reopen it.
