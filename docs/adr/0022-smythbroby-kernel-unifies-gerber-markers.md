---
status: accepted
---

# A data-carrying `SmythBrobyKernel` folds the Gerber/Smyth–Broby marker types into one algorithm

## Context

[06_SmythBrobyCovariance.jl](../../src/08_Moments/06_SmythBrobyCovariance.jl) and
[35_GerberIQCovariance.jl](../../src/08_Moments/35_GerberIQCovariance.jl) implemented the
Gerber / Smyth–Broby family of comovement covariances as roughly a dozen concrete "marker" types —
`Gerber0/1/2`, `SmythBroby0/1/2`, `SmythBrobyGerber0/1/2`, `SmythBrobyCount0/1/2` — each with its
own full pairwise-accumulation loop. The types encode two orthogonal axes:

- a **normalisation variant** (`0` = raw counts, `1` = includes a neutral-zone term, `2` =
  standardised comovement), shared across every family; and
- a **scoring family** (plain Gerber threshold counts vs Smyth–Broby delta/gerber/count scoring).

Because each of the ~12 markers carried a near-complete copy of the double loop, the file was ~980
lines of heavily duplicated accumulation code. Any change to the shared comovement mechanics had to
be replicated a dozen times. (This is the refactor flagged speculatively as "arch-review item 6" —
"fold Gerber/SmythBroby ~12 marker types into one data-carrying algorithm struct".)

## Decision

**Separate the two axes into dispatch, and carry the per-run scoring state in one
`SmythBrobyKernel` struct that a single comovement loop drives.**

- The **normalisation variant** becomes three `Union` aliases cutting *across* the families —
  `GerberComovementZero`, `GerberComovementOne`, `GerberComovementTwo` — with one
  `comovement_ratio(::GerberComovement{Zero,One,Two}, …)` method each and a
  `standardise_comovement!` method for the `Two` case. The `0/1/2` behaviour is written once and
  reused by every family.
- The **scoring family** becomes small increment methods dispatched on family aliases
  (`SmythBrobyDeltaAlg`, `SmythBrobyGerberAlg`, `SmythBrobyCountAlg`): `sb_add_pos`, `sb_add_neg`,
  `sb_add_neutral`. Each is a handful of lines; the surrounding traversal is shared.
- `SmythBrobyKernel{...}` bundles the algorithm plus the precomputed per-run arrays the loop needs,
  exposing `comovement_pair_state(pol, i, j)` and `comovement_step(pol, acc, st, xi, xj, …)`. The
  single `gerber_comovement!(rho, executor, X, …)` loop (parallelised via `FLoops`) walks pairs and
  calls the kernel — the duplicated per-marker loops are gone.

## Considered options

- **Keep one full method per marker.** Rejected: ~12× duplication of the double loop is the defect;
  the shared comovement mechanics could not be fixed in one place.
- **A macro that stamps out the twelve methods.** Rejected: it removes the *edit* duplication but
  keeps twelve generated loop bodies (harder to read, profile and debug) and still hard-codes the
  two axes as one flat product instead of factoring them, so it does not express that normalisation
  and scoring are independent.
- **Full type-parameter explosion** (`Gerber{Variant, Family}`). Rejected as a larger breaking
  change to the public marker names for no additional expressive power over the alias + kernel split,
  which keeps the existing user-facing types intact.

## Consequences

- The Smyth–Broby file drops from ~1350 to ~960 lines (and there is a matching cut in
  `GerberIQCovariance.jl`); the duplicated per-marker accumulation loops collapse into one shared
  loop plus the kernel, so the comovement mechanics live in a single place.
- The **public marker types are unchanged** — `Gerber0`, `SmythBrobyGerber2`, etc. still exist and
  construct as before; only the internal accumulation is restructured. Reference outputs
  (`SmythBrobyCovariance`, `GerberIQCovariance` fixtures) were regenerated where the consolidated
  path differs at the last ULPs.
- The two axes are now independently extensible: a new normalisation variant is one
  `comovement_ratio` method; a new scoring family is a `sb_add_*` triple. This is the concrete
  realisation of the arch-review "one data-carrying algorithm" idea, applied conservatively (a
  kernel plus family aliases) rather than as a public-API type-parameter overhaul.

## Amendment (2026-08-17): the fold now reaches `05_GerberCovariance.jl`

The Context above names only [06_SmythBrobyCovariance.jl](../../src/08_Moments/06_SmythBrobyCovariance.jl)
and [35_GerberIQCovariance.jl](../../src/08_Moments/35_GerberIQCovariance.jl). The plain Gerber
estimator in [05_GerberCovariance.jl](../../src/08_Moments/05_GerberCovariance.jl) kept its own three
loops, so `Gerber0`, `Gerber1` and `Gerber2` were members of the `GerberComovementZero/One/Two`
unions that nothing in their own file ever dispatched on. The fold was incomplete, not superseded.

That gap had a cost. The `05_` copies divided without a guard, while the shared
`comovement_ratio` / `standardise_comovement!` pair guards every denominator:

- `Gerber0` divided by `(U + D)' * (U + D)`, which is zero for a pair that never crosses a threshold
  together.
- `Gerber1` divided by `T .- N' * N`, which is zero for a pair that is always neutral.
- `Gerber2` divided by an **unclamped** `sqrt.(diag(H))`, which is zero for an asset that never
  crosses its own threshold.

One constant column in `X` is enough to reach all three. The estimator returned a `NaN` row, and
`posdef!` then threw `ArgumentError: matrix contains Infs or NaNs`. The Smyth-Broby path returned
finite values on the same input. The three methods also repeated the `U` / `D` preamble and an
identical three-line commented-out block verbatim.

**The `05_` methods now route through the shared reduction.** The preamble is one `gerber_updown`
call. `Gerber2` calls `standardise_comovement!` directly, so it inherits the clamp. `Gerber0` and
`Gerber1` recover the concordant and discordant counts from the matrix products with
`concordance_counts` and reduce them with `comovement_ratio`, so the denominator policy is written
once for the whole family. The split is exact and the reduction is bit-identical to the old matrix
formulas wherever the denominator is non-zero; only the previously-`NaN` entries change, and they
become the guarded zero that the Smyth-Broby path already returned.

`concordance_counts` exists so that the counts cost two matrix products rather than four: the
matrix formulation delivers `nconc - ndisc` and `nconc + ndisc` directly, and the two counts follow
from those by elementwise arithmetic on an `N x N` matrix.
