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
