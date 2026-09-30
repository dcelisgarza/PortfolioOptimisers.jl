---
status: accepted
---

# The Cross-Sectional Factor Prior refits on the online seam first and folds as a host next

## Context

[ADR 0186](0186-an-oracle-mode-is-built-when-a-caller-cannot-reach-its-output-and-four-differences-are-deliberate.md)
put the online update of the Cross-Sectional Factor Prior in scope of map
[#1375](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1375) (mode 16a), and
[#1409](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1409) decided the route. The
standard of the map is that the library never has less capability than the oracle.

### What the library held

- `CrossSectionalFactorPrior` has no `cache` field, so `Online` refuses it.
- `SampleBufferState` holds the returns, both masks and the factor returns, and no Asset Panel.
  [ADR 0136](0136-a-prior-folds-and-carries-the-buffer-is-owned-once-and-a-cap-is-either-a-scenario-cap-or-a-window.md)
  said that the Asset Panel is fold context, not sample.
- The online step passes an active mask alone to the prior. It refuses every Panel Field and every
  estimation mask that differs from the active mask, for every prior. But the generic refit of a
  prior already takes an estimation mask.
- The windowed wrappers
  ([ADR 0039](0039-windowed-estimators-are-generated-from-one-declaration.md)) keep the last `w`
  rows of every fit, so on the online seam their window rolls.

### What the oracle does

Its prior folds every part: each Descriptor keeps a ring buffer or an EW state, the regression runs on
the new dates only, and three buffers keep the last `lag` rows of the exposures, the family ratios
and the market capitalisation. Its asset universe is fixed at the first call. Its window on the EW
moments cuts only the first batch, and every later row folds. It pins the automatic choice of the
dropped member of each Factor Family at its first call. Its own tests accept its fold against its
batch fit at `rtol = 1e-10`.

### What a new row changes

A Descriptor value at a date reads only the rows up to that date, and the regression and its weights
are per date. So a new row changes no past factor return, except through the automatic choice of the
dropped member, which reads the whole exposure history.

### What a fit costs

One batch fit at 500 assets and 2520 observations takes 2.9 s and allocates 3.0 GB. A profile gives
about one half of the time to the regression over every date, one third to the exposures, and one
tenth to the EW moments, which already fold exactly.

## Decision

### A refit first, a host fold next

1. **The refit.** The prior gains `cache`. `Online(CrossSectionalFactorPrior(…); max_history)` refits
   over a Sample Buffer, and its read-out equals the batch fit over the buffer's rows, exactly. It
   costs one batch fit at each step, and `max_history` bounds it.
2. **The host fold.** An unwrapped prior seeds a carry state of its own, as `EmpiricalPrior` does, and
   applies the rule of ADR 0136: a host folds what folds and refits the rest. At each step it computes
   the exposures of the new rows from the carried panel, runs the regression on the new dates only,
   and folds `pe`, `ve` and `ce`. The return forecast and the idiosyncratic correlation refit from the
   carried rows.
3. The folds of the Descriptors and of the return forecast are not specified yet. They can come one
   family at a time.

### The Sample Buffer holds the panel

`SampleBufferState` gains a slot for the Panel Fields beside the factor returns and the masks,
under the same count and cap. The masks stay in their own slots. The slot holds the rows of the
valid region alone, so the offset does not index it, and an append joins the rows with `vcat` of
two Asset Panels. The first append fixes whether it records the Panel Fields. A per-type predicate,
`reads_panel_fields`, recursive through an embedded prior, says whether the tree of a prior reads
them. The prior's buffer owns the panel once, and the Fold Context reads it back, as it reads the
factor returns.

### The step decides its two refusals by route

The step passes the estimation mask to every prior whose route honours it: every refit, and the host
fold. It refuses a differing estimation mask only for a route that cannot honour it, such as the carry
of `EmpiricalPrior`. It passes the Panel Fields to a prior whose buffer records the panel, and it
refuses them for the other priors.

### One public form of a prior fold

`partial_fit!(pe, rd::ReturnsResult)` is the fold of every prior. It mirrors `prior(pe, rd)`: it reads
the returns, the factor returns, both masks, and the Panel Fields when the prior records the panel.
The matrix form stays for the priors that read no panel.

### A window follows one of two rules

The windowed wrappers gain a field that holds the window rule, as two singleton types:

| Rule | Meaning |
| --- | --- |
| `RollingWindow()` | The default. The last `w` rows of every fit, the rule of #997. |
| `SeedWindow()` | The last `w` rows of the first fit alone; the inner estimator then folds every row. The oracle's rule. |

The two agree in a batch fit. A seed window needs an inner estimator that folds exactly, held by a
host that folds. A refit refuses it by name, because a refit has no first fit to remember.

### A whole-sample choice follows one of two rules

A Choice Rule is a pair of library-wide singleton types under one abstract type:

| Rule | Meaning |
| --- | --- |
| `BatchChoice()` | The default. The fit chooses again over every row, so the online read-out equals the batch fit. |
| `PinnedChoice()` | The choice of the first fit is kept. The oracle's rule for the dropped member. |

The field sits on the estimator that makes the choice: `CrossSectionalFactorPrior` for the dropped
member of each family, `StepwiseRegression` for its factor set, and `DimensionReductionRegression` for
its projection. A refit honours a pinned choice with no state: after the first fit, the step writes
the choice into the configuration of the estimator that it returns. In the host fold, a batch choice
that moves refits every past date, and a pinned choice is recorded in the carry state.

### The host fold carries only the rows that the Descriptors read

Each Descriptor states its look-back: an `Integer`, or `nothing` for a recursion from the first row.
The carry keeps the last `look-back + lag` panel rows when every look-back is finite, and every row
otherwise. The output is exact in both cases.

## Considered options

- **The full fold of the oracle, now.** Rejected for now. It gives an O(1) step, but it needs about
  fifteen Descriptor states and the lag buffers before any caller can take a step. The host fold
  removes about 60 % of the cost with a small part of that build, and the Descriptor folds stay open.
- **A refit only.** Rejected. It reaches every output, but a daily walk-forward over ten years costs
  about one hour at 500 assets.
- **A new state type for the panel, or the panel in the Fold Context.** Rejected. A second state type
  needs a second read-back path. A panel in the Fold Context leaves a prior that folds alone, outside
  an optimiser, with no panel.
- **A start-row window.** Rejected. It is exact in a refit, but the caller must know the offset of the
  rows that the moment member gets, which the Descriptor warm-up, the lag trim and a cap all move.
- **A pinned choice by default.** Rejected. It breaks the identity of the online read-out and the batch
  fit when the choice moves. It stays one keyword away.
- **A cap on the carried panel.** Rejected. It is not exact with an EW Descriptor.

## Consequences

- The builds are children of map #1375:
  [#1467](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1467) (the panel on the
  online seam), [#1468](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1468) (the
  refit and the Choice Rule), [#1469](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1469)
  (the Seed Window), [#1470](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1470)
  (the Descriptor look-back), [#1471](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1471)
  (the host fold) and [#1472](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1472)
  (the Choice Rule on the selection regressions of `FactorPrior`).
- ADR 0136 and ADR 0039 carry amendments that point here.
- A pinned choice and a seed window are two routes on which the online read-out equals no batch fit.
  Each is documented, and each is tested against the oracle.
