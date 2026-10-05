---
status: accepted
---

# The Cross-Sectional Factor Prior refits on the online seam first and folds as a carry next

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
- The Fold Context of an optimiser keeps the Exogenous Series, because no prior kept it. The
  call with no data of a prior passes no Exogenous Series to the batch verb, so an observed factor
  of the Cross-Sectional Factor Prior had no series to read on the online step.
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

### A refit first, a carry fold next

1. **The refit.** The prior gains `cache`. `Online(CrossSectionalFactorPrior(…); max_history)` refits
   over a Sample Buffer, and its call with no data equals the batch fit over the buffer's rows, exactly. It
   costs one batch fit at each step, and `max_history` bounds it. A prior with no buffer refuses
   the step by name. The buffer records the Exogenous Series when the tree of the prior reads it,
   so an observed factor takes the step (see below).
2. **The carry fold.** An unwrapped prior seeds a carry state of its own, as `EmpiricalPrior` does, and
   applies the rule of ADR 0136: a carry folds what folds and refits the rest. At each step it computes
   the exposures of the new rows from the carried panel, runs the regression on the new dates only,
   and folds `pe`, `ve` and `ce`. The return forecast and the idiosyncratic correlation refit from the
   carried rows. A member that does not fold refits over the carried histories: a factor prior that
   is not an `EmpiricalPrior` at each call with no data, and a variance estimator that does not fold, such as
   a rolling window, by a fit of every carried observation at each step. The call with no data builds the
   result with the code of the batch fit, so the two routes differ only in how they reach its inputs.
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

### The prior's state holds the Exogenous Series

[#1476](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1476) decided this section.
The Exogenous Series follows the ruling for the panel: it is sample of the prior, so the prior's
state owns it once, and the Fold Context reads it back.

- A per-type predicate, `reads_exogenous_series`, recursive through the factor list and through an
  embedded prior, says whether the tree of a prior reads the series. It answers `true` for every
  member that reads `E`: the observed members and an estimated Descriptor that reads a series, such
  as `EWMacroSensitivity`. It also recurses through the Descriptor scores of the Return Forecast
  Estimator, which can hold such a Descriptor.
- The state keeps every column of the series and its names, not only the columns that the tree
  names. The Fold Context then holds no copy, and it rebuilds the whole `ReturnsResult` from the
  prior's state, as it does for the factor returns.
- `SampleBufferState` renames its estimation mask from `E` to `M`, and `E` and `ne` hold the
  Exogenous Series, so `E` means one thing on the whole online seam.
- A step of a tree that reads the series must bring it. The first step pins `ne`. A non-finite value
  is refused only on the rows that the fit reads, as ADR 0184 states. The step records such a value,
  and the call with no data refuses it when the fit reads its row, so a later cap can drop it.
- The refit route is built
  ([#1478](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1478)). Its call with no
  data equals the batch fit over the buffer's rows, bit for bit, with `CurrencyExposure`,
  `ObservedExposure` and `EWMacroSensitivity`, and its pinned choice under currency factors is at
  parity with the oracle's online update.
- The carry fold derives each new row of the net returns `Xl`, and of the returns net of the
  observed members that read no returns, one time, and carries the rows beside `X`. It also carries
  the series over its window and the observed returns of every fitted row. It never derives a row
  again, because the batch fit derives the first `lag` rows of the sample from the exposure of the
  same row, and a window that derives them again from its own first rows gives other values.
- The carry fold is built
  ([#1479](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1479)). Its buffer records
  the series as the refit's buffer does, so the Fold Context of an optimiser reads it back. Its call
  with no data equals the batch fit over the same rows, bit for bit, with `CurrencyExposure`,
  `ObservedExposure` and `EWMacroSensitivity`, and its pinned choice under currency factors is at
  parity with the oracle's online update. The carry keeps every fitted row, so a step refuses a
  non-finite observed return on a row that it fits, before the regression, as the batch fit does.
  An `EWMacroSensitivity` states no look-back, so the carry keeps every panel row under it, which is
  exact for a recursion from the first row.

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
| `BatchChoice()` | The default. The fit chooses again over every row, so the online call with no data equals the batch fit. |
| `PinnedChoice()` | The choice of the first fit is kept. The oracle's rule for the dropped member. |

The field sits on the estimator that makes the choice: `CrossSectionalFactorPrior` for the dropped
member of each family, `StepwiseRegression` for its factor set, and `DimensionReductionRegression` for
its projection. A refit honours a pinned choice with no state: after the first fit, the step writes
the choice into the configuration of the estimator that it returns. The first fit is the first step
whose buffer the batch fit accepts, which for the Cross-Sectional Factor Prior is `lag + 2` rows
after the Descriptor warm-up. The step runs the part of the fit that the choice reads over the
buffer, so the member that it writes is the member that the call with no data of that step drops. In the carry fold, a batch choice
that moves refits every past date, and a pinned choice is recorded in the carry state.

### The selection regressions of a factor prior

- `StepwiseRegression` holds the factor set of each asset in `included`, one entry per asset, and
  a fit runs no search for an asset whose entry is set. The search of one asset reads only its own
  column and the factor returns, so the step pins every asset whose column of the buffer is finite,
  and an asset that the fit does not cover yet pins at the first fit that covers it. The field is on
  the asset axis, so a view slices it, and each prior views the regression to the assets that it
  fits.
- `DimensionReductionRegression` holds its components in `proj`, as weights of the original factors.
  The pin keeps the standardisation scale of the first fit inside those weights, because the
  components are combinations of the original factors and a new scale moves them. The mean is not
  pinned: a constant shift of the components changes only the intercept of the fit, which the
  recovery rebuilds from the mean of the rows that it fits.
- The first fit is the first step whose buffer holds two rows, the fewest that both regressions
  accept.
- One hook runs after the fold of every refitting prior. It pins the regression of `FactorPrior`,
  `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior`, and it reaches a factor prior
  inside a prior that embeds it, because that prior refits from its own buffer and never steps the
  embedded one.

### The carry fold carries only the rows that the Descriptors read

Each Descriptor states its look-back: an `Integer`, or `nothing` for a recursion from the first row.
The carry keeps the last `look-back + lag` panel rows when every look-back is finite, and every row
otherwise. A Return Forecast that reads the panel keeps a value at every fitted observation in its
Result, and it aligns the fitted observations with the last rows of its returns data, so the carry
keeps every row under it. The output is exact in every case.

## Considered options

- **The full fold of the oracle, now.** Rejected for now. It gives an O(1) step, but it needs about
  fifteen Descriptor states and the lag buffers before any caller can take a step. The carry fold
  removes about 60 % of the cost with a small part of that build, and the Descriptor folds stay open.
- **A refit only.** Rejected. It reaches every output, but a daily walk-forward over ten years costs
  about one hour at 500 assets.
- **A new state type for the panel, or the panel in the Fold Context.** Rejected. A second state type
  needs a second read-back path. A panel in the Fold Context leaves a prior that folds alone, outside
  an optimiser, with no panel.
- **The Exogenous Series in the Fold Context alone, passed to the call with no data.** Rejected. It
  is smaller, but a prior that folds alone cannot read an observed factor, and the carry needs the
  new rows of the series at each step. The same argument rejected the panel in the Fold Context.
- **The series as an argument of the call with no data.** Rejected. The caller must keep the whole
  history, and the call with no data no longer equals the batch fit by itself.
- **Only the named columns of the series in the prior's state.** Rejected. The Fold Context must
  then keep a second copy, or join two parts of the series again in the order of `ne`.
- **A refit only for an observed factor.** Rejected. Each step then costs one batch fit, and the
  oracle folds its currency factors.
- **A start-row window.** Rejected. It is exact in a refit, but the caller must know the offset of the
  rows that the moment member gets, which the Descriptor warm-up, the lag trim and a cap all move.
- **A pinned choice by default.** Rejected. It breaks the identity of the online call with no data and the batch
  fit when the choice moves. It stays one keyword away.
- **A cap on the carried panel.** Rejected. It is not exact with an EW Descriptor.

## Consequences

- The builds are children of map #1375:
  [#1467](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1467) (the panel on the
  online seam), [#1468](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1468) (the
  refit and the Choice Rule), [#1469](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1469)
  (the Seed Window), [#1470](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1470)
  (the Descriptor look-back), [#1471](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1471)
  (the carry fold) and [#1472](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1472)
  (the Choice Rule on the selection regressions of `FactorPrior`),
  [#1478](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1478) (the refit records
  the Exogenous Series) and
  [#1479](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1479) (the carry fold
  takes an observed factor).
- ADR 0136 and ADR 0039 carry amendments that point here.
- A pinned choice and a seed window are two routes on which the online call with no data equals no batch fit.
  Each is documented, and each is tested against the oracle.
