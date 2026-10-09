---
status: accepted
---

# The Cross-Sectional Factor Prior refits on the online seam first and folds as a carry next

## Context

[ADR 0186](0186-an-oracle-mode-is-built-when-a-caller-cannot-reach-its-output-and-five-differences-are-deliberate.md)
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
   and folds `pe` and `ve`. The return forecast and the idiosyncratic correlation `ce` refit from the
   carried rows at the call with no data. A member that does not fold refits over the carried histories: a factor prior for
   which `carry_folds` answers `false` at each call with no data, and a variance estimator that does not fold, such as
   a rolling window, by a fit of every carried observation at each step. The call with no data builds the
   result with the code of the batch fit, so the two routes differ only in how they reach its inputs.
   A step appends its rows to the histories in place, into the spare rows of a backing array, as
   the Sample Buffer does, and each state keeps a view of its own rows (#1564). A step of a state
   that a later step passed copies the state first, so the later state and the results read out of
   it keep their rows.
3. The folds of the Descriptors are not specified yet. They can come one family at a time. A return
   forecast folds when `folds_forecast_rows` answers `true`: the state carries its Descriptor scores,
   its history and a fold state of its own. A step gives the forecast its new rows and the
   `forecast_target_gap` rows before them, whose targets mature at the step. `FixedWeightedReturnForecast`
   folds with no fold state (#1573). `ExpWeightedReturnForecast` carries its normal equations, its
   count and its coefficients, because its batch fit is the same forward recursion (#1574).
   `TargetReturnForecast` under its prequential default carries its normal equations, its model,
   its two calibration regressions and the coefficients of the rows whose target has not matured,
   and its batch fit runs the same step from an empty state (#1581, ADR 0200).

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
- A step of a tree that reads the series must bring it. The first step pins `ne`. A `NaN` is a gap,
  and the fit reads it by its own gap rule (#1530). An infinite value is refused only on the rows that the fit reads, as ADR 0184 states. The step records such a
  value, and the call with no data refuses it when the fit reads its row, so a later cap can drop
  it.
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
  parity with the oracle's online update. The carry keeps every fitted row, so a step refuses an
  infinite observed return on a row that it fits, before the regression, as the batch fit does. A
  `NaN` marks a gap, and the step folds it as the batch fit reads it (#1530).
  An `EWMacroSensitivity` states no look-back, so the carry keeps every panel row under it, which is
  exact for a recursion from the first row.

### The step decides its two refusals by route

The step passes the estimation mask to every prior whose route honours it: every refit, and the carry
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

`RollingWindow()` is the default because it keeps the definition of a window: after `s` steps a
seed window holds `w + s` rows, so it is no longer the window of `w` rows that it states, and the
rolling window equals the batch fit over the same rows (#1416, row R28). `SeedWindow()` stays for
the oracle's numbers and for its state of fixed size. On the idiosyncratic variance of the prior,
the oracle folds one row at a time also at its first call, so its stated window never cuts
anything (measured difference exactly 0). Ours cuts the first fit's block, because a window that a
caller states must act. The oracle's output there is the inner variance with no window (#1416,
row R104).

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
buffer, so the member that it writes is the member that the call with no data of that step drops. In the carry fold, a pinned
choice is recorded in the carry state.

A batch choice that moves the dropped member on the carry fold runs no regression again
([#1601](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1601)). The zero-sum
condition of a family is the same set for every dropped member. So the raw-axis factor returns, the
residuals and the idiosyncratic variances do not depend on the member, and the reduced history under
the new member is a selection of the columns of the raw-axis history. Under the default Unseen
Member rule `ZeroUnseenMember()`, an Unseen Member has a return of zero, and the zero-sum condition
of its observation holds over the other members
([#1606](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1606)). So every row is
identified, and its raw-axis answer does not depend on the member either. A move selects those
columns and refits the factor prior over the selected history. It solves no row again. The marks
of the Empty Factors of the family that moved are read again in the new basis, one observation at
a time from the first one, until each member is marked. Under `SolvedUnseenMember()` a row with an
Unseen Member stays rank-deficient, and its pseudo-inverse answer depends on the parametrisation.
A `CrossSectionalTargetRegression` fits a target that the library does not know, and a target can
penalise its coefficients. Under either one a move fits every carried observation again, as it did
before the fold (#1605). A row whose design holds a dependent factor set is rank-deficient under
every rule, and the fold keeps the answer of the old basis there. The
default factor covariance reads every column at once, so a sub-block of a raw-axis state is not the
batch answer. On the panel of #1592 (2520 rows, 500 assets), the step of the move costs 0.013 s,
against 0.0036 s for a step with no move and 1.66 s for a refit of every row, and the carry still
equals the batch fit to rounding (7.3e-15 on `sigma`). `BatchChoice()` stays the default.

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
otherwise. A Descriptor that folds from a state of its own counts as one row: a rolling return
keeps the last `window + skip + 1` rows of its cumulative sums in its `cache`, and reads the
Descriptor of a new row off them, equal to the batch fit to the last bit
([#1583](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1583)). So a momentum factor
with a look-back of 273 rows leaves the carry at `1 + lag` rows. A Return Forecast that reads the panel keeps a value at every fitted observation in its
Result, and it aligns the fitted observations with the last rows of its returns data, so the carry
keeps every row under it, unless the forecast folds. A forecast that folds reads the look-back of its
Descriptors alone, and a Descriptor of its scores that folds from a state of its own counts as one
row there too: the carry keeps the Descriptor Scores with the state of each such Descriptor
([#1587](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1587)). The output is exact
in every case.

### The Carry Rule

Some members make the cost of a step grow with the stream: a Descriptor or an Exposure member with
no finite look-back and no state, a Return Forecast that reads the panel, a `ve` that does not fold,
a factor prior `pe` that does not fold, and the idiosyncratic correlation (`th > 0`) when its `ce`
does not fold, because it then refits at each read-out. A `ce` that folds, as the default
`ExpWeightedCovariance` does, folds the standardised rows of each step, and the read-out reads its
state ([#1594](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1594)). The oracle refuses each of them on its fold. Ours fits them again over the rows
that they need, so the step stays exact
([#1590](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1590)).

The rule is a field of the prior, `carry::AbstractCarryRule`, with two singleton types:

| Rule | Meaning |
| --- | --- |
| `FoldOrRefit()` | The default, the rule of ADR 0136. A member that does not fold is fitted again at each step. Its cost is documented. |
| `FoldOnly()` | The constructor refuses a prior with a member whose step grows, and names each one. A bounded window passes. |

- The refusal is in the constructor, because every test reads the configuration alone, and an
  error must come as early as possible. A batch fit and `Online` ignore the rule, but a caller who
  writes `FoldOnly()` asked for it.
- `FoldOnly()` refuses a Batch Choice with an automatic dropped member, because a move refits the
  factor prior over every carried factor return
  ([#1601](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1601)). A Pinned Choice and
  a named member never move, so both pass.
- `lookback` and `supports_partial_fit` are `public`, because a user subtype implements them to
  pass `FoldOnly()`. The stateful seam of a Descriptor is `public` too: `descriptor_step` and
  `carry_lookback`, with the state in a field of the Descriptor. The exponentially weighted mean
  Descriptors are its second family, after the rolling return
  ([#1586](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1586)), and the lag
  Descriptors are its third
  ([#1598](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1598)). The exponentially
  weighted volatility Descriptors are its fourth
  ([#1607](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1607)): their state holds
  the state of the variance estimator `ce`, so they fold only where `ce` folds, and refit under
  `FoldOrRefit()` or are refused under `FoldOnly()` otherwise. A subtype that
  implements both verbs folds on the carry with no change to the prior.

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
- **A refit of every row when a batch choice moves.** Rejected. It is exact, but it costs about half
  a batch fit at each move, and the fold of the move is exact to rounding at about 0.013 s a step.
- **A pinned choice by default on the carry fold alone.** Rejected. One field serves the batch fit
  and the carry, so a default that differs by mode makes the two disagree for one estimator. With
  the fold of a move, the batch choice costs about the same as the pinned choice.
- **Accept a Batch Choice under `FoldOnly()`, and document the cost of a move.** Rejected. The cost
  of a move grows with the stream, and `FoldOnly()` promises that no step cost grows.
- **Accept a Batch Choice under `FoldOnly()` when the factor prior is separable by column.**
  Rejected. The carry then needs a raw-axis state beside the reduced one, and the default factor
  covariance is not separable, so the rule depends on the factor prior.
- **A cap on the carried panel.** Rejected. It is not exact with an EW Descriptor.
- **Refuse a member that does not fold, as the oracle does.** Rejected as the default. The refit is
  exact and costs the user nothing to write. It stays one keyword away as `FoldOnly()`.
- **Warn once, then refit.** Rejected. A cost is documented, not warned.
- **A `Bool` field for the strict rule.** Rejected. The Choice Rule and the repair rule of the same
  prior are singleton types, and a third rule can join the family with no breaking change.
- **The refusal at the entry of `partial_fit!`.** Rejected. It keeps a prior constructible for a
  batch fit, but the error comes later than the configuration lets it.
- **Refuse every member with no state, a bounded window too.** Rejected. A bounded window has a
  cost that does not grow.

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
- [#1602](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1602) built the Carry
  Rule, a child of sub-map #1562. `carry_growing_parts` lists the parts that grow, and
  `assert_carry_rule` applies the rule in the constructor.
  [#1594](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1594) folds the
  idiosyncratic correlation, so `FoldOnly()` refuses `th > 0` only with a `ce` that does not
  fold.
  [#1595](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1595) made the verb
  "this factor prior folds", the `public` function `carry_folds`. An `EmpiricalPrior` on its carry
  route answers `true` when its `me` and `ce` fold, and every other prior answers `false`. The
  carry folds the factor prior and reads it out by the verb, not by its type, so a user prior that
  implements `carry_folds`, `partial_fit!` and `prior` with no data folds and passes `FoldOnly()`.
  The verb is not `supports_partial_fit`, which answers `false` for an `EmpiricalPrior` because an
  outer estimator that holds the rows refits it from them.
- [#1605](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1605) builds the fold of a
  move of a batch choice and the refusal of an automatic member under `FoldOnly()`.
  [#1606](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1606) adds the Unseen
  Member rule `unseen`. The default `ZeroUnseenMember()` gives an Unseen Member a return of zero
  and holds the zero-sum condition over the other members, so a row whose factor returns depended
  on the dropped member is identified, and the fold of a move needs no second solve.
  `SolvedUnseenMember()` keeps the answer that depends on the dropped member, and a move under it
  fits every carried observation again.
- ADR 0136 and ADR 0039 carry amendments that point here.
- A pinned choice and a seed window are two routes on which the online call with no data equals no batch fit.
  Each is documented, and each is tested against the oracle.
