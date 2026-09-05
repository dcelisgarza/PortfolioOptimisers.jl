---
status: accepted
---

# The two evaluation switches are separate, and the library does not bundle them

## Context

A backtest reads a fold in two places, and each place holds a question the library had answered
once and for all.

**The first question is what a fold's return series means.** `calc_net_returns(w, X, fees)`
returned `X * w` net of fees on every observation. That is the reading the optimiser maximises,
and it is the right reading of the decision. It is not the reading of a fund: a fund buys the
weights once, and each position then grows at its own return, so the weights it holds move away
from the weights it chose. The gap is real money on a long fold.

**The second question is which weights the next fold starts from.** `fold_loop` threaded the
previous fold's *target* weights, so `Turnover`, `TurnoverEstimator`, `WeightsTracking`,
`TurnoverRiskMeasure` and the turnover fee all measured the change in the decision. A fund does
not trade the change in the decision. It trades the distance from what it holds to what it now
wants, and that distance is larger, because the holdings moved while the decision stood still.

A reference implementation of the drifted series exists, and it answers both questions with one
flag. Turning that flag on makes a fold report the drifted series **and** carry its held weights
into the next fold. The two answers arrive together, and there is no way to take one without the
other.

Two further facts shaped the answer, and each was measured rather than assumed.

- **A drifted fold adds no cross-fold dependency.** The drift reads one fold's own weights and
  one fold's own asset returns. It is a function of a fold, so it does not make a fold wait for
  its predecessor.
- **Threading the held weights does make a fold wait, and that wait already existed.** A run whose
  optimiser reads previous weights is sequential today, whatever the source of those weights is.
  `needs_previous_weights` decides it, and it reads the optimiser, not the scheme.

So the bundle costs something the split does not: bundling makes the first switch inherit the
second switch's sequential run.

## Decision

**The two questions are two switches, and each one is a field of its own.**

| Switch | Field | Off (`nothing`) | On |
| --- | --- | --- | --- |
| Weight Drift | `wd::Option{<:AbstractWeightDrift}` | `X * w` net of fees | the wealth ratio of the drifted holdings |
| Previous-Weights Source | `pws::Option{<:AbstractPreviousWeightsSource}` | the previous fold's target weights | the previous fold's held weights |

`nothing` on either switch is the library's original behaviour, so every number a caller has
today is unmoved until that caller sets a field.

**Each switch reaches the schemes that can read it, and no others.** `wd` is a flat field on
`KFold`, `CombinatorialCrossValidation`, `IndexWalkForward` and `DateWalkForward`. `pws` is a
flat field on the two walk-forwards alone, because a `KFold` fold and a combinatorial fold carry
no history: each of their folds is independent, so there is no previous fold whose held weights
could be read. `MultipleRandomised` and the two search estimators inherit both through their
inner `cv`. A switch that a scheme cannot honour is therefore absent from that scheme, rather
than present and refused.

**The drift is a cumulative product, and the held weights are one step beyond the last row of the
path.** Position values are `cumprod(1 .+ R; dims = 1) .* transpose(w)`, the wealth of an
observation is the sum of that row, and the return of an observation is the ratio of two
successive wealths. The weights held *through* observation `t` are the position values of `t`
divided by the wealth of `t`, and the weights carried *forward* are one step beyond the last
observation, which is what the next fold starts from.

**A fold that drifts carries a Held Weights record, and a fold that does not carries nothing.**
`HeldWeightsResult` holds the fold's asset returns, the held weights, the drift form that ran and
the weight path when the scheme asked for it. It rides in an `Option`-bound `hw` field on
`PredictionResult`, so a consumer that needs the path serves the fold that carries a record and
refuses the fold that does not, separated by dispatch rather than by an `isnothing` branch.

**The one-off cost has a clock of its own, and it is a third switch on a third type.** `Fees` and
`FeesEstimator` carry `fa::Option{<:AbstractFeeAmortisation}`, whose one leaf `AmortisedFees`
spreads the turnover and the two fixed charges over a holding period. It is orthogonal to both
evaluation switches: a caller can drift without amortising, and amortise without drifting.

## Consequences

- **A drifted run stays parallel.** `needs_previous_weights` did not move, and neither switch
  reaches it. A caller who wants the fund's reading of the return series pays no run-time for it.
  Under the bundle the same caller would pay a sequential run.
- **Four settings exist where the reference implementation has two.** Off/off is the decision's
  reading. On/off reads the fund's series while it measures the change in the decision. Off/on is
  the setting a caller reaches for when the trades matter and the series does not. On/on is the
  fund's reading of both.
- **A turnover cap means what the caller thinks it means only when the source is on.** With the
  source off, the cap binds the change in the decision, and the trade the fund places is that
  change plus the drift. With the source on, the cap binds the trade.
- **The weight path is rebuilt, not stored, by default.** The record carries the drift form that
  ran, so the rebuild is bit-identical to the store. `store_weight_path` on the scheme turns the
  eager store on for a caller who reads the path many times.
- **A ruined member is dropped, and a ruined single vector raises.** A drifted wealth that reaches
  zero or turns negative has no return series, so the drift refuses to form one. Under a
  population the failing member takes an `OptimisationFailure` retcode and the existing filters
  drop it; a single weight vector is a population of one, so it raises `NonPositiveWealthError`.
- **The reproduction is exact.** With both switches on, the library reproduces the reference
  implementation's return series, held weights and executed turnover at the reference's own
  tolerance, at any panel size.

## Alternatives considered

- **One flag for both questions, as the reference implementation has.** Refused. It is one
  switch fewer to explain and two capabilities fewer to offer, and it makes a drifted run
  sequential for a dependency the drift does not have.
- **One typed block carrying both axes, held as one field on every scheme.** Refused after it was
  first accepted. The block would have to ride on `KFold` and on the combinatorial scheme, where
  half of it can never be honoured, so the type would promise a capability the scheme does not
  have. Flat fields let each scheme carry only the switches it can read.
- **A field on every optimisation estimator rather than on the scheme.** Refused. It would go on
  sixteen or more structs and on `Pipeline`, and a Result-side read is not universal, because a
  naive optimisation result carries no fee field at all.
- **Two `Bool` fields.** Refused. The drift needs to name *which* drift ran, so that a rebuild of
  the path is bit-identical, and a `false` beside a stated form is a state that means nothing.
- **A new fee type for the smoothed charge, beside the existing one.** Refused on a measured
  cost: 201 `Option{<:Fees}` type bounds across 29 files would have had to widen. A field on the
  two existing fee types moves no bound.
