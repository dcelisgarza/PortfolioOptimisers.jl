---
status: accepted
---

# A forced liquidation is charged through two fee carriers on the complement of the Investable Mask

## Context

[ADR 0115](0115-every-optimisation-estimator-reduces-once-at-its-entry-and-its-result-carries-the-investable-mask.md)
reduces every optimisation to the Investable Mask at its entry, through a `port_opt_view` of the
optimiser that slices every constraint the caller stated by the same asset index, and its result
carries the objects of the reduced universe beside the mask.
[ADR 0120](0120-a-fold-scores-on-the-investable-mask-a-prior-free-head-and-pre-selection-reduce-to-the-coverage-universe-and-a-failed-candidate-loses-the-search.md)
carries that rule to the fold. Ticket
[#888](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/888) on map
[#667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/667) found what the slice
throws away.

A fund that holds asset `j` at the end of fold `i` must sell it when `j` leaves the Investable
Mask in fold `i+1`. The sale is a trade, and a commission is due on it. The library charged nothing
for it, at both doors, and each door dropped it deliberately:

- `Turnover` tags `w` and `val` `@vprop`, and `Fees` tags `tn`, `l`, `s`, `fl` and `fs` `@vprop`,
  so the view at the mask drops **both** halves of the charge, the previous weight of the exiting
  asset and the per-asset rate that would price it. After the reduction the library no longer held
  the rate at all, so no site downstream could price the exit even if it wanted to.
- The JuMP fee expression therefore priced no exit, and the fold's realised series subtracted no
  exit cost. On a probe, one asset held at a quarter of the book with a rate of one percent left a
  quarter of a percent unpaid on that fold. A walk-forward over a delisting panel overstated its
  return by that much per exit, in silence.
- Every family resolved a `FeesEstimator` **after** the door, against the reduced `sets`, so a
  name-keyed rate for the asset that left was dropped with a warning under `strict = false` and
  refused with `ArgumentError: variable c not in asset universe` under `strict = true`.
- The finite allocation took a result's reduced `Fees` beside the result's full-length weights
  and prices, and raised `DimensionMismatch`.

The reference implementation added the same charge, after map
[#746](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/746) closed, as a fix in its
own backtest path: an exit is full turnover at the previous weight, the cost rides at the previous
weight times the asset's rate, and the turnover bound does not bound it. It keeps the caller's
name-keyed inputs beside the aligned arrays and computes the charge from the names outside the
investable set. Its convex family alone does this, because its hierarchical families never reduce.

Two defects were found and fixed on the way,
[#892](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/892): the fold viewed a
result's `Fees` at the mask a second time, and the two hierarchical leaves answered no mask
because they forward it as a property the verb cannot dispatch on.

## Decision

### A forced liquidation costs money, at both doors

A position in an asset that leaves the Investable Mask is charged as a trade to zero. The charge
enters the model's fee expression, multiplied by the homogenising variable as the turnover term is,
so under a ratio objective it moves the argmin as it does in the reference. It enters the fold's
realised series through `calc_fees`. The turnover bound does not bound it, because the trade is
forced rather than chosen, and a bound that refused it would make the programme infeasible for a
reason the caller cannot act on. The docstrings of `Turnover` and `Fees` state this.

### Two carriers on `Fees`, each a `Turnover`

`Fees` gains `lq`, the proportional liquidation carrier, and `flq`, the fixed liquidation
carrier. Each is an `Option{<:Turnover}`, because a forced exit is a trade against a reference
vector, which is what `tn` already is: a `w` and a `val`. `FeesEstimator` gains the same two
fields as `Option{<:TnE_Tn}`, so a name-keyed rate with a default resolves through
`turnover_constraints` as `tn` does. Both are `@fprop`, so `factory` threads the fold's previous
weights into them beside `tn.w`, and `needs_previous_weights` on a `Fees` reads all three. Neither
is `@vprop`: after the door they live on a different axis from the five per-asset fields, and the
generic view must leave them alone. `fixed` keeps its meaning on a carrier and is not refused.

The proportional charge is the rate times the absolute previous weight, summed over the carrier's
entries. The fixed charge is the amount for every entry whose absolute previous weight is not
`isapprox` to zero under `kwargs.atol`, the threshold `fl` and `fs` use.

**The two carriers are on different clocks**, which
[#898](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/898) settles. `lq` is a
proportional rate, so it is a turnover-like term and charges on every period beside `l`, `s` and
`tn`; no clock reaches it. `flq` is a currency amount charged one time for the whole holding
period, so it falls on the clock `fa` names, beside `fl` and `fs`: a `nothing` `fa` charges it on
the first observation of the series, and an `AmortisedFees` spreads it evenly over the observation
count the charging site knows. The model states the same rule, and always spreads the one-off
terms into an expected return, which answers for the fixed terms the open question of
[#815](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/815).

### The fit sites resolve the fee on the full universe, then view it by the mask

Every fit site resolves its `FeesEstimator` **before** the door, through the `fees_constraints`
call it already made, against the **full** `sets` the caller stated, and hands the resolved `Fees`
to `investable_fees_view(fees, imsk, X)`. That verb takes `port_opt_view(fees, findall(imsk), X)`,
which is the one two-axis view `Fees` writes by hand: `tn`, `l`, `s`, `fl` and `fs` at the mask, and
the two carriers at its complement, derived from the width of the *unreduced* `X`. When `imsk` is
`nothing` it hands the fee to `strip_liquidation_carriers` instead, which drops both carriers
because nothing exits, so the all-investable path allocates nothing. The build refused the verb
this decision first named, `investable_fees(fees, sets, imsk; …)`: the resolution and the view were
already two verbs the library owned, and a third that fused them would have been a copy. The
value-level door of
[ADR 0118](0118-a-fold-zeroes-a-held-gap-once-and-a-value-level-verb-reduces-to-the-investable-mask.md)
takes the same `port_opt_view` at the prior's unreduced `X`. A carrier whose length is not the
universe's is refused with a `DimensionMismatch` naming both lengths. Resolution on the full
universe is what lets a name-keyed rate for a departed asset resolve, so the `strict` refusal goes.

The reduced `Fees` therefore carries two axes: the investable one for the five per-asset fields,
and the complement for the two carriers. The fold reads it as the result carries it and views it
no second time, which is what #892 fixed.

### A reduced `Fees` records the mask it was reduced on, and the door reads it before it acts

A caller states the two carriers over the full universe, and a result carries them on the
complement of its mask. Width cannot tell the two apart — with eight assets and four dead, both
are four long — and the same `nothing` arm of the value-level door serves both, because an
all-investable prior derives no mask and neither does a result's reduced prior.
[#1067](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1067) found what that
costs: `investable_reduction(::Nothing, pr, w, fees, strict)` passed the fee through, so
`expected_return(ArithmeticReturn(), w, pr, fees)` on an all-investable prior charged a caller's
full-universe carrier as a forced exit of the whole previous book, on every period, though nothing
left. Stripping in that arm is the fix that reads naturally, and it breaks `expected_risk(r, res)`
on a masked result, which hands the same arm the result's reduced fee beside the result's reduced
prior and relies on the pass-through to charge the exit.

`Fees` therefore gains a tenth field, `imsk::Option{<:BitVector}`, defaulting to `nothing`.
`investable_fees_view` is the one verb that writes it, and it reads it before it acts, so the door
is idempotent: an unmarked fee is a caller's statement and takes the arm the mask names — the
carriers dropped under `nothing`, the two-axis view and the mark under a `BitVector`; a fee marked
with the door's own mask, or marked and meeting a `nothing` mask, is returned as it is; a fee marked
with another mask is refused with an `ArgumentError` naming both, because its carriers were sliced
to the complement of its own mask and hold no rate for an asset the other reduction kept, and
charging nothing for it would understate the return. Every arm of `investable_reduction` at the
value-level door now takes the fee through this verb, the bare-matrix and `ReturnsResult` arms
under a `nothing` mask, so the door of
[ADR 0118](0118-a-fold-zeroes-a-held-gap-once-and-a-value-level-verb-reduces-to-the-investable-mask.md)
and the fit sites are one door for the fee.

The generic `port_opt_view(fees, i, X)` marks nothing. A cluster of a nested optimiser takes the
same slice, and its inner fit must still strip the carriers it is handed under its own `nothing`
mask, so a mark written by the view would make an inner fit read a cluster's complement as the
assets that left. `strip_liquidation_carriers` reads no mark either: the hierarchical fits call it
on a fee the door has already reduced, to price a cluster's risk with no exit, and the mark is
carried through so the fee stays reduced with its carriers gone. `lift_fees` takes the fee alone
and lifts at the mask it carries, so the fee is the one source of the axes it is on, and
`FiniteAllocationInput` reconciles its own `imsk` with the fee's through `mark_fees`: a stated mask
marks an unmarked fee, a marked fee supplies a missing mask, and a pair that disagrees is refused.
`show` hides the field while it is `nothing`, so a fee a caller wrote renders as it did before the
field existed.

### A fit charges the liquidation at its outer level only

A non-investable asset cannot be clustered, because its column is `NaN`, so no cluster holds it.
The hierarchical fits read their intra-cluster risk from a `Fees` with the carriers stripped, and
their result carries the carriers. The nested-clustered fit strips them for its per-cluster outer
returns and sets them on the outer problem's fees, as it already resets the outer problem's `sets`.
An inner head of a meta-optimiser is viewed to a universe of investable assets, so its own door
meets a mask of `nothing` and it charges none. The Schur optimiser reads no fees today, so it
charges nothing until it gains them.

### The finite allocation charges the liquidation inside its own model

[#900](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/900) removed the cash
pre-adjustment and the whole price-carrying fee family with it, and
[ADR 0123](0123-the-finite-allocation-charges-its-fees-inside-its-own-model-on-the-money-it-buys.md)
records that decision. The allocation holds the share
counts and the prices, so it holds the money in each position exactly, and it charges every fee
inside its model, through a budget constraint on `x .* p`. The two liquidation carriers therefore
enter that model's own fee expression, priced on the complement's previous money, which the
`prev_cash` field of `FiniteAllocationInput` supplies. The requirement is that a result's reduced
`Fees` is accepted beside its full-length weights and prices, and that the liquidation is charged
one time on the complement.

### The per-asset split is one matrix, and the mask says which columns each axis owns

`calc_asset_fees` returns a pair of pairs, `((periodic, periodic_exit), (one_off, one_off_exit))`:
on each clock, the split on the investable axis and the charge per liquidated asset. The separation exists so each charge reaches the right place, and it ends
there: `calc_net_asset_returns` returns **one** matrix, on the caller's own universe, whose row
sums are the net return series.

`calc_net_asset_returns(w, X, fees, imsk)` takes the Investable Mask as a fourth argument,
defaulting to `nothing`. `w` and `X` span the caller's universe; `fees` spans the two reduced
axes, because the door sliced it. The mask is what reunites them, and `charge_asset_fees` charges
each axis in the columns it owns, by the same two steps: the per period vector on every
observation, and the one-off vector on the observation `fees.fa` names. A liquidated asset earns
no return, so its column is zero and holds its charge alone — the charge is neither smeared over
the assets that stayed nor carried in a matrix of its own.

Nothing is materialised for the one-off terms. They are a vector written into one row of the
columns their axis owns, or, under an `AmortisedFees`, folded into the per period vector and
charged on every row.

A `nothing` mask charges the investable axis alone. A `Fees` that carries `lq` or `flq` then has
nowhere to put its charge, and is refused with an `ArgumentError` naming the mask: dropping the
charge would understate the return, which is the defect this ADR exists to fix. Only a hand-built
`Fees` reaches that refusal, because a carrier is set by `port_opt_view`, which holds the mask.

`lq` and `flq` are set independently, and the verb that prices an unset carrier returns an empty
vector rather than a vector of zeros, because it holds no length to build one from. So each step
is skipped by its own vector and never by the other's, and `add_liquidation_terms` sums the two
terms of the axis when only one of them is set.

### What is not changed

The held-weights record expands with a zero at the liquidated asset, so the next fold charges it
no second time. No diagnostic is emitted, because the charge is handled, and the reduced `Fees` on
the result is the record. The holding costs `l`, `s`, `fl` and `fs` price the new weights, which
are zero for the exit, so nothing else is owed.

## Considered options

- **Charge nothing, and document that a forced sale is free.** Refused: a silently wrong number,
  which the map's Destination rules out.
- **Charge it behind a switch on `Fees`, default off.** Refused: a second switch for one rule, and
  the default keeps the wrong number.
- **Two bare rate fields and a previous-weights field on `Fees`.** Refused: `fees.w` and `tn.w`
  would carry one fact twice, and a guard would be owed.
- **One carrier holding `w`, the proportional rate and the fixed amount.** Refused: a new type with
  its own methods, and one field where two were asked for.
- **The mask travels through the view**, `port_opt_view(opt, idx, X, imsk)`, and a view with no
  mask empties the carriers. Refused: six view methods learn a fourth argument, the resolution of
  a departed name still needs its own fix, and a view with no mask silently empties the carriers.
- **The reduced `Fees` carries the mask**, and the carriers stay on the full universe. Refused as
  first named: a derived mask stored on a constraint object, which no constraint object did, and
  carriers that every charging site would have had to mask at charge time. The build of #1067 took
  half of it: the carriers stay on the complement, where every charging site already reads them,
  and the reduced `Fees` records the mask because a door with no record cannot tell a caller's
  full-universe carrier from a result's reduced one, and the two routes that avoid the record —
  the result-taking arities bypassing the door, or a value-level contract of "complement only" —
  each leave a `Fees` meaning two things at two doors.
- **The per-asset split excludes the charge, or spreads it pro rata.** Refused: the first breaks
  the identity between the split and the total, the second puts a number on an asset that did not
  cause it.

## Consequences

- A walk-forward over a delisting panel reports a lower return than before, by the liquidation
  charge of each exit. That is the released number moving, and it moves towards the truth.
- `Fees` and `FeesEstimator` gain two fields, so every constructor call that spells the positional
  form changes, and the docstrings of both, of `Turnover`, of `calc_fees`, of `calc_asset_fees`,
  of `calc_net_asset_returns` and of `predict` state the rule. `Fees` alone gains a third, `imsk`,
  last and defaulting to `nothing`, which a door writes and a caller never does.
- A value-level figure on an all-investable prior under a fee that states a carrier moves: it
  charged the whole book as a forced exit and now charges nothing, which is the released number
  moving towards the truth. A value-level figure on a masked prior, and every figure on a result,
  is unchanged.
- The seven fit sites that resolve a fee — the shared JuMP prelude, the two `HierarchicalRiskParity`
  methods, `HierarchicalEqualRiskContribution`, `NestedClustered`, `Stacking` and
  `SubsetResampling` — hoist their `fees_constraints` call above the door and gain one
  `investable_fees_view` binding. The finite allocation input learns the mask.
- Reporting a per-fold turnover that includes the liquidation, as the reference does, is not
  decided here: the library reports no per-fold turnover today, so there is no reader to be wrong.
