---
status: accepted
---

# The fee clock a fold reports on is a switch of the scheme, and the fit keeps its own

## Context

[Issue #898](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/898) settled what a fee
charges over a horizon. `l`, `s` and `tn` are rates per period, so each of them charges on every
observation of a return series. `fl` and `fs` are charged one time for the whole holding period,
and the `fa` field of `Fees` names the clock they fall on: a `nothing` charges them on the first
observation, and an `AmortisedFees` spreads them evenly over the count the charging site hands in.

One field then served three readers, and one of them could not honour it.

| Reader | What it did with `fa` |
| --- | --- |
| the fit's net return series, `set_net_portfolio_returns!` | followed the clock |
| the fit's expected return, `add_fees_to_ret!` | always spread, because an expected return is a per period number |
| the fold's realised series, `charge_fees` through `predict` | followed the clock |

So a caller who wanted the optimiser to price a fixed fee one way and the **report** to charge it
the other could not say so. There was one field, and it reached both.

The post-hoc verbs already carried the flexibility. `calc_net_asset_returns(pred, fees)`,
`risk_contribution(r, pred, fees)` and their siblings take a `fees` argument, so a caller can
score a stored prediction under a fee of their own. What no caller could change is the series
`cross_val_predict` **stores** in `pred.rd.X`, which is built from the result's own fee. A
drawdown read off that series is therefore read off the optimiser's clock, whatever clock the fund
actually paid on.

The two questions are separate.

1. **What does the optimiser pay for a fixed fee?** A term in the objective, which shifts the
   argmin.
2. **What does the report charge for it?** A term in the realised series, which shifts the
   reported drawdown and nothing else.

They are the same kind of pair as the two evaluation switches of
[ADR 0110](0110-the-two-evaluation-switches-are-separate-and-the-library-does-not-bundle-them.md),
`wd` and `pws`, and that ADR's reasoning applies: one field for two questions offers one capability
fewer and explains nothing better.

## Decision

**The clock a fold reports on is a switch of the scheme, and it overrides the fee's own.**

`fa::Option{<:AbstractFeeAmortisation}` is a flat field on `KFold`,
`CombinatorialCrossValidation`, `IndexWalkForward` and `DateWalkForward`, beside `wd`.
`MultipleRandomised` and the two search estimators inherit it through their inner `cv`, exactly as
`wd` does. `fold_evaluation` reads it, and it travels to `predict` as a keyword beside `wd` and
`hwd`.

`nothing` inherits the clock the fee itself states, so every number a caller has today is unmoved
until that caller sets the field. This is the same off-state ADR 0110 gives its two switches.

**The switch reaches the fold's realised series and nothing else.** `predict` resolves it against
the fee it extracted from the result, through `override_fee_amortisation`, and charges the series
under the answer. The result the fold carries is the fit's, so `pred.res.fees` still states what
the optimiser priced. The fit's own `fa` is untouched, and the two JuMP readers above are untouched.

**The field is three-state, so the family gains a second leaf.** `nothing` on a scheme means
*inherit*, which is a third answer that `Fees.fa` does not need. So the two answers a `Fees`
spells `nothing` and `AmortisedFees()` need two words of their own:

| Leaf | The clock it names |
| --- | --- |
| `FirstObservationFees` | the two fixed terms are charged on the first observation of the series |
| `AmortisedFees` | the two fixed terms are spread evenly over the count the charging site hands in |

`Fees.fa` accepts the new leaf too, where it is a synonym for its own `nothing`.

**No behaviour hangs off `AbstractFeeAmortisation` any more.** Each of the five dispatch sites now
names its leaf: the first-observation arm takes `Union{Nothing, <:FirstObservationFees}` and the
spreading arm takes `AmortisedFees`. A third clock added to the family therefore gets a
`MethodError` until its own methods are written. The supertype names the question, and it decides
no answer.

## Alternatives refused

**A three-state `Bool`, `Option{Bool}`.** `nothing` inherits, `true` spreads, `false` charges the
first observation. One fewer type, and a `Bool` beside a stated algorithm is a state that means
nothing the day a third clock appears. The library's idiom is a named leaf, and a named leaf
leaves room for that third clock.

**Keeping the catch-all on `AbstractFeeAmortisation`.** The new leaf could have taken a method
that forwards to the `nothing` one, and the spreading arm could have stayed on the supertype. Then
a third clock added later would silently spread, because it would inherit the arm that answers a
question it never asked. The supertype means "a clock", not "spread".

**Bundling the clock with `wd`.** A caller who drifts a fold's series usually wants the fund's
reading of the fee as well, so a bundle would serve the common case. It would also make the two
readings inseparable, which is the fault ADR 0110 was written to avoid. The switches are cheap and
independent; the bundle is neither.

**A field on `Fees` naming the report's clock beside the fit's.** A `Fees` would then carry two
clocks, and every site that charges a fee would have to know which of the two it is. The question
belongs to the *scheme*, because it is the report that asks it, and a scheme already carries the
other two evaluation switches.

## Consequences

- A caller can report the drawdown a fund actually saw while the optimiser prices the fee the way
  the optimiser must.
- One more switch to explain. It sits beside `wd` and `pws` in each scheme's docstring, under its
  own *Fee clock* heading.
- No released number moves. `nothing` inherits, so a caller who sets nothing keeps what #898 gives
  them.
- A third clock added to `AbstractFeeAmortisation` costs five methods. That is the price of the
  supertype deciding nothing, and it is paid once per clock rather than silently by every caller.
