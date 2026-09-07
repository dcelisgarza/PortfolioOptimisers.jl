---
status: accepted
---

# The finite allocation charges its fees inside its own model, on the money it buys

## Context

A finite allocation turns a weight vector into integer share counts against a cash budget.
`setup_alloc_optim` charged the fee before either sub-problem ran:

```julia
if !isnothing(T) && !isnothing(fees)
    cash -= calc_total_fees(w, p, T, fees)
end
```

Ticket [#900](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/900) found four faults
in that line, and the last of them is defect 1 of
[#898](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/898).

- **It priced a portfolio that did not exist yet.** The estimate came from the *target* weights.
  What is owed is owed on the *realised* allocation, which is integer share counts and a leftover.
  Those are two different portfolios.
- **The allocator never saw the fee.** A fixed fee of `5.0` on a position worth `30.0` is a bad
  trade, and the model could not know it, because the money left `cash` before the model ran. It
  could neither drop the position nor weigh the fee against the tracking error.
- **It charged before the long and short split**, so both sides saw one net figure that was
  neither side's own fee.
- **The estimate was not money.** `calc_total_fees(w, p, T, fees)` charged `rate * dot(w, p)`. A
  weight is a dimensionless fraction of the capital and a price is currency per share, so
  `w ⊙ p` is a currency-per-share quantity that names nothing. Measured: on `w = [0.5, 0.5]`,
  `p = [100.0, 200.0]`, `cash = 1e6`, `l = 0.01` and `tn.val = 0.002` against `w₀ = [0, 0]`, the
  code charged `1.5` and `0.3` per period where the money says `10,000` and `2,000`. The gap is
  `cash / dot(w, p)`, and it is not a constant: double every price and the charge doubles, though
  no money moved.

The allocator is the one site in the library that holds the share counts and the prices, so it is
the one site that holds the money in each position exactly. It is therefore the site that can
price a fee correctly, and the price-carrying fee family existed only to serve the line above.

## Decision

### A fee is a cost of the portfolio the allocator buys, and it is priced on `x ⊙ p`

Every fee term is written against the money in each position, `x ⊙ p`. No weight and no price
appears on its own. `l`, `s` and `tn` are rates per period, so each of them charges on every one
of the `T` periods; `fl` and `fs` are currency amounts charged one time for the whole horizon.
That is the rule #898 settled, written in the allocator's own quantities. `Fees.fa` names where a
one-off cost lands on a *return series*, and the allocation reports no series, so it reaches no
term here.

### The fee enters the budget constraint, and never the objective

The objective reads *track well, and leave no capital idle*:

```julia
JuMP.@expression(model, r, cash - LinearAlgebra.dot(x, p))
fee = set_allocation_fees!(model, p, cash, sf)
JuMP.@constraint(model, cr, sc * (r - fee) >= 0)
JuMP.@objective(model, Min, so * (u + r))
```

A fee added to `r` and minimised would reward the model for paying **more** fees, because a larger
fee shrinks the leftover. So the objective does not change, and the budget learns that the fee must
be affordable. The objective still pushes `dot(x, p)` up, so every unit of fee competes with a unit
of position, and the model drops a position whose fixed fee buys too little tracking. Measured: a
third asset worth about `30` of a `1000` budget is held with no fee and dropped under a fixed fee
of `40`.

### The reported fee is priced on the realised shares, not read off the model

The turnover term needs an absolute value, which in a linear model is an epigraph, `t ≥ |m - m₀|`.
Only the budget constraint pushes `t` down. Whenever the budget does not bind, a solver may leave
`t` slack, and the model's own fee expression then overstates the charge. A slack epigraph is safe
— it only makes the budget more conservative, so the allocation the model bought is affordable
under the exact charge — but it is not the number to report. Both allocators therefore report
`allocation_fee` of the realised shares, which is exact. Measured on the `test_11` fixture, the
model's expression stood at `821.97` where the realised charge was `801.07`.

### The result carries the fee it paid

`DiscreteAllocationResult` and `GreedyAllocationResult` gain a `fees` field, between `cash` and the
trailing `fb`, so the generic `factory` still rebuilds them by swapping the last field alone. The
field is the sum of the two sides' charges and is never signed. `cash` is the cash left over after
both the shares and the fee, so `sum(cost) + cash + fees` is the cash the side started with.

### `FiniteAllocationInput` gains `prev_cash`

The turnover term needs the money held per asset **before** the trade. `Turnover.w` is a weight
vector, so that money is `prev_cash .* tn.w`. The field is supplied like `cash` and defaults to it,
so a caller who did not trade before states nothing. A short side is allocated with its weights
negated, so its previous money is negated too, and both sides then hold a non-negative figure.

### The whole price-carrying fee family is deleted

`setup_alloc_optim` was the only caller in `src/` of `calc_total_fees(w, p, T, fees)`. When it
stopped charging, the family that multiplies a weight by a price lost its last caller, so it goes
rather than being corrected:

- `calc_fees(w, p, …)` and `calc_asset_fees(w, p, …)`, all four method groups each
- `calc_periodic_fees(w, p, fees)` and `calc_asset_periodic_fees(w, p, fees)`
- `calc_total_fees(w, p, T, fees)` and `calc_total_asset_fees(w, p, T, fees)`

Defect 1 of #898 disappears with them. The no-price family stays, and so do the exported
`calc_total_fees(w, T, fees)` and `calc_total_asset_fees(w, T, fees)`, which answer a caller's own
question about the cost of a holding period. `setup_alloc_optim` also loses `p`, which it read only
to price that fee.

### The greedy allocator charges the same fee as it goes

`finite_sub_allocation!` holds no model. It buys shares against a running cash figure in two
passes, so the affordability test of each pass is a test against the cost of the purchase **plus**
what the purchase adds to the fee. That delta is exact and costs no loop:

- proportional: `T * l[i] * qty * p[i]`
- turnover: `T * tn[i] * (|m₁ - m₀ᵢ| - |m₀ - m₀ᵢ|)`, which can be **negative** when the purchase
  moves the position towards the money it held before the trade
- fixed: `fl[i]`, and only when the position was zero and becomes non-zero

A side that held money before the trade owes a turnover fee even when it buys nothing, because
selling out is a trade. Both allocators therefore charge `allocation_fee` against an empty book
before they start.

## Considered options

- **Correct the price-carrying family in place**, so that `calc_fees(w, p, cash, fees)` charges
  `rate * cash * w`. Refused. It would keep two families that answer the same question, and the
  corrected one would still price the *target* weights rather than the allocation that is bought.
  The allocator holds the share counts, so it needs no weight and no cash at all.
- **Add the fee to the objective.** Refused. A fee in `r` under a minimisation rewards paying more
  fees, because a larger fee shrinks the leftover.
- **Report the model's fee expression.** Refused. The turnover epigraph is slack whenever the
  budget does not bind, so the number would be an upper bound rather than the charge.
- **Emit the fixed-fee binaries always.** Refused. A problem that states no fixed fee keeps the
  variable count it had; only a stated `fl` or `fs` turns the sub-problem into a larger integer
  programme.
- **Charge the fee once against the whole cash before the split.** That is what the old line did,
  and it gives both sides one net figure that is neither side's own fee. Each sub-problem now takes
  its own rates: `l` and `fl` on the long call, `s` and `fs` on the short one.

## Consequences

- A released number moves at every finite allocation that states a fee. It moves towards the truth
  twice over: the fee is priced on the money rather than on `w ⊙ p`, and it is priced on the
  allocation that is actually bought.
- `DiscreteAllocation` gains `N` binaries whenever a fixed fee is set, so a problem that was an
  integer programme stays one but grows.
- `FiniteAllocationInput` and both result types gain a field, so a positional constructor call
  changes.
- A caller that read the price-carrying fee family has no replacement for it, and none is owed: the
  quantity it returned named nothing.
- [ADR 0121](0121-a-forced-liquidation-is-charged-through-two-fee-carriers-on-the-complement-of-the-investable-mask.md)
  states that the two liquidation carriers enter the allocation model's own fee expression, priced
  on the complement's previous money through this ADR's `prev_cash`. Its build,
  [#897](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/897), takes this decision.
