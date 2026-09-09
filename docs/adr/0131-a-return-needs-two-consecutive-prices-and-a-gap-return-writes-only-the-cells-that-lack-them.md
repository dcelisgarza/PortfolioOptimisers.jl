---
status: accepted
---

# A return needs two consecutive prices, and a Gap Return writes only the cells that lack them

## Context

[Map #955](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/955) designs the ingestion
layer from zero, so that a point-in-time asset panel is the natural case rather than a special one.
[ADR 0129](0129-the-ingestion-layer-seams-at-the-listing-span-and-the-clock-draws-the-pipeline-boundary.md)
fixed the pieces — `listing_span` derives a **Listing Span** on the price clock, `PricesToReturns`
converts, `universe_masks(span, R)` projects the span onto the returns clock and intersects it with
finiteness — and [ADR 0130](0130-a-universe-policy-is-fitted-and-the-only-fill-is-a-span-bounded-price-convention.md)
fixed the policy surface around them.

Neither fixed a **value**. A price gap of `k` observations on a listed asset admits more than one
return series, and the choice decides the length of the **Held Gap** the layer emits, what a fold's
test window reads on the observation an asset resumes trading, and whether a suspension is a wealth
event or a data hole. It also decides the arithmetic of the projection at an asset's inception,
which ADR 0129 named but left open: its *Avoid* line on the Listing Span says only that the
projection "is where the padding convention is resolved".

### The three candidate series

[Issue #957](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/957) measured that the
reference implementation produces the first two of these and never the third.

| Rule | Inside the gap | The observation that ends it | Wealth across the gap |
| --- | --- | --- | --- |
| no previous price | `NaN` × `k` | `NaN` | discarded: the catch-up appears nowhere |
| forward fill | `0.0` × `k` | the catch-up | conserved, as `k` flat observations and one jump |
| catch-up on re-pricing | `NaN` × `k` | the catch-up | conserved, with a Held Gap of exactly `k` |

### What the arithmetic already fixes

`prices_to_returns` computes returns with `TimeSeries.percentchange`, which reads two **consecutive**
prices. A run of `k` gapped prices therefore makes exactly the `k + 1` returns that read one of them
non-finite, and every later return of that column is finite. A gap is confined to its own column for
the same reason: no asset's return reads another's price.

Two consequences follow without a decision. The catch-up `P[t+k] / P[t-1] - 1` is computable by no
vectorised slice — it needs a per-column scan that remembers the last observed price. And forward
fill is not a return convention at all: it rewrites the *numerator* on the flat observations, which
is an act on prices, and ADR 0130 already owns it as `PriceGapFill` under a **Held Price**.

### What the masks can and cannot say

Map #955's decision 3 sets `emsk = amsk .& isfinite.(returns)`. Under a catch-up the re-pricing cell
is finite, so `emsk` reads true there and a `(k + 1)`-period return enters a one-period moment as one
draw, at roughly `√(k + 1)` the scale.

Flagging it non-estimable was measured to buy less than it looks. `coverage_mask` reads finiteness
and the active mask and **never** the estimation mask, so a plain moment estimator on the **Coverage
Universe** takes the cell either way; only a mask-aware estimator reading `min_obs` would refuse it.
And the Listing Span compresses to `first`/`last` per asset — exactly because an interior gap leaves
an asset listed — so it cannot record where an interior gap ended. A mask that disagreed with the
values would need state the span deliberately does not carry.

## Decision

### A return is the change between two consecutive observations

The default is the arithmetic: a return whose pair of prices does not exist is not a return. A run
of `k` gapped prices leaves `k + 1` non-finite returns, the Held Gap is `k + 1`, and the move across
the gap is recorded nowhere.

The rule is chosen because it fabricates nothing, and because it makes an interior gap and an
asset's inception one case rather than two. A newly listed asset has no return on its first
observation and nobody calls that a defect; a re-pricing after a suspension is a re-listing, and it
reads the same way. The cost is stated rather than hidden: `emsk` is false on an observation the
asset printed a price, and a fold that held the asset through a suspension shows it flat and never
books the re-pricing move.

### A Gap Return is the one override, and it is optional

`gap_return_alg` is a field on `PricesToReturns` and a keyword of `prices_to_returns`, bound to
`Union{Nothing, <:AbstractGapReturnAlgorithm}` and defaulting to `nothing`, which is the rule above.
`CatchUpGapReturn` is the one concrete member the library ships: it books the move on the
re-pricing observation, shortening the Held Gap to exactly `k`.

The family is an *algorithm* rather than a flag because the conventions are open — a caller who
wants a suspension's move spread across its observations is asking a question of the same kind — and
because the repository already spells this shape one directory over: `AbstractPanelFillAlgorithm`
with `NoPanelFill`, `ConstantPanelFill` and `ForwardPanelFill`, and the verb `panel_fill(alg, …)`.
`AbstractGapReturnAlgorithm` mirrors it, with the verb `gap_return`.

### A Gap Return writes only the cells the default rule left non-finite

`TimeSeries.percentchange` runs first under `ret_method`, and the algorithm then overwrites only the
cells it left non-finite. The writable set is narrower still: a non-finite cell **inside the asset's
Listing Span that has an earlier observed price in its column**. Everything else is frozen.

The invariant is what keeps the family from becoming a general returns rewriter. A cell computed
from two observed prices can never be altered, so the "a gap does not spread" property is held once
rather than re-argued per algorithm. And no algorithm can manufacture a return before an asset's
first price, nor on the row `padding = true` pads for every asset, because neither has an earlier
observed price to anchor on.

The alternative shapes were both rejected. Answering only *which earlier price* the return reads
keeps the return formula in one place, but fixes the family to a choice of anchor, so a spread is
not expressible. Answering the whole return column is fully general, and duplicates both
`ret_method` branches into every algorithm.

### The estimation mask follows the values

Decision 3 is unchanged: `emsk = amsk .& isfinite.(returns)`, read off whatever the conversion
emitted. There is one derivation, it is unaware of which algorithm produced the values, and the
library never quietly disagrees with the numbers it emitted. A caller who opts into a catch-up owns
the `(k + 1)`-period observation everywhere, which is the honest reading given that `coverage_mask`
would take the cell regardless.

The fold's accounting is unaffected either way, because the Held Gap zeroing reads finiteness rather
than the estimation mask.

### The span projects to `[first + 1, last]`

`universe_masks` shifts the opening bound of the Listing Span by one observation when it crosses to
the returns clock, because a return consumes the **earlier** price of its pair. An asset's first
return is one observation after its first price; its last return sits exactly on its last price.

This is the padding convention ADR 0129 left open, and it refines map #955's decision 2, which
marked the active mask from the first price on the *price* clock and said nothing about the
projection. Under it the active mask bounds exactly the run of a column's finite returns at both
ends, symmetric rather than off by one at the leading edge; the two masks differ only at an interior
gap, which is what makes an interior gap the one thing they disagree about; and no inception emits a
Held Gap. An asset with a single priced observation gives the empty interval, which is correct — one
price yields no return.

It remains a deterministic function of the price gaps, so ADR 0129's panel-wide license is
untouched, and it holds for the whole Gap Return family, since no algorithm can write before an
asset's first price.

ADR 0129 lets a caller's own `AbstractMatrix{Bool}` enter where the Listing Span does, and such a
declaration is not an interval, so `[first + 1, last]` does not name it. The rule generalises rather
than doubling: **a return is active exactly when both prices of its pair lie inside the
declaration**, `span[a, i] && span[a + 1, i]`, with `a` the earlier price of the pair under whichever
padding convention is in force. On an interval that is `[first + 1, last]` again — `first <= a` and
`a + 1 <= last` is `a + 1 ∈ [first + 1, last]` — so the Listing Span is a compression of the general
rule and not a special case beside it, and a constituency that leaves and rejoins books no return
across its absence, which is the same refusal the sentence above gives a re-pricing.

### A contradicted algorithm reports an `@info`

Under `nan_to_missing = true`, `dropmissing!` removes every row still holding a gap before the
conversion, so the table reaching `percentchange` is gap-free and the writable set is provably
empty for every algorithm. A non-`nothing` `gap_return_alg` that finds no cell to write reports an
`@info` at conversion.

It is neither a refusal nor a warning. A refusal would reject a configuration that computes a
correct answer, and a warning cannot distinguish a self-contradicting configuration from a panel
that simply holds no gap, so it would fire where nothing is wrong. Sweeping `nan_to_missing` to
escape the report is not available in any case: it moves the observation clock, and ADR 0129 rules a
clock-changing hyperparameter unsearchable in principle.

## Consequences

`CONTEXT.md` mints **Gap Return**, and tightens the *Avoid* line on **Listing Span**, which said the
projection is where the padding convention is resolved without saying how.

`AbstractGapReturnAlgorithm` is unexported, as the library's abstract types are. `CatchUpGapReturn`
is a type a caller names, so it is exported and owes a Capability Catalogue entry.

ADR 0129 and ADR 0130 are unchanged. Neither fixed a value rule, and the projection this ADR
specifies is the one ADR 0129 named and deferred.

The default emits a Held Gap of `k + 1` for a suspension of `k`, and `CatchUpGapReturn` emits one of
`k`. Nothing else about the Held Gap moves: ADR 0118 still names it through the strictness policy, a
warning by default and a refusal under `strict`, and still zeroes the missing returns of investable
columns before the drift compounds.

A spread of a gap's move across its observations is expressible under the invariant and is not
shipped. It is a convention no one has asked for, and the family is open precisely so that it costs
one type when someone does.
