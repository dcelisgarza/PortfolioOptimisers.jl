---
status: accepted
---

# A universe policy is fitted, never panel-wide, and the layer's only fill is a span-bounded price convention

## Context

[Map #955](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/955) designs the ingestion
layer from zero, so that a point-in-time asset panel is the natural case rather than a special one.
[ADR 0129](0129-the-ingestion-layer-seams-at-the-listing-span-and-the-clock-draws-the-pipeline-boundary.md)
fixed the layer's pieces and their seams: `listing_span` reads the raw price gaps and emits a
**Listing Span**, `PricesToReturns` converts, and `universe_masks(span, R)` projects the span onto
the returns clock and intersects it with finiteness to give the active and estimation masks.

Once a gap is *carried* rather than removed, dropping a row or a column stops being a structural
necessity and becomes a caller's policy. What that policy surface is, and where it sits, is what
this ADR settles.

### The license ADR 0129 granted, and its exact ground

ADR 0129 derives the Listing Span **panel-wide**, where `assert_split_position` otherwise forbids
any work before the split. It granted itself that license on one ground, and the ground is narrow:
a listing calendar is **a fact about the instruments, not an estimate from returns**. The split rule
guards against a *stateful* step learning from held-out rows, and a span learns nothing.
[Issue #958](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/958) then measured that
the panel-wide derivation is leak-free: identical Coverage Universe and identical weights against a
per-window derivation on every fold.

A coverage floor, a sparsity threshold and a survivorship filter are none of them facts about
instruments. Each is a number applied to a quantity computed from the data. Whether the license
extends to them is the question every remaining item on this ticket hangs on.

### What the reference does

[Issue #957](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/957) measured five knobs
in the reference implementation, and all five run panel-wide, before any split: a missing-share row
filter, an unlimited forward fill of prices (on by default), an inception cut that drops every row
holding a gap (on by default), an endpoint survivorship filter that keeps a column iff its first and
last rows are finite (no threshold, on by default), and a per-estimator warm-up. Under the span
rule, "last row finite" is exactly "not delisted by the end of the panel", so the reference's default
decides today's universe from the end of the sample. Separately, its *panel* fill takes a limit and
stops at the active boundary, which its *ingestion* fill does not.

### What the library already holds

Three mechanisms bear directly on the question, and all three were measured on `dev`.

- **The selector family.** `AbstractAssetSelector` (ADR 0029) is a returns-level, universe-dropping
  family whose funnel fits on the training window, records **names**, and replays them on every
  later window. `port_opt_view` slices the Asset Panel's masks and Panel Fields out with the column,
  so `assert_panel_masks` survives a drop by construction.
- **The Coverage Universe.** `coverage_mask` drops a column that is non-finite *or* inactive at any
  row of the window. On a gapped panel that is exactly "complete history over this window", and
  every Prior Estimator applies it at its entry (ADR 0117), as do the pre-selection funnel and a
  cross-validation subset draw.
- **The warm-up.** `min_obs` is an established field with a `field_dict` entry, carried by the
  estimators that can consume a partial column, and answering `NaN` before it is reached.

Two facts constrain what a policy may promise. `fit_preprocessing(sel::AbstractAssetSelector, rd)`
calls `coverage_reduction(rd)` **before** `select_assets`, so a selector is handed a window whose
every column is 100% complete by construction. And `coverage_mask` reads finiteness and the active
mask and **never** the estimation mask, which `CONTEXT.md` states outright.

## Decision

### A fact is derived panel-wide; a judgement is fitted

ADR 0129's license does not extend. Only a fact about the instruments — the Listing Span, and the
two masks projected from it — is derived over the whole panel. Anything computed from returns that
changes the asset universe is a **Universe Policy**: fitted on a training window, replayed by name,
never derived panel-wide.

A caller-stated threshold does not escape the rule by being the caller's. The *number* is declared,
but the *set it selects* is computed from the whole panel, so the universe would be decided using
observations that postdate every fold. That is survivorship bias in its exact definition. The
legitimate half of a panel-wide declaration — a caller who genuinely holds a listing calendar or an
index constituency — is already served by ADR 0129's `AbstractMatrix{Bool}` bound on the span, with
no policy machinery at all.

Recomputing a policy per window without fitting it is also refused: the universe then moves between
train and test, which is the failure #958 measured for a window-local deletion, and
`assert_universe_aligned` refuses the fold because the terminal weights are indexed by the training
universe.

### The layer offers no filter

There is no sparsity filter, no minimum-history filter and no survivorship filter in the ingestion
layer. "Not enough history" already has two answers, each at the altitude that can act on it.

A **plain** moment estimator cannot consume a partial column at all, so its floor is all-or-nothing,
and it is the **Coverage Universe**. A caller who leaves everything unset already gets the behaviour
a sparsity filter would buy: an asset with two weeks of history in a ten-year window is not in the
Coverage Universe, its `mu` and its `sigma` diagonal are `NaN`, it is outside the Investable Mask,
and it holds exactly zero — with the mask saying why.

A **mask-aware** estimator can consume a partial column, so its floor is a warm-up, and it is
`min_obs` on that estimator. Only the estimator knows how many observations its own recursion needs,
so the number belongs to it.

A third number in the ingestion layer would have to be reconciled with both, and it could not even
be expressed in the family that would own it: a coverage threshold below 1.0 placed in the selector
family is vacuous, because the funnel reduces to the Coverage Universe first and hands the selector
columns that are complete by construction. Buying such a threshold would cost a second entry point
into a funnel whose single invariant is that no selector can forget the reduction.

A survivorship filter therefore exists in this library only in its per-window form, where it is not
bias but simply "the assets live throughout the training window" — and in that form it is the
Coverage Universe, which every Prior Estimator already applies. Its panel-wide form is offered
nowhere, and could not be a default anywhere.

### The one fill is a span-bounded price convention

The layer offers exactly one fill, and it is off unless a caller asks for it.

A fill exists to **state a price convention**, not to remove a gap. That is what changed: a fill
used to be the way to stop a deletion step removing the row, and the row is no longer removed. The
convention a caller most often wants is that across a suspension or a holiday *the price did not
move*, so the returns inside the gap are flat and the move across it lands on the first priced
observation. That price is the **Held Price**, and carrying it forward is what a Gap Fill states.

A per-asset constant cannot state that convention. Filling a suspension with an asset's median price
manufactures two large moves the market never printed, one into the gap and one out of it. So the
fill's convention field admits a carried price as well as a per-asset reduction, and the carried
price is the one a caller reaching for a fill normally wants.

The fill is **bounded by the Listing Span**. It fills only where `first[i] <= t <= last[i]`, so it
touches Held Gaps alone and can never fabricate a price where the asset was not yet listed or has
been delisted. This is the guarantee the reference's ingestion fill lacks and its panel fill has.
The span is asked of the carrier, and a carrier that states none **is not filled**: under
[ADR 0133](0133-the-conversion-computes-a-return-and-ingestion-is-the-only-door.md) every carrier
the layer builds states one, so a carrier without a span was built by hand and the step has no
listing calendar to bound itself by. It says so — it warns that it filled nothing and why, and
refuses under `strict` — rather than filling window-locally, because a window-local span reads a
suspension straddling the window's edge as an inception and fills the wrong cells. That ADR rewrote
this paragraph, which previously derived the span window-locally and filled from it.

### A fill runs at the price level; a drop runs at the returns level

ADR 0129's consequences said the policy surface "sits at the returns level, after the conversion".
That is correct for a drop and wrong for a fill, and this ADR replaces it with a rule that splits by
what the policy does.

A **fill** runs at the price level, after the span and before the masks. A carried price is a
statement about a price, so it cannot be expressed after the conversion at all: across a gapped run
`p₀, _, _, p₃` the unfilled returns are `NaN, NaN, NaN` — three gaps for a two-observation gap,
because the first priced day reads a missing previous price, which #957 measured — and zeroing all
three discards the `p₀ → p₃` move entirely. Carried forward at the price level the same run gives
`0, 0, p₃/p₀ - 1`, which conserves wealth.

The mask follows from the position rather than needing a state of its own. A price-level fill runs
before the conversion, so a filled cell is finite in the returns and `emsk = amsk .& isfinite.(R)`
is `true` there. A caller who filled has said the asset traded, and the estimation mask tells the
truth about the data as converted. No fourth mask state is minted, and none could be read if it
were: `coverage_mask` reads finiteness and the active mask and never the estimation mask, so a
finite cell flagged non-estimable would enter every plain moment estimator's fit regardless.

A **drop** runs after the span, at either level. At the returns level it is an
`AbstractAssetSelector`, and `port_opt_view` slices the masks and the Panel Fields out with the
column, so `assert_panel_masks` holds by construction: both masks are still present together, still
match the Panel Fields' shape, and `emsk ⊆ amsk` is untouched by a column removal. At the price
level it is `MissingDataFilter`, and `span_carrier_view` slices the span with the rows and columns
that survived rather than re-deriving one — which is what keeps a row drop from erasing the trailing
gaps a delisting is read from.

The span is read before either, so neither can move it. A fill cannot erase a delisting, and a drop
cannot resurrect one. That ordering, not the level, is the rule:
[ADR 0133](0133-the-conversion-computes-a-return-and-ingestion-is-the-only-door.md) makes the
price-level drop the route that replaces the conversion's own deletion, and rewrote this paragraph,
which previously placed every drop at the returns level.

### Names

`PriceGapFill` is the estimator, an `AbstractPricesPreprocessingEstimator` and an ordinary
`Pipeline` step. Its convention field is `fill`, bound to `Union{CarriedPrice, Num_VecToScaM}`:
`CarriedPrice()` is a fieldless singleton stating the Held Price convention, and a `Num_VecToScaM`
is the per-asset reduction, reusing the union the library already spends on a
vector-to-scalar measure. It carries `strict` beside it, because the refusal above is the
estimator's own and `fit_preprocessing` takes no keywords: the strictness policy reaches the step
through a field or not at all.

One result type serves both conventions, as `AssetSelectorResult` serves the whole selector family:
`PriceGapFillResult` holds `nx` and one value per asset. Under `CarriedPrice` that value is the last
observed **training** price, which seeds a carry-forward when a window opens inside a gap; under a
reduction it is the reduced scalar. `apply_preprocessing` then runs the convention forward through
the window, seeded by the fitted value and bounded by the span. The result carries `fill` and
`strict` too, and neither is a copy for the reader's benefit: the fitted object is what runs on an
unseen window, so it is where the convention `apply_preprocessing` dispatches on has to live, and
where the refusal has to be read from.

The span the fill is bounded by is asked of the carrier rather than derived by the step, so the two
never disagree. That question is one verb, and under ADR 0133 every carrier the ingestion layer
builds answers it, so on the layer's path the fill is always bounded. A hand-built carrier is the
only one that answers `nothing`, and the step reports rather than guesses.

The name deliberately does not reuse `gap_fill_value`, which is a trait on a covariance estimator
naming the value it substitutes internally for a gap it was handed. That is an estimator's private
repair; a Held Price is a caller's statement about the data.

### One estimator occupies the fill role, and `Imputer` is not it

"Exactly one fill" is literal. `Imputer` and `ImputerResult` are removed, and `PriceGapFill` is the
whole of the fill role.

`Imputer` predates the Listing Span. It is fitted and replayed, which is the half this ADR requires,
but it is `PriceGapFill` with one guard missing and one convention missing, and neither omission is
a case it serves:

- **Its configuration space is contained.** `Imputer.stat` is bound to `Num_VecToScaM`;
  `PriceGapFill.fill` is bound to `Union{CarriedPrice, Num_VecToScaM}` — the same union with the
  Held Price added.
- **The two fit the same number.** Both collect a column's observed prices, skip a column that holds
  none, and reduce the rest; `gap_fill_seed(::Num_VecToScaM, obs)` *is* the
  `vec_to_real_measure(stat, obs)` call `Imputer` makes.
- **The two applies differ by one guard**, `span[t, j] &&`, which is the bound this ADR exists to
  place.

So the only behaviour `Imputer` holds alone is the **unbounded** fill — a price written before an
asset's first listing or after its delisting. That is not a case the layer declines to serve by
oversight; it is the fabrication the bound was introduced to forbid, and it is the one thing
`Imputer` actually does on this repo's own fixtures. Measured on an eight-observation, three-asset
panel with one inception, one suspension and one delisting: `Imputer(; stat = MedianValue())`
invents a price for the asset that had not yet listed and for the asset that had been delisted, and
`PriceGapFill(; fill = MedianValue())` under the same reduction writes only the suspension.

**A caller who genuinely wants the unbounded fill is not blocked, and is not given a switch.** They
state a listing calendar in which nothing is ever unlisted:

```julia
pr = PricesResult(; X = Xm, span = trues(size(values(Xm))))
Pipeline(; steps = (PriceGapFill(; fill = MedianValue()), PricesToReturns(), …))
```

That is ADR 0129's caller-calendar route, used as intended, and it is **bit-identical** to
`Imputer(; stat = MedianValue())` — verified on the panel above. The difference is where the
assumption is written: a declared span states *every asset is listed at every observation*, which
is a claim a reader can check against the data, where an unbounded fill states nothing and is
checked by nobody. The bound is therefore a property of the type rather than a default on it, and
no `bound` field is added to `PriceGapFill` to hold the alternative.

## Consequences

`CONTEXT.md` mints **Universe Policy** and **Held Price**, and the *Avoid* line on Held Price
separates it from `gap_fill_value`.

ADR 0129 is rewritten in place rather than amended: its decision has not reached `main`, so no
reader outside the branch has seen the sentence being replaced. The sentence in its *Consequences*
placing the whole policy surface at the returns level is replaced by the split rule above.

The five knobs #957 measured in the reference map onto this library as follows. The row filter and
the inception cut have no counterpart, because a gap is carried rather than deleted. The unlimited
forward fill becomes an opt-in `PriceGapFill` under `CarriedPrice`, bounded by the span. The
endpoint survivorship filter has no panel-wide counterpart and never will; its per-window reading is
the Coverage Universe. The per-estimator warm-up is `min_obs`, and stays with its estimator.

`AbstractAssetSelector`, its funnel and the Coverage Universe are unchanged. No abstract type is
added, no `Pipeline` slot is added, and `assert_split_position` keeps its rule: a `PriceGapFill` is
a stateful step and sits after the split like any other.

**`Imputer` and `ImputerResult` are removed**, and with them a second estimator in the fill role and
a second entry in the `Pipeline` step list a caller reads first. This is a hard break, and the map's
destination accepts one. `MissingDataFilter` keeps `src/03_InputData/09_PriceFilters.jl`; the
Catalogue's `Cap(:Imputer, :ImputerResult)` row goes, as do the two type-hierarchy rows and the
`docs/src/api/03_Preprocessing.md` entries.

**The examples' leakage exemplar moves to `CarriedPrice`.**
`examples/5_validation_tuning/03_Pipelines.jl` §2.3 teaches that a fill's parameters are fitted
state by fitting an imputer on two windows and showing the numbers disagree. It keeps teaching that,
through `PriceGapFill` on a carrier that states a span: the fitted value is the last observed
*training* price, which disagrees across two windows exactly as sharply, and the page now teaches
the Held Price and the Span Rule alongside the leakage lesson. The fixture's leading run on one
asset stays non-finite and the Asset Panel keeps it out of the weights, which is the point the old
text could not make while the imputer was filling it. That page's prose about the conversion
silently dropping assets is stale under ADR 0133 and is rewritten in the same change.

**`CONTEXT.md` needs nothing.** It never named `Imputer`, and its **Held Price** entry already
speaks of *the* fill in the singular and states the Listing Span bound as that fill's property. The
removal makes the entry true rather than aspirational.

The descriptor whose warm-up exceeds a fold's training window is not settled here. It is a fact
about a descriptor rather than about ingestion, and it is owned elsewhere.
