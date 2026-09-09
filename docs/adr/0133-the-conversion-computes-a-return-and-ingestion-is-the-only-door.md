---
status: accepted
---

# The conversion computes a return, and ingestion is the only door

## Context

[Map #955](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/955) asked one question of
[#980](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/980): whether a bare
`PricesToReturns()` carries a price gap or deletes it, its `nan_to_missing` field having been left
at `true` when the gap-carrying conversion was built. Measuring the default dissolved the question
into a larger one, and this ADR answers the larger one.

### What the default does, measured

A 40 × 4 price fixture, one asset delisted after observation 30 and one suspended for three
observations, converted by a bare call and by the same call with `nan_to_missing = false`:

| | `true` | `false` |
| --- | --- | --- |
| return rows, of 39 | **26** | **39** |
| assets kept | 4 | 4 |
| non-finite cells | 0 | 14 |
| `AssetPanel` emitted | none | yes |
| report | **none** | warns by name, naming `price_ingestion` |
| `MeanRisk` weights | `[0.276, 0.156, 0.329, 0.239]` | `[0.640, 0.360, 0, 0]`, plus an `@info` naming the two |

The released default places **0.329 on a delisted asset and 0.239 on a suspended one — 57% of the
book in instruments that cannot be held** — on a clock thirteen observations shorter than the one
the caller handed in, and says nothing. The gap-carrying path refuses both, reports both, and keeps
the clock. The deleting path is the silent one; the ticket had assumed silence ran in both
directions, and it does not.

### Why the default is not the defect

Three facts, each measured on `dev`, say the flag is a symptom.

1. **The conversion is stateless by declaration and decides fitted state anyway.**
   `fit_preprocessing(ptr::PricesToReturns, ::PricesResult) = ptr` — the estimator returns itself,
   which is the library's spelling for *this step learns nothing*. Inside it, `dropmissing!` and
   `select!(…, Not(names(X, Missing)))` delete observations and assets, and
   [ADR 0130](0130-a-universe-policy-is-fitted-and-the-only-fill-is-a-span-bounded-price-convention.md)
   rules that a judgement changing the asset universe is fitted on a training window and replayed by
   name. [#958](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/958) measured the
   consequence directly: a fold's train and test windows disagree on the universe, because the
   deletion is window-local.

2. **The library wrote the supersession down and did not finish it.** `MissingDataFilter`'s own
   docstring says it *"supersedes the `missing_col_percent`/`missing_row_percent` keywords of
   `prices_to_returns`, making the thresholds fitted state and independently tunable"*. Both
   keywords are still on the conversion, and beside them sits an unconditional deletion that has no
   keyword at all.

3. **The flag means two unrelated things.** `nan_to_missing` chooses a *spelling* for absence and,
   through it, a *universe policy*. It also silently selects which gaps an `impute_method` can see,
   because the `NaN` → `missing` step runs before the imputer and only under `true`:

   | gapped table spelled as | `true` | `false` |
   | --- | --- | --- |
   | `missing`, with `Impute.LOCF()` | filled | filled |
   | `NaN`, with `Impute.LOCF()` | filled | **not filled, 14 cells left** |

   `TimeSeries.merge` pads an outer join of ragged histories with `NaN`, so the unserved spelling is
   the one a real point-in-time source produces.

### The size of the verb

`prices_to_returns` carries seventeen keywords and does thirteen jobs: join, elementwise map,
collapse, spelling unification, imputation, row filtering, column filtering, unconditional deletion,
the return arithmetic, the gap-cell arithmetic, implied-volatility alignment, span projection and
panel slicing. Every one of the map's earlier decisions gave one of those jobs an owner —
[ADR 0129](0129-the-ingestion-layer-seams-at-the-listing-span-and-the-clock-draws-the-pipeline-boundary.md)
the clock-moving three, ADR 0130 the policies and the fill,
[ADR 0132](0132-the-layer-emits-one-carrier-and-alignment-splits-into-a-fixed-axis-and-a-provenance-check.md)
the emission — and none of them removed the job from the verb.

`PricesResult` already carries `X`, `F`, `B`, `iv`, `ivpa`, `pnl` and `span`. Every datum the
conversion reads is therefore already on the carrier, and every keyword that names one of them is a
second way to say what the carrier says.

## Decision

### A keyword survives on the conversion if and only if it changes the arithmetic of a return

That is the dividing rule, and it is the whole decision. Three keywords pass it — `ret_method`,
which chooses simple or log; `padding`, which chooses whether the first observation is kept; and
`gap_return_alg`, which chooses what a cell lacking two consecutive prices carries (ADR 0131).
Fourteen fail it, and each already has an owner.

```julia
prices_to_returns(pr::PricesResult;
                  ret_method::Symbol = :simple,
                  padding::Bool = false,
                  gap_return_alg::Option{<:AbstractGapReturnAlgorithm} = nothing)
```

| leaves the conversion | owner | why |
| --- | --- | --- |
| `F`, `B`, `iv`, `ivpa`, `pnl`, `span` | fields of `PricesResult` | they already are |
| `join_method`, `collapse_args` | `PriceIngestion` | they move the observation clock (ADR 0129) |
| `map_func` | an elementwise `Pipeline` step | ADR 0129 assigned it there |
| `nan_to_missing` | nobody — deleted | absence has one spelling, fixed at the door |
| `impute_method` | `PriceGapFill` | ADR 0130 fixed the layer's one fill |
| `missing_col_percent`, `missing_row_percent` | `MissingDataFilter` | its docstring already claims them |
| `strict` | `PriceIngestion` | it guards a carrier's provenance |

### Absence has one spelling, and it is fixed once at the door

`missing` and `NaN` are both live in the Julia ecosystem — `TimeSeries.merge` pads with `NaN`, a
wide table built from a tidy one leaves `missing` — so something must normalise them. That is a
boundary job, there is exactly one boundary, and `PriceIngestion` already does it with exactly one
answer.

`NaN` wins over `missing` because the returns level must carry absence into `X::Matrix{Float64}`. A
`Matrix{Union{Missing, Float64}}` would fail every `MatNum` bound in the library and every BLAS path
under it, and the library already spells a **Held Gap** `NaN`. After the door, nothing in the
library asks about spelling again: `is_missing_value` reads both because it stands at the door, and
no verb behind it needs to.

### `nan_to_missing` is deleted, not flipped

Flipping it would leave a stateless verb holding an unfitted universe policy behind a Boolean, which
is the arrangement ADR 0130 forbids everywhere else. Deleting it removes the arrangement. The
conversion carries the gap because there is nothing in it that could do otherwise, not because a
default says so.

### Deleting an observation or an asset is a fitted step, and `row_thr` gains `0`

`MissingDataFilter` owns every deletion. Its `col_thr` is already fitted — it records the surviving
names on the training window and replays them — and its `row_thr` is window-local, which is the
correct split and the one `dropmissing!` gets wrong by being window-local on both axes.

Its two thresholds are widened from `(0, 1]` to `[0, 1]`, so that `row_thr = 0.0` and
`col_thr = 0.0` spell *no gap is tolerated*. Under the released domain the nearest spelling is
`row_thr = 1e-6`, which expresses the intent by arithmetic accident rather than by saying it.

### Filling is `PriceGapFill`, and `impute_method` goes with the flag

ADR 0130 fixed the layer's one fill: `PriceGapFill`, off by default, bounded by the Listing Span so
it touches Held Gaps alone, stating a **Held Price** or a per-asset reduction. `impute_method` is an
unfitted, unbounded fill reaching the same job around that ruling, and its presence is what made the
spelling asymmetry above observable at all. It is deleted with the flag.

Whether `Imputer` — the fitted per-asset constant fill that predates `PriceGapFill` — survives
beside it is a consequence to measure, not something this ADR settles.

### The bare `TimeArray` call runs the layer

```julia
rd = prices_to_returns(X)          # X::TimeArray

# is exactly
pr = price_ingestion(PriceIngestion(), X)
rd = prices_to_returns(pr)
```

The friendliest call in the library stays a one-liner, and it is now the layer's own path rather
than a way around it. This is what makes the map's destination literally true of one line: a caller
hands raw prices spanning listings, delistings and suspensions and receives returns-level data whose
non-investable cells carry `NaN`, together with an `AssetPanel` stating the two universe masks.

### Two reports are deleted rather than reworded, and a refusal goes with them

`assert_span_convertible` exists because a span-carrying carrier can be handed to a conversion that
deletes the gaps the span describes. With no flag that combination cannot be written, so the check,
its warning and its `strict` refusal are deleted.

`returns_universe_masks`' window-local branch — *"the price carrier holds gaps and no Listing
Span"* — derives a span from the window when the carrier states none. A window-local span reads a
delisting that straddles the window end as an asset that was never listed, so the branch answers a
question it cannot answer correctly, and it answers it only for the windows that happen to hold a
gap. It is deleted, with its warning, its `strict` refusal and its `listing_span` call, leaving a
total two-line rule:

```julia
returns_universe_masks(::Nothing, ::Any, ::Any) = nothing, nothing
returns_universe_masks(span::AbstractMatrix{Bool}, P, R) = compress_all_true(universe_masks(span, R)...)
```

A hand-built carrier holding gaps therefore states **no universe** rather than a window-local guess
at one. Its gaps are still handled: with no panel the Coverage Universe reads finiteness alone, and
`pnl === nothing` keeps the single meaning ADR 0132 gave it.

Deleting that branch also removes a failure it causes. Measured on a 200 × 4 fixture with one
delisting and one suspension, `cross_val_predict` over a span-less gapped carrier throws from
`assert_universe_aligned` under **both** settings of the flag: under `true` because a window drops
the delisted asset entirely and the universes differ, and under `false` because the branch emits a
panel only for a window that happens to hold a gap, so train and test disagree on panel presence.
ADR 0132's *always emits a panel* is true of the layer's path and of no other. Under one door there
is no other.

## Consequences

- **The conversion has one job.** `prices_to_returns` computes returns from prices. It does not
  decide a universe, does not choose a spelling, does not fill, does not filter and does not move a
  clock.
- **Every universe policy is fitted, without exception.** ADR 0130 stated the rule and the
  conversion was the one place that broke it. The rule now holds by construction rather than by
  convention.
- **A hard break of the released preprocessing API.** Fourteen keywords are removed from
  `prices_to_returns` and from `PricesToReturns`'s fields. The map's destination accepts one, and
  weighs no option by what it costs to rewrite.
- **Nothing shipped changes numerically.** `examples/SP500.csv.gz` holds no gap in 8,313 × 20, and a
  gapless table converts bit-identically with or without the deleted flag, emitting no panel under
  either. Every example, user-guide page and doctest is unaffected.
- **A caller who wanted the dense matrix composes it**, and gains the fitted replay they did not
  have:

  ```julia
  Pipeline(; steps = (MissingDataFilter(; row_thr = 0.0),
                      PricesToReturns(),
                      EmpiricalPrior(), mr))
  ```

- **The shipped rule stops being contradicted.** `user_guide/08_Point_in_Time_Universe.jl` opens
  with *"nothing in the library silently drops an asset, back-fills a price, or treats a gap as a
  zero return"*. The released default silently drops observations. It no longer exists.
- **ADR 0129 and ADR 0130 are rewritten in place**, neither having reached `main`. Both stated the
  window-local fallback as a live path — ADR 0129 for the conversion's masks, ADR 0130 for
  `PriceGapFill`'s bound — and both paragraphs are replaced. A carrier with no span now states no
  universe and is not filled; `PriceGapFill` keeps its `strict` field, because it now reports that
  it filled nothing and why, and refuses under `strict`, rather than guessing a span from a window.
- **`CONTEXT.md`** mints no term. Its **Universe Policy** entry loses the exception the conversion
  was, and its **Span Rule** *Avoid* line is unchanged.
- **What this ADR does not settle**, left to the builds it graduates: whether `Imputer` survives
  beside `PriceGapFill`, and what `PriceIngestion`'s own defaults are once a bare `TimeArray` call
  runs it.
