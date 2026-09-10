#=
# Data preprocessing and the ingestion layer

Real price data is rarely clean. Assets list partway through the window (leading gaps), trade
halts and stale quotes leave flat or missing stretches, names get delisted (trailing gaps), and
exchanges keep different holiday calendars so timestamps do not line up. Feeding that straight
into an optimiser is a recipe for silent errors.

The library answers it in one place. [`price_ingestion`](@ref) is the door: it unifies the two
spellings an absent price arrives under, joins and collapses the series, and reads the **Span
Rule** off the panel to state each asset's **Listing Span** — where a leading run of gaps is an
asset not yet listed, a trailing run is a delisting, and an interior run is a suspension on an
asset that is still listed. [`prices_to_returns`](@ref) then computes returns and **carries every
gap**, handing the [`ReturnsResult`](@ref) an [`AssetPanel`](@ref) that says which assets are in
the universe and which of them can be estimated at each observation.

Nothing on that path deletes an observation or an asset, and that is deliberate: an observation
deleted for one asset's gap is an observation lost for every other asset too. Deleting is a
separate, *fitted* step ([`MissingDataFilter`](@ref)), and so is filling
([`PriceGapFill`](@ref)) — a **Universe Policy** is fitted on a training window and replayed by
name, or a walk-forward scores a fold against a universe the future chose.

!!! tip "When to reach for this"
    Reach for the layer whenever the raw price table has missing values — newly listed or
    delisted assets, halted or stale prices, or non-overlapping trading calendars. The decision
    for each gap is *carry, fill, or drop*: carrying is the default and needs no configuration,
    filling states a price convention across a suspension, and dropping is for an asset too
    sparse to trust.
=#

using PortfolioOptimisers, PrettyTables, DataFrames, Statistics

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. A clean slice, then realistic damage

We start from the usual S&P 500 slice and deliberately injure it to mimic the messes above: one
asset that is mostly missing (a late lister), a halted block of one asset, and scattered
single-day gaps across the rest.
=#

using CSV, TimeSeries

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
ts = timestamp(X)
nx = string.(colnames(X))
T, N = size(values(X))

vals = Matrix{Union{Float64, Missing}}(values(X))
vals[1:160, 3] .= missing                 # asset 3: a late lister, missing for 160 days
vals[100:120, 2] .= missing               # asset 2: a ~3-week trading halt
using StableRNGs
rng = StableRNG(42)
for _ in 1:60                              # scattered single-day gaps elsewhere
    vals[rand(rng, 1:T), rand(rng, 4:N)] = missing
end
Xmiss = TimeArray(ts, vals, Symbol.(nx))

#=
## 2. Diagnose the missingness

Before deciding what to carry, fill or drop, measure it. The per-column and per-row missing
fractions are the quantities [`MissingDataFilter`](@ref)'s two thresholds count.

Their names read as the axis that is *counted*, not the axis that is dropped: `col_thr` counts the
missing rows of a column and drops the column, and `row_thr` counts the missing columns of a row
and drops the row. Both default to `1.0`, so nothing is dropped unless you ask, and both admit
`0.0`, which tolerates no gap at all. The conversion itself drops neither axis.
=#

col_missing = vec(mean(ismissing, vals; dims = 1))
worst_cols = sort(DataFrame(; asset = nx, missing_frac = col_missing), :missing_frac;
                  rev = true)
pretty_table(first(worst_cols, 6); formatters = [resfmt],
             title = "Missing fraction per asset (worst six)")

#=
## 3. Ingest, and read the Listing Span

`price_ingestion` unifies `missing` and `NaN` as `NaN` — the one spelling the returns level can
carry in a `Matrix{Float64}` — and reads the Span Rule off the whole panel. Reading it *once*,
outside any fold, is what makes it a fact about the instruments rather than a judgement a window
made.

Asset 3 is missing for its first 160 observations, so the span says it is not yet listed there.
Asset 2's halt is interior, so the span says it is listed throughout and the halt is a **Held
Gap**.
=#

pr = price_ingestion(PriceIngestion(), Xmiss)
span = Matrix(pr.span)
pretty_table(DataFrame(; asset = nx[[3, 2, 4]],
                       listed_from = [findfirst(view(span, :, j)) for j in (3, 2, 4)],
                       listed_to = [findlast(view(span, :, j)) for j in (3, 2, 4)],
                       observations = fill(T, 3));
             title = "The Span Rule: a leading run is not-yet-listed, an interior one is a halt")

#=
## 4. Carry the gaps into the returns

The conversion computes returns and nothing else. Every asset keeps its column, every date keeps
its row, and the cells a gap left behind are non-finite. A run of `k` gapped prices makes exactly
the `k + 1` returns that read one of them — the gap does not spread, because no asset's return
reads another's price.

The [`AssetPanel`](@ref) that comes back states the universe: `amsk` is the span projected onto
the returns clock, and `emsk` is that intersected with finiteness.
=#

rd = prices_to_returns(PricesToReturns(), pr)
pretty_table(DataFrame(;
                       quantity = ["assets", "observations", "non-finite return cells",
                                   "estimable cells"],
                       value = [length(rd.nx), size(rd.X, 1), count(!isfinite, rd.X),
                                count(Matrix(rd.pnl.emsk))]);
             title = "Nothing is deleted, and the panel states what is estimable")

#=
## 5. State a price convention across the halts

Carrying a gap is enough for everything downstream — the Coverage Universe excludes an asset from
the windows it spoils. A caller who instead wants to *say* what happened during a halt reaches for
[`PriceGapFill`](@ref): it is fitted on a training window, replayed by name, and bounded by the
Listing Span, so it touches Held Gaps alone and can never fabricate a price where an asset was not
yet listed or has been delisted.

[`CarriedPrice`](@ref) states the **Held Price** — the last priced observation carried forward — so
a halt becomes a flat stretch and zero returns through it, with the whole move landing on the
observation that ends it. A per-asset reduction such as `MedianValue()` states a constant instead,
which manufactures two moves the market never printed. Both conserve wealth across the gap; they
differ in where the move lands.
=#

fill_res = fit_preprocessing(PriceGapFill(), pr)
pr_filled = apply_preprocessing(fill_res, pr)
rd_filled = prices_to_returns(PricesToReturns(), pr_filled)

pretty_table(DataFrame(; table = ["carried", "filled with the Held Price"],
                       non_finite = [count(!isfinite, rd.X), count(!isfinite, rd_filled.X)],
                       asset_3_still_unlisted = [true, !isfinite(rd_filled.X[1, 3])]);
             title = "The fill closes the halts, and stops at asset 3's listing")

#=
## 6. Drop what is too sparse to trust

Asset 3 is missing ~64% of the window. Carrying it costs nothing — it simply never enters a
cross-section it cannot be estimated in — but a caller who wants it gone entirely says so with
[`MissingDataFilter`](@ref), and only with it: the conversion deletes no asset and no observation,
because deleting either is a **Universe Policy** and a policy is fitted on a training window and
replayed by name. That is exactly what this step does — it records the surviving names on the
training window and replays them — so the universe is not re-chosen with each window's own
hindsight.
=#

mdf = fit_preprocessing(MissingDataFilter(; col_thr = 0.5), pr)
println("Assets kept after the 50% column filter: $(length(mdf.nx)) of $N")

#=
## 7. Straight into the pipeline

A [`ReturnsResult`](@ref) carrying a panel is an ordinary one — it flows into the rest of the
package with no special handling, and the panel is what keeps an asset the window cannot estimate
out of the weights.

Solving on both tables shows what the fill buys. The **Coverage Universe** admits an asset only
where its return is finite over the whole window, so the scattered single-day gaps alone exclude
most of the book from the carried table. Closing the Held Gaps brings them back, and asset 3 stays
out because it was not listed — which is the distinction the span exists to draw.
=#

using Clarabel
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
function mv(rr)
    return optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), rr),
                                                 slv = slv)))
end
res = mv(rd)
res_filled = mv(rd_filled)
pretty_table(DataFrame(; table = ["carried", "filled with the Held Price"],
                       estimable = [count(res.imsk), count(res_filled.imsk)],
                       active_names = [count(>(1e-6), res.w), count(>(1e-6), res_filled.w)]);
             title = "The panel keeps what the window cannot estimate out of the weights")

#=
## 8. Visualising the damage

A heatmap of the missingness mask makes the structure obvious: the wide band is asset 3's late
listing, the short block is asset 2's halt, and the speckle is the scattered gaps.
=#

using StatsPlots, GraphRecipes
heatmap(1:N, 1:T, Float64.(ismissing.(vals)); xlabel = "Asset", ylabel = "Day",
        colorbar_title = "missing", title = "Missingness pattern (raw data)", yflip = true)

#=
## Summary

The ingestion layer is the single entry point for cleaning price data:

  - [`price_ingestion`](@ref) unifies the two absent-price spellings, joins and collapses the
    series, and reads the **Listing Span** off the whole panel.
  - [`prices_to_returns`](@ref) computes returns and carries every gap, handing back an
    [`AssetPanel`](@ref) that states which assets are estimable when.
  - [`PriceGapFill`](@ref) states a price convention across a **Held Gap**, bounded by the span;
    [`MissingDataFilter`](@ref) deletes what is too sparse to trust, and it is the only thing that
    deletes anything. Both are fitted on a training window and replayed, because a universe chosen
    with hindsight is a look-ahead.
=#

#=
!!! note "ADR 0133"
    `nan_to_missing`, `impute_method`, `missing_col_percent` and `missing_row_percent` used to live
    on `prices_to_returns`, and the default deleted every observation row holding a gap. ADR 0133
    removed all four: a keyword survives on the conversion if and only if it changes the arithmetic
    of a return. Filling is [`PriceGapFill`](@ref)'s and deleting is [`MissingDataFilter`](@ref)'s,
    whose thresholds admit `0.0` for *no gap is tolerated*.

    Three keywords pass the rule — `ret_method`, `padding` and `gap_return_alg` — and the rest of
    what the conversion used to take is now a field of the [`PricesResult`](@ref) it reads.
    `join_method` and `collapse_args` move the observation clock, so they belong to
    [`PriceIngestion`](@ref). `prices_to_returns(X)` on a bare price table is exactly
    `prices_to_returns(price_ingestion(PriceIngestion(), X))`, so the friendliest call in the
    library is the layer's own path, and a caller wanting a different join, a collapse, a declared
    span, or factor and benchmark series writes the two steps.
=#
