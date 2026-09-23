#=
```@meta
Description = "Cleaning real price data with price_ingestion: leading and trailing gaps, stale quotes, halts and mismatched calendars, before any optimiser sees it."
```

# Data preprocessing and the ingestion layer

Real price tables have gaps. An asset that lists partway through the window has no prices before
its listing, which is a leading gap. A delisted asset has none after it, which is a trailing gap. A
trading halt or a stale quote leaves a flat or missing stretch in the middle. Exchanges keep
different holiday calendars, so the dates of two assets do not line up.

Two functions prepare such a table for the rest of the library. [`price_ingestion`](@ref) treats
`missing` and `NaN` as the same absent price. By default it puts any factor and benchmark series
on the dates of the price table, and it collapses the series to a lower frequency only when you
give [`PriceIngestion`](@ref) a non-empty `collapse_args`. It states each asset's listing span,
the observations between its listing and its delisting. It takes the span
from the whole table. A run of gaps at the start of an asset's column means the asset was not yet
listed, a run at the end means it was delisted, and a run in the middle is a suspension of an asset
that stayed listed. [`prices_to_returns`](@ref) then computes the returns and keeps every gap in
them. The [`ReturnsResult`](@ref) it returns holds an [`AssetPanel`](@ref), which states the assets
in the universe and the ones the library can estimate at each observation.

Neither function deletes an observation or an asset. A row deleted because one asset has a gap
there is a row lost for every other asset. Deleting is a separate step,
[`MissingDataFilter`](@ref), and so is filling, [`PriceGapFill`](@ref). Each is fitted on a
training window, which records its choice of assets, and applied to later windows with that same
choice. We call such a step a universe policy. If a walk-forward chose the assets afresh on every
window, it would score each fold against a universe that the future of that fold chose.

!!! tip "When to reach for this"
    Reach for these functions whenever the raw price table has missing values: assets that list or
    delist inside the window, halted or stale prices, or trading calendars that do not overlap. For
    each gap you choose to carry it, fill it or drop the asset. Carrying is the default and needs no
    configuration. Filling states a price convention across a suspension. Dropping removes an asset
    with too few prices to estimate.
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
## 1. A clean slice with gaps added

We start from the S&P 500 slice the other examples use and add three kinds of gap to it. Asset 3
is missing for most of the window, like an asset that lists late. Asset 2 is missing for a block of
three weeks, like a halt. Sixty single days are missing, scattered over the other assets.
=#

using CSV, TimeSeries

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
ts = timestamp(X)
nx = string.(colnames(X))
T, N = size(values(X))

vals = Matrix{Union{Float64, Missing}}(values(X))
vals[1:160, 3] .= missing
vals[100:120, 2] .= missing
using StableRNGs
rng = StableRNG(42)
for _ in 1:60
    vals[rand(rng, 1:T), rand(rng, 4:N)] = missing
end
Xmiss = TimeArray(ts, vals, Symbol.(nx))

#=
## 2. Measure the gaps

Before you choose what to carry, fill or drop, measure the gaps. We compute the fraction of
missing prices in each asset's column. [`MissingDataFilter`](@ref) compares this fraction, and the
same fraction for each row, against its two thresholds.

Each threshold is named after the axis it counts, not the axis it drops. `col_thr` counts the
missing rows of a column and drops the column. `row_thr` counts the missing columns of a row and
drops the row. Both default to `1.0`, so the filter drops nothing unless you ask it to. Both accept
`0.0`, which drops a column or a row with a single gap. The conversion to returns drops neither.
=#

col_missing = vec(mean(ismissing, vals; dims = 1))
worst_cols = sort(DataFrame(; asset = nx, missing_frac = col_missing), :missing_frac;
                  rev = true)
pretty_table(first(worst_cols, 6); formatters = [resfmt],
             title = "Missing fraction per asset (worst six)")

#=
## 3. Ingest the prices and read the listing span

`price_ingestion` writes every `missing` as `NaN`, because a `Matrix{Float64}` of returns can hold
`NaN` and cannot hold `missing`. It finds the listing span once, from the whole table, outside
any training window, so the span describes the asset and does not depend on the window a later step
fits on.

Asset 3 is missing for its first 160 observations, so its span starts after them. Asset 2's halt
sits in the middle of its column, so its span covers the whole window, and the halt is a gap inside
the span, a held gap.
=#

pr = price_ingestion(PriceIngestion(), Xmiss)
span = Matrix(pr.span)
pretty_table(DataFrame(; asset = nx[[3, 2, 4]],
                       listed_from = [findfirst(view(span, :, j)) for j in (3, 2, 4)],
                       listed_to = [findlast(view(span, :, j)) for j in (3, 2, 4)],
                       observations = fill(T, 3));
             title = "First and last listed observation of assets 3, 2 and 4")

#=
## 4. Carry the gaps into the returns

The conversion computes the returns and changes nothing else. Every asset keeps its column, every
date keeps its row, and each return computed from a missing price is `NaN`. A run of `k` missing
prices inside a column makes `k + 1` returns non-finite, and a run at either end of the column makes
`k`. No other return changes, because the return of one asset uses only the prices of that asset.

The [`AssetPanel`](@ref) in the result states the universe. `amsk` is the listing span on the dates
of the returns, and `emsk` marks the cells of `amsk` whose return is also finite.
=#

rd = prices_to_returns(PricesToReturns(), pr)
pretty_table(DataFrame(;
                       quantity = ["assets", "observations", "non-finite return cells",
                                   "estimable cells"],
                       value = [length(rd.nx), size(rd.X, 1), count(!isfinite, rd.X),
                                count(Matrix(rd.pnl.emsk))]);
             title = "Counts on the returns after the conversion")

#=
## 5. State a price convention across the halts

Every later step can work with the gaps left in. An optimiser uses only the assets whose return
is finite on every observation of a window, the coverage universe of that window, and gives the
other assets no weight. If you want to state what the price was during a halt, use
[`PriceGapFill`](@ref). It is a universe policy, fitted on a training window and applied to later
windows. It fills held gaps only, so it never writes a price before an asset listed or after it
delisted.

[`CarriedPrice`](@ref), the default, carries the last observed price forward, the held price. A
halt then becomes a flat stretch of zero returns, and the whole move of the price falls on the
observation that ends the halt. A reduction over each asset, such as `MedianValue()`, fills the gap
with one constant instead, which makes two moves that the market never printed, one into the gap
and one out of it. Both leave the price at the end of the gap unchanged. They differ in which
observations carry the move.
=#

fill_res = fit_preprocessing(PriceGapFill(), pr)
pr_filled = apply_preprocessing(fill_res, pr)
rd_filled = prices_to_returns(PricesToReturns(), pr_filled)

pretty_table(DataFrame(; table = ["carried", "filled with the held price"],
                       non_finite = [count(!isfinite, rd.X), count(!isfinite, rd_filled.X)],
                       asset_3_still_unlisted = [!isfinite(rd.X[1, 3]),
                                                 !isfinite(rd_filled.X[1, 3])]);
             title = "Non-finite returns before and after the fill")

#=
The table counts the non-finite returns before and after the fill. The last column shows, for
each table, whether the first return of asset 3 is non-finite. Asset 3 was not yet listed on that
date.

## 6. Drop what has too few prices

Asset 3 is missing for about 63% of the window. Keeping it costs nothing, because an optimiser
leaves it out of every window whose coverage universe excludes it. If you want it gone from the
universe, use [`MissingDataFilter`](@ref), the only step that deletes an asset or an observation.
It is a universe policy too. We fit it on the table, and the fit records the assets that pass the
threshold. Applied to a later window, the filter keeps those assets and does not choose
again from that window's own data.
=#

mdf = fit_preprocessing(MissingDataFilter(; col_thr = 0.5), pr)
println("Assets kept after the 50% column filter: $(length(mdf.nx)) of $N")

#=
## 7. Optimise on the result

A [`ReturnsResult`](@ref) that holds a panel goes into the rest of the package like any other,
with no extra arguments. The optimiser finds the coverage universe of the window from the returns
and the panel.

We optimise on both tables to show what the fill changes. The scattered single-day gaps put most
of the assets of the carried table outside the coverage universe. The fill closes the
held gaps and brings those assets back. Asset 3 stays out, because it was not listed at the start
of the window, and the listing span tells a missing listing apart from a halt.
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
pretty_table(DataFrame(; table = ["carried", "filled with the held price"],
                       estimable = [count(res.imsk), count(res_filled.imsk)],
                       active_names = [count(>(1e-6), res.w), count(>(1e-6), res_filled.w)]);
             title = "Estimable assets and assets with a weight above 1e-6")

#=
## 8. Plot the gaps

A heatmap of the missing prices shows the three kinds of gap. The wide band is asset 3's late
listing, the short block is asset 2's halt, and the scattered dots are the single-day gaps.
=#

using StatsPlots, GraphRecipes
heatmap(1:N, 1:T, Float64.(ismissing.(vals)); xlabel = "Asset", ylabel = "Day",
        colorbar_title = "missing", title = "Missing prices by asset and day", yflip = true)

#=
## Summary

  - [`price_ingestion`](@ref) treats `missing` and `NaN` as one absent price, puts any factor and
    benchmark series on the dates of the price table by default, collapses the series only under
    a non-empty `collapse_args`, and computes each asset's listing span from the whole table.
  - [`prices_to_returns`](@ref) computes the returns and keeps every gap. The
    [`AssetPanel`](@ref) it returns states which assets the library can estimate on which dates.
  - [`PriceGapFill`](@ref) fills a held gap with a price convention and writes nothing outside the
    listing span. [`MissingDataFilter`](@ref) deletes the assets with too few prices, and no other
    step deletes anything. Both are fitted on a training window and applied to later windows, so a
    later window never chooses its own universe.

[The point-in-time universe](../../user_guide/08_Point_in_Time_Universe.md) takes a table with gaps
through these steps to a walk-forward, and shows where a fitted fill goes in a [`Pipeline`](@ref)
over the prices.
=#

#=
!!! note "Why these keywords moved"
    `nan_to_missing`, `impute_method`, `missing_col_percent` and `missing_row_percent` used to be
    keywords of `prices_to_returns`, and by default it deleted every row that held a gap. All four
    are gone. A keyword stays on the conversion only if it changes how a return is computed.
    [`PriceGapFill`](@ref) now fills, and [`MissingDataFilter`](@ref) deletes.

    Three keywords stayed, `ret_method`, `padding` and `gap_return_alg`. The other inputs the
    conversion used to take are now fields of the [`PricesResult`](@ref) it takes. `join_method`
    and `collapse_args` change the dates of the observations, so they belong to
    [`PriceIngestion`](@ref). `prices_to_returns(X)` on a bare price table returns the same result
    as `prices_to_returns(price_ingestion(PriceIngestion(), X))`. To use a different join, a
    collapse, a listing span of your own, or factor and benchmark series, call the two functions
    yourself.
=#
