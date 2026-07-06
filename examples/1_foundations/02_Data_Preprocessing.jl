#=
# Data preprocessing and imputation

Real price data is rarely clean. Assets list partway through the window (leading gaps), trade
halts and stale quotes leave flat or missing stretches, names get delisted (trailing gaps), and
exchanges keep different holiday calendars so timestamps do not line up. Feeding that straight
into an optimiser is a recipe for silent errors. [`prices_to_returns`](@ref) handles all of it in
one call, *before* it computes returns, through three cooperating controls:

  - `missing_col_percent` — drop any **asset** whose fraction of missing observations exceeds
    this threshold (too sparse to trust).
  - `missing_row_percent` — drop any **timestamp** whose fraction of missing values across assets
    exceeds this threshold (e.g. a misaligned holiday).
  - `impute_method` — fill the *remaining* gaps with an [`Impute`](https://github.com/invenia/Impute.jl)
    imputor before differencing prices into returns.

The order matters: filter the hopeless rows/columns first, then impute what is left, then compute
returns.

!!! tip "When to reach for this"
    Reach for these whenever the raw price table has missing values — newly listed or delisted
    assets, halted or stale prices, or non-overlapping trading calendars. The decision for each
    asset or date is *drop or fill*: drop what is too sparse to trust (`missing_*_percent`), fill
    what is recoverable (`impute_method`), and let `prices_to_returns` do both in one pass.

!!! note "Impute.jl imputors"
    Imputation methods are [`Impute.jl`](https://github.com/invenia/Impute.jl) imputors passed via
    `impute_method`, so bring in `using Impute`. The two most useful for prices are
    `Impute.LOCF()` (last observation carried forward — the right model for a halt or stale quote,
    since it holds the last traded price) and `Impute.Interpolate()` (linear interpolation —
    natural for short gaps between two good prices).
=#

using PortfolioOptimisers, PrettyTables, DataFrames, Statistics, Impute

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
asset that is mostly missing (a late lister we will want to drop), a halted block of one asset,
and scattered single-day gaps across the rest.
=#

using CSV, TimeSeries

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
ts = timestamp(X)
nx = string.(colnames(X))
T, N = size(values(X))

vals = Matrix{Union{Float64, Missing}}(values(X))
vals[1:160, 3] .= missing                 # asset 3: mostly missing (late lister → drop)
vals[100:120, 2] .= missing               # asset 2: a ~3-week trading halt (fillable)
using StableRNGs
rng = StableRNG(42)
for _ in 1:60                              # scattered single-day gaps elsewhere
    vals[rand(rng, 1:T), rand(rng, 4:N)] = missing
end
Xmiss = TimeArray(ts, vals, Symbol.(nx))

#=
## 2. Diagnose the missingness

Before deciding what to drop or fill, measure it. The per-column and per-row missing fractions
are exactly the quantities `missing_col_percent` and `missing_row_percent` threshold against.
=#

col_missing = vec(mean(ismissing, vals; dims = 1))
worst_cols = sort(DataFrame(; asset = nx, missing_frac = col_missing), :missing_frac;
                  rev = true)
pretty_table(first(worst_cols, 6); formatters = [resfmt],
             title = "Missing fraction per asset (worst six)")

#=
Asset 3 is missing ~64% of the time — there is no honest way to impute that, so it should be
dropped. Asset 2's halt and the scattered single-day gaps, by contrast, are short and
recoverable.

## 3. Drop the hopeless, keep the rest

`missing_col_percent = 0.5` drops any asset more than half missing (asset 3), while
`missing_row_percent = 0.5` would drop any date more than half missing across assets. With no
imputation yet, the kept assets still contain gaps, so we ask only how many assets survive the
filter.
=#

rd_droponly = prices_to_returns(Xmiss; missing_col_percent = 0.5, missing_row_percent = 0.5,
                                impute_method = Impute.LOCF())
println("Assets kept after the 50% column filter: $(length(rd_droponly.nx)) of $N")

#=
## 4. Impute the survivors

The kept assets still have a halt and scattered gaps. We fill them with two different imputors
and compare. `Impute.LOCF()` carries the last observed *price* forward — so a halt becomes a flat
price stretch and therefore zero returns through the halt, which is the honest accounting for a
non-trading period. `Impute.Interpolate()` draws a straight line across the gap, spreading a
small constant return over the missing days.
=#

rd_locf = prices_to_returns(Xmiss; missing_col_percent = 0.5, impute_method = Impute.LOCF())
rd_interp = prices_to_returns(Xmiss; missing_col_percent = 0.5,
                              impute_method = Impute.Interpolate())

pretty_table(DataFrame(; method = ["LOCF", "Interpolate"],
                       assets = [length(rd_locf.nx), length(rd_interp.nx)],
                       observations = [size(rd_locf.X, 1), size(rd_interp.X, 1)]);
             title = "Both imputors yield a complete returns matrix")

#=
## 5. Straight into the pipeline

A cleaned [`ReturnsResult`](@ref) is just an ordinary one — it flows into the rest of the package
with no special handling. We solve a minimum-variance [`MeanRisk`](@ref) on the LOCF-imputed data
to confirm the preprocessing produced something usable end to end.
=#

using Clarabel
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
res = optimise(MeanRisk(; obj = MinimumRisk(),
                        opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), rd_locf),
                                            slv = slv)))
println("Min-variance solve on cleaned data: $(res.retcode), $(count(>(1e-6), res.w)) active names")

#=
## 6. Visualising the damage

A heatmap of the missingness mask makes the structure obvious: the wide band is asset 3 (dropped
by the column filter), the short block is asset 2's halt, and the speckle is the scattered gaps
that imputation fills.
=#

using StatsPlots, GraphRecipes
heatmap(1:N, 1:T, Float64.(ismissing.(vals)); xlabel = "Asset", ylabel = "Day",
        colorbar_title = "missing", title = "Missingness pattern (raw data)", yflip = true)

#=
## Summary

`prices_to_returns` is the single entry point for cleaning price data:

  - `missing_col_percent` / `missing_row_percent` drop assets and dates that are too sparse to
    trust, before any returns are computed.
  - `impute_method` fills the recoverable gaps with an `Impute.jl` imputor — `Impute.LOCF()` for
    halts and stale quotes, `Impute.Interpolate()` for short gaps.
  - The result is an ordinary [`ReturnsResult`](@ref) that feeds the rest of the pipeline
    unchanged.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New page; closes the data-preprocessing backlog item for 1_foundations.
#src - TODO-on-run: confirm Impute.LOCF()/Impute.Interpolate() are the correct 0.6 imputor names
#src   and that prices_to_returns(; impute_method=...) fills the halt + scattered gaps so the
#src   returns matrix is complete and the downstream MeanRisk solves. Check whether the leading
#src   missings on a late-lister need NOCB/row-drop rather than LOCF (LOCF cannot fill a leading
#src   gap), and record the behaviour. Impute added to docs/Project.toml for this page.
