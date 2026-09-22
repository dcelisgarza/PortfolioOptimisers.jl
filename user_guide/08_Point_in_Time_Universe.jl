#=
```@meta
Description = "Assets that list, delist or go quiet mid-sample: the point-in-time universe, from a gapped price table to a walk-forward."
```

# The point-in-time universe

Real asset universes move. A company lists halfway through your sample, another is acquired and
stops quoting, a third is suspended for a month. `PortfolioOptimisers.jl` answers all three cases
with one rule. A layer either handles the missing asset, or it throws an error that names it.

A layer that can state a correct answer for a missing asset does so, and its docstring says how.
A layer that cannot throws, instead of returning a number that is wrong. Nothing in the library
drops an asset without telling you, back-fills a price, or reads a gap as a zero return.

You ingest a gapped table with the call a clean table takes. [`prices_to_returns`](@ref) on a raw
price table runs the ingestion layer. The layer reads the first and the last quote of each asset,
its listing span, carries every gap into the returns, and hands back a [`ReturnsResult`](@ref)
whose [`AssetPanel`](@ref) states the universe. You never build a returns matrix or a mask by
hand.

This page shows what handles a gap and what refuses one. It ends with a walk-forward over a
universe that changes inside the window.
=#

using PortfolioOptimisers, CSV, TimeSeries, Clarabel, Statistics, LinearAlgebra

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]

#=
## 1. A gapped price table, and the one call that ingests it

We gap the prices below. One asset lists 60 observations into the sample, another delists 40
observations before the end, and a third is suspended for a month in the middle. Everything else
is untouched. The gaps sit in the prices, which is how a data vendor sends them. A gap is `NaN` or
`missing`, and the layer reads both as one absence.
=#

nx = string.(colnames(X))
P = Matrix{Float64}(values(X))
T, N = size(P)
late, dead, halt = 3, 7, 5                           # lists late, delists early, suspended
P[1:60, late] .= NaN                                 # not yet listed for 60 observations
P[(T - 39):T, dead] .= NaN                           # delists 40 observations before the end
P[100:130, halt] .= NaN                              # suspended, then prices again
Xg = TimeArray(timestamp(X), P, colnames(X))

rd = prices_to_returns(Xg)

(size(rd.X), count(!isfinite, rd.X))

#=
The conversion deleted nothing. Every asset keeps its column, every date but the first keeps its
row, and a missing return is a `NaN` in the returns matrix. There is no sentinel value and no
imputation. A return reads two consecutive prices, so a run of `k` gapped prices leaves the
`k + 1` returns that read one of them non-finite. The gap never spreads to a neighbouring asset.

## 2. The panel states the universe

The layer also builds the [`AssetPanel`](@ref) in `rd.pnl`. Its active mask `amsk` says when each
asset is in the universe, and the layer reads it off the gaps of each price column. A leading run
of gaps is an asset not yet listed, a trailing run is a delisting, and an interior run is a
suspension on an asset that is still listed. Its estimation mask `emsk` is the active mask and
finiteness together. The two masks differ on a suspension, where the asset is in the universe and
has no return.
=#

amsk = Matrix(rd.pnl.amsk)
emsk = Matrix(rd.pnl.emsk)
[(nx[j], findfirst(amsk[:, j]), findlast(amsk[:, j]), count(emsk[:, j]))
 for j in (late, dead, halt)]

#=
The late lister is active from its first return, one observation after its first price. The
delisted asset is active up to its last. The suspended asset is active throughout, and it is
estimable everywhere but inside the halt. The layer reads the span once over the whole table, so
the span describes the instrument and does not change with the window. Take a walk-forward fold
whose window ends inside the delisting, as one below does. That fold reads the delisted asset as
one you still hold, and not as one that was never listed.

`prices_to_returns(Xg)` is the short spelling of two steps. [`price_ingestion`](@ref) builds the
ingested price table. It unifies the two gap spellings, joins factor and benchmark series onto the
asset clock, collapses to a lower frequency if you ask for one, and reads the span.
[`PricesToReturns`](@ref) converts that table to returns. Write the two steps when you need any of
those options. The [data preprocessing example](../examples/1_foundations/02_Data_Preprocessing.md)
walks each one.

## 3. The prior answers, and the mask follows from its answer

A prior estimator fits on the assets it can estimate, and it returns a result over the full asset
universe. An asset it could not estimate carries `NaN` in `mu` and on the diagonal of `sigma`.

The investable mask, the assets a result can weight, is not stored in a field. Every consumer
reads it off the result with the one test below, so there is one definition and it cannot go
stale.
=#

pr = prior(EmpiricalPrior(), rd)
imsk = isfinite.(pr.mu) .& isfinite.(diag(pr.sigma))

(nx[.!imsk], pr.mu[late], pr.mu[dead], pr.mu[halt])

#=
All three assets are outside the mask over this window. The assets a window can estimate, its
coverage universe, are the ones with a finite return and an active mask at every observation of
the window. An asset that is absent for any part of the window is not estimable over the whole of
it. The suspension, and not the listing, is what keeps the third asset out. Over a window that
misses the halt it is back in.

## 4. An optimiser reduces once, and expands back

Every optimiser family reduces to the investable assets at its entry, after the prior fit and
before the clustering, the sampling or the solve. It then expands the solved weights back into a
vector of the full length. The weight of an asset that was not investable is zero, and not a small
number the solver landed on.
=#

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res = optimise(MeanRisk(; opt = JuMPOptimiser(; slv = slv)), rd)

(length(res.w), res.w[late], res.w[dead], res.w[halt], sum(res.w))

#=
The hierarchical families do the same. They reduce before they cluster, so a gap never reaches a
distance matrix.
=#

res_hrp = optimise(HierarchicalRiskParity(), rd)

(res_hrp.w[late], res_hrp.w[dead], res_hrp.w[halt], sum(res_hrp.w))

#=
## 5. What refuses, and what the refusal says

A plain moment estimator has no correct answer for a gapped sample. It reads a matrix and nothing
else. It has no mask, no panel, and no way to tell a holiday from a delisting, so it throws
instead of guessing. The cell below prints the error.
=#

try
    cov(PortfolioOptimisersCovariance(), rd.X)
catch err
    showerror(stdout, err)
end

#=
The message names the count of non-finite entries, the first of them, and the two ways forward.
You fit through a prior, which reduces and expands, or you use a mask-aware estimator, which reads
the active mask of an `AssetPanel`.

The second error is the empty universe. A window that leaves no asset investable has nothing to
weight, so the library throws instead of returning an empty portfolio.
=#

Xdead = copy(rd.X)
Xdead[1, :] .= NaN                                   # every asset misses the first observation

try
    optimise(EqualWeighted(), ReturnsResult(; nx = nx, X = Xdead))
catch err
    showerror(stdout, err)
end

#=
## 6. A declared listing calendar states what the prices cannot

A price series can be finite while the asset is untradeable. A suspension can carry stale quotes,
a corporate action can start a holding period, and your mandate can exclude a name. The prices
alone say none of that. If you hold a listing calendar, pass it to [`PriceIngestion`](@ref) as
`span`, sized `price observations × assets`. It replaces what the layer read off the gaps, and the
layer never overrides a calendar you gave it.
=#

calendar = trues(T, N)
calendar[100:130, halt] .= false                     # out of the universe, with a finite price

pr_cal = price_ingestion(PriceIngestion(; span = calendar), X)
rd_cal = prices_to_returns(pr_cal)

res_cal = optimise(EqualWeighted(), rd_cal)

(all(isfinite, rd_cal.X), count(Matrix(rd_cal.pnl.amsk)[:, halt]), res_cal.w[halt],
 sum(res_cal.w))

#=
Every return is finite, and the asset still holds zero. The calendar alone excluded it. If you
hold a calendar and gapped prices, you get both answers. The calendar states which asset is listed
when, and the conversion still carries the gaps into `emsk`.

## 7. A walk-forward over a universe that changes

Each fold reads the coverage universe of its own training window, so the universe can differ from
fold to fold. Each fold's weights come back over your full universe, with a zero where that fold
could not trade.
=#

cv = IndexWalkForward(120, 40)
mpr = cross_val_predict(HierarchicalRiskParity(), rd, cv)

[(count(isnothing(p.res.imsk) ? trues(N) : p.res.imsk), sum(p.res.w)) for p in mpr.pred]

#=
The training windows of the first two folds straddle the late listing and the halt, so both assets
are out of those folds. The third window sits inside the listed life of the young asset and after
the halt, so only the suspension keeps an asset out there. `imsk` is `nothing` when every asset
was investable, because a fold that reduces nothing stores nothing, rather than a vector of
`true`.

The delisting never falls in a training window. It falls in the test window of the third fold,
where the fold already holds the asset. The score zeroes that weight once, on the observation
where the asset stops quoting, and the weight it held sits in cash from there on. The score leaves
the other weights alone, because renormalising them would raise your exposure. Pass
`strict = true` to throw there instead. The stitched folds give an out-of-sample series with a
number at every observation, over a universe that moved.
=#

(length(mpr.mrd.X), all(isfinite, mpr.mrd.X),
 expected_risk(LowOrderMoment(; alg = SecondMoment()), mpr))

#=
A fill and a filter change the universe itself. [`PriceGapFill`](@ref) states a price across the
halt, and [`MissingDataFilter`](@ref) drops an asset too sparse to trust. Each one is fitted on
the training window of a fold and replayed by name on the test window, or the fold would be scored
against a universe the future chose. Each one therefore runs inside a [`Pipeline`](@ref) over the
ingested price table, and the conversion to returns is the step after it. The folds are cut on the
price clock, so we build the ingested price table once, outside the pipeline, and the pipeline
starts from it.
=#

pr_g = price_ingestion(PriceIngestion(), Xg)
pipe = Pipeline(;
                steps = (PriceGapFill(), PricesToReturns(), EmpiricalPrior(),
                         HierarchicalRiskParity()))
mpr_pipe = cross_val_predict(pipe, pr_g, cv)

[(count(isnothing(p.res.imsk) ? trues(N) : p.res.imsk), sum(p.res.w))
 for p in mpr_pipe.pred]

#=
The fill carries the last quote forward across the halt, and it stays inside the listing span, so
it never states a price before the late listing or after the delisting. The suspended asset is
back in every fold, and the third fold now reduces nothing. The late lister is still out of the
two windows that straddle its listing, and the third fold still zeroes the delisted asset in its
test window. Both cases follow from the listing span, which a fill does not change.

## Where to go next

  - [Data preprocessing and the ingestion layer](../examples/1_foundations/02_Data_Preprocessing.md)
    covers each step of the layer in turn: unify, join, span, carry, fill, drop.
  - [Data and priors](01_Data_and_Priors.md) covers the prior that reduces here.
  - [Optimisers](02_Optimisers.md) covers the families that reduce and expand.
  - [Validation and tuning](05_Validation_and_Tuning.md) covers the cross-validation this page ran.
=#
