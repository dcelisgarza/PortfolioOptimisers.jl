#=
# The point-in-time universe

Real asset universes move. A company lists halfway through your sample, another is acquired and
stops quoting, a third is suspended for a month. `PortfolioOptimisers.jl` handles this
end to end, and the rule it follows is one line:

> **Handle it, or throw a named error.**

A layer that can state a correct answer for a missing asset does so, and says how in its
docstring. A layer that cannot refuses loudly, by name, rather than returning a number that is
quietly wrong. There is no third option: nothing in the library silently drops an asset, back-fills
a price, or treats a gap as a zero return.

The way in is the same call a clean table takes. [`prices_to_returns`](@ref) on a raw price table
runs the **ingestion layer**: it reads each asset's **Listing Span** off the gaps, carries every
gap into the returns, and hands back a [`ReturnsResult`](@ref) whose [`AssetPanel`](@ref) states
the universe. You never build a returns matrix or a mask by hand.

This page shows both halves — what handles a gap, and what refuses one — and finishes with a
walk-forward over a universe that changes inside the window.
=#

using PortfolioOptimisers, CSV, TimeSeries, Clarabel, Statistics, LinearAlgebra

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]

#=
## 1. A gapped price table, and the one call that ingests it

Below, one asset lists 60 observations into the sample, another delists 40 observations before
the end, and a third is suspended for a month in the middle. Everything else is untouched. The
gaps are punched into the **prices**, which is how they arrive from a data vendor — `NaN` or
`missing`, the layer reads both as one absence.
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
The conversion deleted nothing: every asset keeps its column, every date but the first keeps
its row, and a missing return is a `NaN` in the returns matrix. That is the whole convention at
the returns level — there is no sentinel value and no imputation. A return reads two consecutive
prices, so a run of `k` gapped prices leaves the `k + 1` returns that read one of them
non-finite, and the gap never spreads to a neighbouring asset.

## 2. The panel states the universe

What the layer adds is the [`AssetPanel`](@ref) in `rd.pnl`. Its **active mask** `amsk` is the
**Span Rule** read off each price column: a leading run of gaps is an asset not yet listed, a
trailing run is a delisting, and an interior run is a suspension on an asset that is still
listed. Its **estimation mask** `emsk` is the active mask intersected with finiteness. The two
differ exactly on a suspension: the asset is in the universe, and it has no return.
=#

amsk = Matrix(rd.pnl.amsk)
emsk = Matrix(rd.pnl.emsk)
[(nx[j], findfirst(amsk[:, j]), findlast(amsk[:, j]), count(emsk[:, j]))
 for j in (late, dead, halt)]

#=
The late lister is active from its first *return*, one observation after its first price; the
delisted asset is active up to its last; the suspended asset is active throughout and estimable
everywhere but inside the halt. Reading the span **once over the whole table** is what makes it
a fact about the instruments rather than a judgement a window made: a walk-forward fold below
sees a delisting that straddles its window end as an asset that is still held, not one that was
never listed.

`prices_to_returns(Xg)` is the friendliest spelling of two steps. [`price_ingestion`](@ref)
assembles the price carrier — unifying the gap spellings, joining factor and benchmark series
onto the asset clock, collapsing to a lower frequency if asked, and reading the span — and
[`PricesToReturns`](@ref) converts it. Write the two steps when you need any of those: the
[data preprocessing example](../examples/1_foundations/02_Data_Preprocessing.md) walks each one.

## 3. The prior answers, and the mask is derived from its answer

A Prior Estimator fits on the assets it can estimate and returns a result on the **full** asset
universe. An asset it could not estimate carries `NaN` in `mu` and on the diagonal of `sigma`.

The Investable Mask is never stored as a field. It is *derived* from the result, which is what
keeps it from going stale: there is one definition, and every consumer applies it.
=#

pr = prior(EmpiricalPrior(), rd)
imsk = isfinite.(pr.mu) .& isfinite.(diag(pr.sigma))

(nx[.!imsk], pr.mu[late], pr.mu[dead], pr.mu[halt])

#=
All three are outside the mask over this window: the Coverage Universe asks for a finite return
*and* an active mask at every observation of the window, so an asset that is absent for any part
of it is not estimable over the whole of it. The suspended asset is excluded by its Held Gap, not
by its listing — over a window that misses the halt, it is back in.

## 4. An optimiser reduces once, and expands back

Every optimiser family reduces to the investable assets at its entry — after the prior fit and
before the clustering, the sampling or the solve — then expands the solved weights back into a
vector of the full length. The weight of an asset that was not investable is **exactly** zero,
not a small number the solver happened to land on.
=#

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res = optimise(MeanRisk(; opt = JuMPOptimiser(; slv = slv)), rd)

(length(res.w), res.w[late], res.w[dead], res.w[halt], sum(res.w))

#=
The same holds for the hierarchical families, which reduce before they cluster, so a gap never
reaches a distance matrix:
=#

res_hrp = optimise(HierarchicalRiskParity(), rd)

(res_hrp.w[late], res_hrp.w[dead], res_hrp.w[halt], sum(res_hrp.w))

#=
## 5. What refuses, and what the refusal says

A **plain moment estimator** has no correct answer for a gapped sample. It is handed a matrix
and nothing else — no mask, no panel, no way to tell a holiday from a delisting — so it refuses
rather than guessing. This is the second half of the rule, and it is what the error looks like:
=#

try
    cov(PortfolioOptimisersCovariance(), rd.X)
catch err
    showerror(stdout, err)
end

#=
The message names the count, the first offending entry, and the two ways forward: fit through a
prior, which reduces and expands, or use a mask-aware estimator, which reads the active mask of
an Asset Panel.

The other named refusal is the empty universe. If a window leaves no asset investable at all
there is nothing to weight, and the library says so rather than returning an empty portfolio:
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

A price series can be perfectly finite while the asset is still untradeable — a suspension with
stale quotes, a holding period after a corporate action, a name you have excluded from the
mandate. The prices alone cannot express that. A caller holding a listing calendar hands it to
[`PriceIngestion`](@ref) as `span`, `price observations × assets`, and it replaces the Span
Rule's answer outright: the layer never second-guesses a calendar it was given.
=#

calendar = trues(T, N)
calendar[100:130, halt] .= false                     # out of the universe, with a finite price

pr_cal = price_ingestion(PriceIngestion(; span = calendar), X)
rd_cal = prices_to_returns(pr_cal)

res_cal = optimise(EqualWeighted(), rd_cal)

(all(isfinite, rd_cal.X), count(Matrix(rd_cal.pnl.amsk)[:, halt]), res_cal.w[halt],
 sum(res_cal.w))

#=
Every return is finite, and the asset still holds zero: the declared calendar excluded it on its
own. A caller who holds the calendar *and* the gaps gets both — the calendar states who is
listed when, and the conversion still carries the gaps into `emsk`.

## 7. A walk-forward over a universe that changes

This is the case the whole chain exists for. Each fold derives the Coverage Universe of **its
own** training window, so the universe is allowed to differ from fold to fold, and each fold's
weights come back on the caller's full universe with zeros where that fold could not trade.
=#

cv = IndexWalkForward(120, 40)
mpr = cross_val_predict(HierarchicalRiskParity(), rd, cv)

[(count(isnothing(p.res.imsk) ? trues(N) : p.res.imsk), sum(p.res.w)) for p in mpr.pred]

#=
The first two folds' training windows straddle the late listing and the halt, so both are out;
the third window sits inside the young asset's listed life and after the halt, so only the
suspended asset's Held Gap keeps it out there. `imsk` is `nothing` when every asset was
investable, which is the all-investable path taking no reduction at all rather than a vector of
`true`.

The delisting is never in a training window — it falls inside the third fold's *test* window.
The library names that as a **Held Gap** when it scores the fold: the weight held in an asset
that stops quoting is zeroed once, and the missing weight sits in cash. It is not renormalised
away, because that would silently re-lever the book. Pass `strict = true` to refuse instead.
Stitching the folds gives an out-of-sample series that carries a number at every observation,
even though the universe moved underneath it.
=#

(length(mpr.mrd.X), all(isfinite, mpr.mrd.X),
 expected_risk(LowOrderMoment(; alg = SecondMoment()), mpr))

#=
A fill or a filter — [`PriceGapFill`](@ref) to state a price across the halt,
[`MissingDataFilter`](@ref) to drop an asset too sparse to trust — is a **Universe Policy**: it
is fitted on each fold's training window and replayed by name, or a fold would be scored against
a universe the future chose. So it runs inside a [`Pipeline`](@ref) over the **price carrier**,
and the conversion is the step after it. Folds are then cut on the price clock, which is why the
carrier is built once, outside the pipeline, and the pipeline starts from it:
=#

pr_g = price_ingestion(PriceIngestion(), Xg)
pipe = Pipeline(;
                steps = (PriceGapFill(), PricesToReturns(), EmpiricalPrior(),
                         HierarchicalRiskParity()))
mpr_pipe = cross_val_predict(pipe, pr_g, cv)

[(count(isnothing(p.res.imsk) ? trues(N) : p.res.imsk), sum(p.res.w))
 for p in mpr_pipe.pred]

#=
The fill states a **Held Price** across the halt — the last quote, carried forward, bounded by
the Listing Span so it never invents a price before the late listing or after the delisting —
and the suspended asset is back in every fold; the third now reduces nothing at all. The late
lister is still out of the two windows that straddle its listing, and the delisting is still a
Held Gap in the third fold's test window: the span put them there, and a fill cannot move them.

## Where to go next

  - [Data preprocessing and the ingestion layer](../examples/1_foundations/02_Data_Preprocessing.md)
    — each piece of the layer in turn: unify, join, span, carry, fill, drop.
  - [Data and priors](01_Data_and_Priors.md) — the prior that does the reducing here.
  - [Optimisers](02_Optimisers.md) — the families that reduce and expand.
  - [Validation and tuning](05_Validation_and_Tuning.md) — the cross-validation this page ran.
=#
