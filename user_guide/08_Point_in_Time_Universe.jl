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

This page shows both halves — what handles a gap, and what refuses one — and finishes with a
walk-forward over a universe that changes inside the window.
=#

using PortfolioOptimisers, CSV, TimeSeries, Clarabel, Statistics, LinearAlgebra

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd_full = prices_to_returns(X)

#=
## 1. A gap is a `NaN`, and it says which observations

A missing return is written as `NaN` in the returns matrix. That is the whole convention: there
is no sentinel value, no separate mask you have to keep in step by hand, and no imputation.

Below, one asset lists 60 observations into the sample and another delists 40 observations
before the end. Everything else is untouched.
=#

nx = rd_full.nx
Xg = copy(rd_full.X)
T, N = size(Xg)
late, dead = 3, 7                                    # the young asset, and the one that leaves
Xg[1:60, late] .= NaN                                # lists at observation 61
Xg[(T - 39):T, dead] .= NaN                          # delists 40 observations before the end
rd = ReturnsResult(; nx = nx, X = Xg, ts = rd_full.ts)

(nx[late], nx[dead], count(!isfinite, Xg))

#=
## 2. The prior answers, and the mask is derived from its answer

A Prior Estimator fits on the assets it can estimate and returns a result on the **full** asset
universe. An asset it could not estimate carries `NaN` in `mu` and on the diagonal of `sigma`.

The Investable Mask is never stored as a field. It is *derived* from the result, which is what
keeps it from going stale: there is one definition, and every consumer applies it.
=#

pr = prior(EmpiricalPrior(), rd)
imsk = isfinite.(pr.mu) .& isfinite.(diag(pr.sigma))

(nx[.!imsk], pr.mu[late], pr.mu[dead])

#=
Both the young asset and the delisted one are outside the mask over this window: the Coverage
Universe asks for a finite return at *every* observation of the window, so an asset that is
absent for part of it is not estimable over the whole of it.

## 3. An optimiser reduces once, and expands back

Every optimiser family reduces to the investable assets at its entry — after the prior fit and
before the clustering, the sampling or the solve — then expands the solved weights back into a
vector of the full length. The weight of an asset that was not investable is **exactly** zero,
not a small number the solver happened to land on.
=#

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res = optimise(MeanRisk(; opt = JuMPOptimiser(; slv = slv)), rd)

(length(res.w), res.w[late], res.w[dead], sum(res.w))

#=
The same holds for the hierarchical families, which reduce before they cluster, so a gap never
reaches a distance matrix:
=#

res_hrp = optimise(HierarchicalRiskParity(), rd)

(res_hrp.w[late], res_hrp.w[dead], sum(res_hrp.w))

#=
## 4. What refuses, and what the refusal says

A **plain moment estimator** has no correct answer for a gapped sample. It is handed a matrix
and nothing else — no mask, no panel, no way to tell a holiday from a delisting — so it refuses
rather than guessing. This is the second half of the rule, and it is what the error looks like:
=#

try
    cov(PortfolioOptimisersCovariance(), Xg)
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

Xdead = copy(rd_full.X)
Xdead[1, :] .= NaN                                   # every asset misses the first observation

try
    optimise(EqualWeighted(), ReturnsResult(; nx = nx, X = Xdead))
catch err
    showerror(stdout, err)
end

#=
## 5. An `AssetPanel` states a universe the prices cannot

A price series can be perfectly finite while the asset is still untradeable — a suspension, a
holding period after a corporate action, a name you have excluded from the mandate. The returns
alone cannot express that. An [`AssetPanel`](@ref) carries an **active mask** that says, per
observation and per asset, whether the asset was in the universe, and the Coverage Universe
reads it alongside the returns.
=#

amsk = trues(T, N)
amsk[100:130, 5] .= false                            # suspended, with a finite price throughout
rd_pnl = ReturnsResult(; nx = nx, X = rd_full.X, ts = rd_full.ts,
                       pnl = AssetPanel(;
                                        pf = [NumericPanelField(; name = "mcap",
                                                                vals = ones(T, N))],
                                        amsk = amsk, emsk = amsk))

res_pnl = optimise(EqualWeighted(), rd_pnl)

(all(isfinite, rd_full.X), res_pnl.w[5], sum(res_pnl.w))

#=
Every return is finite, and asset 5 still holds zero: the panel's mask excluded it on its own.

## 6. A walk-forward over a universe that changes

This is the case the whole chain exists for. Each fold derives the Coverage Universe of **its
own** training window, so the universe is allowed to differ from fold to fold, and each fold's
weights come back on the caller's full universe with zeros where that fold could not trade.
=#

cv = IndexWalkForward(120, 40)
mpr = cross_val_predict(HierarchicalRiskParity(), rd, cv)

[(count(isnothing(p.res.imsk) ? trues(N) : p.res.imsk), sum(p.res.w)) for p in mpr.pred]

#=
A fold whose window is entirely inside the young asset's listed life counts it; a fold that
straddles the listing does not. `imsk` is `nothing` when every asset was investable, which is
the all-investable path taking no reduction at all rather than a vector of `true`.

Stitching the folds gives an out-of-sample series that carries a number at every observation,
even though the universe moved underneath it. A weight held in an asset that stops quoting
inside a *test* window is zeroed once, and the missing weight sits in cash — it is not
renormalised away, because that would silently re-lever the book.
=#

(length(mpr.mrd.X), all(isfinite, mpr.mrd.X),
 expected_risk(LowOrderMoment(; alg = SecondMoment()), mpr))

#=
## Where to go next

  - [Data and priors](01_Data_and_Priors.md) — the prior that does the reducing here.
  - [Optimisers](02_Optimisers.md) — the families that reduce and expand.
  - [Validation and tuning](05_Validation_and_Tuning.md) — the cross-validation this page ran.
=#
