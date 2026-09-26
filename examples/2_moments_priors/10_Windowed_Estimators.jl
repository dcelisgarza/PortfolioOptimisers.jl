#=
```@meta
Description = "Windowed moment estimators in PortfolioOptimisers.jl: compute a moment from a chosen window of observations, or weight the recent ones more."
```

# [Windowed moment estimators](@id example-windowed-moment-estimators)

The estimators of the [expected returns](@ref example-expected-returns-estimation),
[covariance](@ref example-covariance-estimation) and [higher moment](@ref example-higher-moment-estimation)
pages used the whole return sample, with equal weight on each observation. That is the right
default when the process that makes the returns does not change. In markets the process
changes. Volatility comes in clusters and correlations rise in a
crisis, and the risk of an asset today can differ a lot from its risk three years ago. A
windowed estimator computes a moment from a chosen part of the history, or weights some
observations more than others, instead of using the full sample with equal weights.

Each moment estimator has a windowed form: [`WindowedExpectedReturns`](@ref),
[`WindowedCovariance`](@ref), [`WindowedVariance`](@ref), [`WindowedCoskewness`](@ref) and
[`WindowedCokurtosis`](@ref). Each wraps a plain estimator of its moment, such as `ce` for the
covariance or `me` for the mean, and adds two keywords.

  - `window` is an integer or a vector of indices. An integer uses the last `window`
    observations. A vector uses the observations at those indices, which is a period you pick.
  - `w` holds optional observation weights, such as
    [`eweights`](https://juliastats.org/StatsBase.jl/stable/weights/), which give older
    observations less weight without a hard cutoff.

Each call returns one moment, a mean vector or a covariance matrix, for the selected window. The
estimator does not make a rolling time series. To move a window through time you write the loop
yourself, as the last section shows.

!!! tip "When to reach for this"
    Reach for a windowed estimator when you think that recent data tells you more than old data.
    Use a short trailing `window` or `eweights` for that. Use a vector `window` when you want a
    moment over one past episode, such as a crisis or a calm quarter. If you think that the full
    sample represents the future, the plain estimators of the earlier pages are the right default.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, Statistics, StatsBase, StatsPlots,
      PrettyTables, LinearAlgebra, Clarabel

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 2)) %" : v
    end
end;

#=
## 1. Data

The windows must differ for the comparison to show anything, so we take a longer slice than the
other examples, about four years of daily S&P 500 returns instead of one. The timestamps in
`rd.ts` let us select the days of a calendar episode later.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 1008):end]
rd = prices_to_returns(X)

#=
## 2. The length of a trailing window

The simplest control is an integer `window`, and the estimator then uses only the last `window`
observations. We print the annualised volatility of every asset for four trailing windows and
for the full sample.
=#

annvol(ce) = sqrt.(diag(cov(ce, rd.X))) .* sqrt(252)

windows = [60, 120, 252, 504]
vol_table = DataFrame(:asset => rd.nx)
vol_table[!, "full"] = annvol(PortfolioOptimisersCovariance())
for w in windows
    vol_table[!, "win=$w"] = annvol(WindowedCovariance(; window = w))
end

pretty_table(vol_table; formatters = [resfmt],
             title = "Annualised volatility by trailing-window length")

#=
On this slice the 60-day estimate is well above the 504-day and full-sample estimates for most
assets, because the last months were more volatile than the four-year average. A short window
follows a change of regime fast but carries more noise. A long window carries less noise but
follows a change slowly.
=#

#=
## 3. Exponential observation weights

A hard `window` cutoff gives the observation just inside the window its full weight, and the one
just outside no weight. Observation weights remove that step. `eweights(1:T, λ)` builds weights
that fall geometrically with the age of an observation, so the newest one weighs the most. We
pass them through the `w` field of the estimator to get a full-sample moment that weights recent
data more.

!!! note "A weighted covariance needs an uncorrected inner estimator"
    `eweights` returns a plain `StatsBase.Weights`, which does not support Bessel's bias
    correction. The default covariance applies that correction and throws an error on these
    weights. Set `corrected = false` on the inner `SimpleCovariance` when you pass `w`. The other
    way is to pass frequency, analytic or probability weights, which support the correction.
=#

T = size(rd.X, 1)
ew = eweights(1:T, 2 / (T + 1); scale = true)
uncorrected = PortfolioOptimisersCovariance(;
                                            ce = GeneralCovariance(;
                                                                   ce = SimpleCovariance(;
                                                                                         corrected = false)))

vol_w = DataFrame("asset" => rd.nx,
                  "full (equal)" => annvol(PortfolioOptimisersCovariance()),
                  "eweights" => annvol(WindowedCovariance(; ce = uncorrected, w = ew)),
                  "win=252" => annvol(WindowedCovariance(; window = 252)))

pretty_table(vol_w; formatters = [resfmt],
             title = "Annualised volatility with equal weights, exponential weights and a 252-day window")

#=
The weighted estimate keeps every observation. At the decay rate `2 / (T + 1)` the oldest
observation keeps about a seventh of the weight of the newest, so the weighted estimate stays
close to the equal-weight one.
=#

#=
## 4. Conditioning on a named historical episode

An integer window always uses the most recent observations. A vector window uses any indices
you give it, so we can estimate a moment over one episode. We compute the volatility of every
asset over the COVID crash of February to April 2020 and over a calm quarter in mid-2021, with
the same estimator.
=#

covid = findall(d -> Date("2020-02-19") <= d <= Date("2020-04-30"), rd.ts)
calm = findall(d -> Date("2021-04-01") <= d <= Date("2021-06-30"), rd.ts)

regime_vol = DataFrame("asset" => rd.nx,
                       "crash (Feb-Apr 2020)" =>
                           annvol(WindowedCovariance(; window = covid)),
                       "calm (2021 Q2)" => annvol(WindowedCovariance(; window = calm)),
                       "full" => annvol(PortfolioOptimisersCovariance()))

pretty_table(regime_vol; formatters = [resfmt],
             title = "Annualised volatility over the 2020 crash, a calm 2021 quarter and the full sample")

#=
The crash window gives several times the volatility of the calm window, and the full-sample
estimate averages that difference away. With a vector window you can compute the moments you
would get if the coming months looked like one past episode.
=#

#=
## 5. Windowed estimators inside a prior

A windowed estimator goes into a prior as a plain one does. Set the `me` and `ce` fields of an
[`EmpiricalPrior`](@ref) to windowed estimators and give the prior to an optimiser. The prior is
then an estimator and not a computed result, so we pass the [`ReturnsResult`](@ref) to
`optimise`. We solve a [`MinimumRisk`](@ref) portfolio for each window length and print its
largest weight and the number of assets it holds.
=#

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

function windowed_minrisk(w)
    pe = EmpiricalPrior(; me = WindowedExpectedReturns(; window = w),
                        ce = WindowedCovariance(; window = w))
    return optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pe, slv = slv)), rd)
end

full_book = optimise(MeanRisk(; obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)), rd)
books = [windowed_minrisk(w) for w in windows]

concentration = DataFrame("window" => ["full"; string.(windows)],
                          "max weight" => [maximum(full_book.w);
                                           [maximum(b.w) for b in books]],
                          "assets held" => [count(>(1e-4), full_book.w);
                                            [count(>(1e-4), b.w) for b in books]])

concfmt = (v, i, j) -> j == 2 && isa(v, Number) ? "$(round(v * 100, digits = 2)) %" : v
pretty_table(concentration; formatters = [concfmt],
             title = "Minimum-risk concentration by estimation window")

#=
The short windows give portfolios that hold fewer assets than the long windows and the full
sample. A short-window covariance carries more noise, and the optimiser puts its weight in the
few assets that looked least risky over that short window. The order is not strict. The 120-day
portfolio has the largest single weight of all, so the noise of a short window also makes the
weights less predictable.
=#

#=
## 6. Moving a window through time

A trailing window computed at every date shows how a moment changes over time. We compute the
252-day trailing volatility of AAPL at each date from day 252 on, and plot the series. The
full-sample estimate gives one number for the whole period.
=#

roll = 252
rolling_vol = [sqrt(cov(WindowedCovariance(; window = roll), rd.X[1:t, :])[1, 1]) *
               sqrt(252) for t in roll:T]
rolling_dates = rd.ts[roll:T]

plot(rolling_dates, rolling_vol; label = "AAPL trailing 252-day vol", xlabel = "date",
     ylabel = "annualised volatility", legend = :topright, lw = 2,
     title = "AAPL 252-day trailing volatility by date")

#=
The volatility rises in early 2020, when the days of the COVID crash enter the trailing window.
It falls 252 trading days later, when those days leave the window. A single windowed estimate is
one point of this curve.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New page (ADR 0014 coverage gap). All cells verified end-to-end under Kaimon (docs env,
#src   GKSwstype=100) on the last-1008-obs SP500 slice (1008×20).
#src - FINDING (documented inline as a !!! note): passing `eweights` (a plain `StatsBase.Weights`)
#src   to a windowed covariance whose inner `SimpleCovariance` keeps the default `corrected=true`
#src   throws a cryptic StatsBase error ("Weights type does not support bias correction: use
#src   FrequencyWeights, AnalyticWeights or ProbabilityWeights"). Fix is `corrected=false` on the
#src   inner SimpleCovariance (matches the test suite usage). Candidate docstring note on the
#src   Windowed* estimators' `w` field, or a clearer error from `robust_cov`.
#src - VERIFIED numbers (AAPL, annualised vol): full 0.345, win60 0.404, win120 0.360, win252 0.355,
#src   win504 0.308, eweights 0.342; COVID-window 0.769 vs calm-2021 0.211 (3.6x).
#src - VERIFIED MinimumRisk concentration by window: full 8 names/maxw 0.276, win60 5/0.341,
#src   win120 7/0.446, win252 10/0.370, win504 14/0.295 — all OptimisationSuccess (Clarabel).
#src   Non-monotone (win120 most concentrated) — kept honest in prose rather than smoothed.
#src - ERGO: windowed `me`/`ce` need the estimator form of the prior (pe=EmpiricalPrior(...) passed
#src   rd to optimise), not a precomputed prior — see estimator-vs-precomputed distinction.
#src - Rolling demo is caller-driven (loop over trailing windows); the estimator itself is
#src   single-window. Labelled as such so readers don't expect a built-in rolling type.
