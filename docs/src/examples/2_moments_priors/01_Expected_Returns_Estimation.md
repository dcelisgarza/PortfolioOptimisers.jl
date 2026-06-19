The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/2_moments_priors/01_Expected_Returns_Estimation.jl"
```

# Expected returns estimation

The sample mean is the noisiest ingredient in portfolio optimisation. With only a year of
daily data the per-asset average return is estimated very imprecisely, and any optimiser that
*chases* return — maximum return, maximum risk-adjusted ratio, high-risk-aversion utility —
amplifies that noise into extreme, unstable weights. **Shrinkage** estimators pull the raw
sample mean toward a structured target, trading a little bias for a large reduction in
variance.

`PortfolioOptimisers` exposes the expected-returns estimator as the `me` field of a prior.
The default is the plain sample mean ([`SimpleExpectedReturns`](@ref)); the shrinkage
estimator [`ShrunkExpectedReturns`](@ref) wraps it with an algorithm —
[`BayesStein`](@ref), [`BodnarOkhrinParolya`](@ref) or [`JamesStein`](@ref) — each of which
shrinks toward one of three targets: [`GrandMean`](@ref), [`VolatilityWeighted`](@ref) or
[`MeanSquaredError`](@ref).

!!! tip "When to reach for this"
    Reach for a shrunk expected-returns estimator whenever your objective depends on the mean
    — [`MaximumReturn`](@ref), [`MaximumRatio`](@ref), or a risk-averse
    [`MaximumUtility`](@ref) — and especially when the estimation window is short relative to
    the number of assets. If you only ever run [`MinimumRisk`](@ref) or risk-budgeting (which
    ignore the mean), the estimator hardly matters and the default is fine.

````@example 01_Expected_Returns_Estimation
using PortfolioOptimisers, PrettyTables

mmtfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=4)) %" : v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. ReturnsResult data

We use the same S&P 500 slice as the other examples.

````@example 01_Expected_Returns_Estimation
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
````

## 2. Expected-returns estimators

We build one prior per estimator, varying **only** the `me` field so the covariance is
identical across them — any difference in the optimisation later is due purely to the expected
returns. We compare the plain sample mean against Bayes–Stein, Bodnar–Okhrin–Parolya and
James–Stein shrinkage toward different targets.

````@example 01_Expected_Returns_Estimation
mes = ["Vanilla" => SimpleExpectedReturns(),
       "BS(GM)" => ShrunkExpectedReturns(; alg = BayesStein(; tgt = GrandMean())),
       "BS(VW)" => ShrunkExpectedReturns(; alg = BayesStein(; tgt = VolatilityWeighted())),
       "BOP(MSE)" =>
           ShrunkExpectedReturns(; alg = BodnarOkhrinParolya(; tgt = MeanSquaredError())),
       "JS(GM)" => ShrunkExpectedReturns(; alg = JamesStein(; tgt = GrandMean()))]

prs = [k => prior(EmpiricalPrior(; me = me), rd) for (k, me) in mes]
````

The expected-returns vectors side by side. Note that shrinkage toward the grand mean
(`BS(GM)`, `JS(GM)`) preserves the cross-sectional average while pulling the spread in, whereas
volatility-weighted and MSE targets shift the level too.

````@example 01_Expected_Returns_Estimation
pretty_table(DataFrame(["Assets" => rd.nx; [k => p.mu for (k, p) in prs]]);
             formatters = [mmtfmt], title = "Expected returns by estimator")
````

## 3. Visualising the shrinkage

[`plot_mu`](@ref) makes the pull-toward-target visible: compared with the raw sample mean, the
shrunk estimator compresses the dispersion of the per-asset expected returns.

Vanilla sample-mean expected returns.

````@example 01_Expected_Returns_Estimation
using StatsPlots, GraphRecipes
````

Bayes–Stein (volatility-weighted target) expected returns.

````@example 01_Expected_Returns_Estimation
plot_mu(prs[1].second, rd.nx)
plot_mu(prs[3].second, rd.nx)
````

## 4. Why it matters: a return-seeking optimisation

Expected returns only bite when the objective uses them. We maximise the risk-adjusted ratio
with each prior in turn (same covariance, different mean) and compare the resulting weights.
The noisy sample mean concentrates; shrinkage spreads the allocation out and stabilises it.

````@example 01_Expected_Returns_Estimation
using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

ress = [k => optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                               opt = JuMPOptimiser(; pe = p, slv = slv))) for (k, p) in prs]

pretty_table(DataFrame(["Assets" => rd.nx; [k => r.w for (k, r) in ress]]);
             formatters = [resfmt], title = "Maximum-ratio weights by mu estimator")
````

The composition plot drives the point home: swapping the expected-returns estimator alone
reshapes the maximum-ratio portfolio.

````@example 01_Expected_Returns_Estimation
plot_stacked_bar_composition([r for (_, r) in ress], rd;
                             xticks = (1:length(ress), [k for (k, _) in ress]))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
