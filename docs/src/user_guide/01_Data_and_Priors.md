The source files can be found in [user_guide/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/user_guide/).

```@meta
EditURL = "../../../user_guide/01_Data_and_Priors.jl"
```

# Data and priors

The first pipeline stage turns raw prices into a **prior** — the expected-returns vector and
covariance matrix every optimiser consumes. Two calls cover the common path:
[`prices_to_returns`](@ref) and [`prior`](@ref). This page is the quick tour; for the full menu
of moment estimators and view-based priors, see the
[moments & priors examples](../examples/2_moments_priors/01_Expected_Returns_Estimation.md).

````@example 01_Data_and_Priors
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, LinearAlgebra,
      StatsPlots, GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. Prices to returns

Price data usually arrives from an API and must be converted to returns.
[`prices_to_returns`](@ref) handles asset, factor, and benchmark prices (plus implied
volatilities and volatility premiums), validates that the series are consistent, and can impute
missing prices ([Impute.jl](https://github.com/invenia/Impute.jl)) and collapse to lower
frequencies ([TimeSeries.jl](https://github.com/JuliaStats/TimeSeries.jl)). Given a single
`TimeArray` of prices it returns a [`ReturnsResult`](@ref) holding the asset names `nx` and the
return matrix `X`.

Real price tables are rarely clean — newly listed or delisted names leave leading/trailing gaps,
halts and stale quotes leave flat stretches, and exchanges keep different holiday calendars. The
`missing_col_percent` / `missing_row_percent` filters and `impute_method` handle all of it in this
one call, *before* differencing prices into returns; the
[Data preprocessing and imputation](../examples/1_foundations/02_Data_Preprocessing.md) example is
the deep dive.

````@example 01_Data_and_Priors
X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
````

When you also pass factor and benchmark price series, `prices_to_returns` aligns them on matching
timestamps and carries the factor returns `F` and benchmark `iv` through on the same
[`ReturnsResult`](@ref) — everything downstream then has the data it needs.

## 2. Returns to a prior

[`prior`](@ref) applies a prior estimator to a [`ReturnsResult`](@ref) and returns the moments.
The blessed default is [`EmpiricalPrior`](@ref) — the sample mean and covariance.

````@example 01_Data_and_Priors
pr = prior(EmpiricalPrior(), rd)

pretty_table(DataFrame("Asset" => rd.nx, "Expected return" => pr.mu,
                       "Volatility" => sqrt.(diag(pr.sigma))); formatters = [resfmt],
             title = "Empirical prior: per-asset mean and volatility")
````

## 3. Swapping the prior

`EmpiricalPrior` is only the starting point. Every prior estimator has the same `prior(pe, rd)`
interface, so swapping one in is a one-line change. The common alternatives:

- [`FactorPrior`](@ref) — moments from a factor model (deep dive:
    [Factor Priors](../examples/2_moments_priors/04_Factor_Priors.md)).
- [`BlackLittermanPrior`](@ref) — blend market-equilibrium moments with your views (deep dive:
    [Black–Litterman](../examples/2_moments_priors/05_Black_Litterman.md)). The family extends to
    [`BayesianBlackLittermanPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) (views on factor
    premia) and [`AugmentedBlackLittermanPrior`](@ref) (asset *and* factor views together) — see
    [Advanced Black–Litterman](../examples/2_moments_priors/06_Advanced_Black_Litterman.md).
- [`EntropyPoolingPrior`](@ref) / [`OpinionPoolingPrior`](@ref) — reweight the empirical
    scenarios to satisfy views on any moment (deep dives:
    [Entropy Pooling](../examples/2_moments_priors/07_Entropy_Pooling.md),
    [Opinion Pooling](../examples/2_moments_priors/08_Opinion_Pooling.md)).

The covariance estimator inside a prior is itself swappable (shrinkage, denoising, Gerber, …);
see [Covariance Estimation](../examples/2_moments_priors/02_Covariance_Estimation.md). Any moment
estimator can also be **windowed** — restricted to a trailing window or recency-weighted — when
the recent regime is more informative than the full sample (deep dive:
[Windowed Estimators](../examples/2_moments_priors/10_Windowed_Estimators.md); regime-adjusted
estimators are a related sibling).

## 4. A first look at the data

[`plot_prior`](@ref) summarises a prior in one figure — expected returns, per-asset volatility,
and the correlation matrix — a quick sanity check before optimising.

````@example 01_Data_and_Priors
plot_prior(pr, rd)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
