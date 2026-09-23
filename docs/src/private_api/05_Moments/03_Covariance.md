```@meta
Description = "Simple covariance, private API of PortfolioOptimisers.jl: show_fields, library_covariance_estimator, covariance_centre_and_estimator, CovarianceState, …"
```

# Simple covariance: private API

The weights of a portfolio and the covariance matrix of its assets give the portfolio variance, which the classic Markowitz portfolio [markowitz1952](@cite) uses as its measure of risk. The entries below support the two sample estimators of the covariance and the correlation, [`GeneralCovariance`](@ref) and [`Covariance`](@ref).

## General covariance

```@docs
show_fields(::GeneralCovariance)
library_covariance_estimator
```

## Covariance

```@docs
show_fields(::Covariance)
covariance_centre_and_estimator
```

## Incremental fit

The full-moment sample covariance can also take the observations one at a time. Both estimators use one state type, because they update the same three quantities: the observation count, the mean, and the sum of the outer products of the deviations from the mean. [`partial_fit!`](@ref) returns a new estimator whose `cache` field holds that state, and `cov(ce)` computes the estimate from it, with no data.

```@docs
CovarianceState
Base.copy(x::CovarianceState)
covariance_state_seed
partial_fit_corrected(ce::StatsBase.SimpleCovariance)
partial_fit_corrected(ce::GeneralCovariance)
partial_fit_corrected(ce::Covariance{<:Any, <:Any, <:FullMoment})
partial_fit_corrected(ce::StatsBase.CovarianceEstimator)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field, the estimator computes the covariance of each pair of assets from the rows that the pair shares, where both returns are finite and both assets are active. Without a policy, it computes the ordinary sample covariance over the assets that have no gap in the window. [`PortfolioOptimisers.coverage_covariance`](@ref) selects one of the two fits from the type of the `cvg` field.

```@docs
PortfolioOptimisers.coverage_covariance
PortfolioOptimisers.coverage_covariance(f, ce::Covariance{<:Any, <:Any, <:FullMoment}, ::Nothing, X::MatNum)
PortfolioOptimisers.coverage_covariance(f, ce::Covariance{<:Any, <:Any, <:FullMoment}, cvg::CoveragePolicy, X::MatNum)
PortfolioOptimisers.coverage_covariance(f, ce::Covariance{<:Any, <:Any, <:SemiMoment}, ::Nothing, X::MatNum)
PortfolioOptimisers.coverage_covariance(f, ce::Covariance{<:Any, <:Any, <:SemiMoment}, cvg::CoveragePolicy, X::MatNum)
PortfolioOptimisers.coverage_covariance(ce::Union{<:GeneralCovariance, <:Covariance{<:Any, <:Any, <:FullMoment}}, ::Nothing, state::CovarianceState)
PortfolioOptimisers.coverage_covariance(ce::Covariance{<:Any, <:Any, <:FullMoment}, cvg::CoveragePolicy, state::CovarianceState)
PortfolioOptimisers.coverage_correlation
PortfolioOptimisers.coverage_policy
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
