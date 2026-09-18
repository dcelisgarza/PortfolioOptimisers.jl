```@meta
Description = "Simple covariance, private API of PortfolioOptimisers.jl: show_fields, covariance_centre_and_estimator, CovarianceState, Base.copy, covariance_state_seed, …"
```

# Simple covariance: private API

The covariance is an important measure of risk used in portfolio selection and performance analysis. The classic Markowitz [markowitz1952](@cite) portfolio uses the portfolio variance as its risk measure, which is computed from the covariance matrix and portfolio weights. Here we define the most basic covariance/correlation estimator.

## General covariance

```@docs
show_fields(::GeneralCovariance)
```

## Covariance

```@docs
show_fields(::Covariance)
covariance_centre_and_estimator
```

## Incremental fit

The full-moment sample covariance folds one observation at a time, so a long history need not be held or re-read. One state serves both estimators, because they run the same recursion over the same three quantities. [`partial_fit!`](@ref) returns a new estimator whose `cache` field carries the state, and `cov` reads the fit off the estimator alone.

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

With a [`CoveragePolicy`](@ref) in its `cvg` field the estimator fits each pair on the observations that pair shares, and [`PortfolioOptimisers.coverage_covariance`](@ref) routes between that arm and the Coverage Universe one.

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
