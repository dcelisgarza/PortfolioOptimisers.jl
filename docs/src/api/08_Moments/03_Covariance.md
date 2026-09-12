# [Simple covariance](@id api-covariance)

The covariance is an important measure of risk used in portfolio selection and performance analysis. The classic Markowitz [markowitz1952](@cite) portfolio uses the portfolio variance as its risk measure, which is computed from the covariance matrix and portfolio weights. Here we define the most basic covariance/correlation estimator.

## General covariance

```@docs
GeneralCovariance
show_fields(::GeneralCovariance)
cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
```

## [Covariance](@id api-covariance)

```@docs
Covariance
show_fields(::Covariance)
covariance_centre_and_estimator
cov(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cov(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
```

## Incremental fit

The full-moment sample covariance folds one observation at a time, so a long history need not be held or re-read. One state serves both estimators, because they run the same recursion over the same three quantities. [`partial_fit!`](@ref) returns a new estimator whose `cache` field carries the state, and `cov` reads the fit off the estimator alone.

```@docs
CovarianceState
partial_fit!(state::CovarianceState, x::VecNum)
partial_fit!(ce::GeneralCovariance, X::MatNum; dims::Int = 1)
partial_fit!(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1)
partial_fit!(ce::GeneralCovariance, x::VecNum)
partial_fit!(ce::Covariance{<:Any, <:Any, <:FullMoment}, x::VecNum)
partial_fit!(ce::Covariance, ::VecNum_MatNum; kwargs...)
cov(ce::Union{<:GeneralCovariance, <:Covariance{<:Any, <:Any, <:FullMoment}}, state::CovarianceState)
cor(ce::Union{<:GeneralCovariance, <:Covariance{<:Any, <:Any, <:FullMoment}}, state::CovarianceState)
merge_states(a::CovarianceState, b::CovarianceState)
Base.copy(x::CovarianceState)
port_opt_view(x::CovarianceState, i, args...)
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
partial_fit!(state::CovarianceState, x::VecNum, ::Nothing, ::Option{<:AbstractVector{<:Bool}})
partial_fit!(state::CovarianceState, x::VecNum, cvg::CoveragePolicy, active_mask::Option{<:AbstractVector{<:Bool}})
PortfolioOptimisers.fold_inactive!(::ResetCoverage, state::CovarianceState, ni::AbstractVector{<:Bool})
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
