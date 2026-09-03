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
merge_states(a::CovarianceState, b::CovarianceState)
covariance_state_seed
partial_fit_corrected(ce::StatsBase.SimpleCovariance)
partial_fit_corrected(ce::GeneralCovariance)
partial_fit_corrected(ce::Covariance{<:Any, <:Any, <:FullMoment})
partial_fit_corrected(ce::StatsBase.CovarianceEstimator)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
