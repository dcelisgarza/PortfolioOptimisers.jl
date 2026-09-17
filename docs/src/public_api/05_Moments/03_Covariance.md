```@meta
Description = "Simple covariance, public API of PortfolioOptimisers.jl: GeneralCovariance, cov, cor, Covariance, partial_fit!, port_opt_view, merge_states, fold_inactive!."
```

# [Simple covariance](@id api-covariance)

The covariance is an important measure of risk used in portfolio selection and performance analysis. The classic Markowitz [markowitz1952](@cite) portfolio uses the portfolio variance as its risk measure, which is computed from the covariance matrix and portfolio weights. Here we define the most basic covariance/correlation estimator.

## General covariance

```@docs
GeneralCovariance
cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
```

## Covariance

```@docs
Covariance
cov(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cov(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
```

## Incremental fit

The full-moment sample covariance folds one observation at a time, so a long history need not be held or re-read. One state serves both estimators, because they run the same recursion over the same three quantities. [`partial_fit!`](@ref) returns a new estimator whose `cache` field carries the state, and `cov` reads the fit off the estimator alone.

```@docs
partial_fit!(state::CovarianceState, x::VecNum)
partial_fit!(ce::GeneralCovariance, X::MatNum; dims::Int = 1)
partial_fit!(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1)
partial_fit!(ce::GeneralCovariance, x::VecNum)
partial_fit!(ce::Covariance{<:Any, <:Any, <:FullMoment}, x::VecNum)
partial_fit!(ce::Covariance, ::VecNum_MatNum; kwargs...)
cov(ce::Union{<:GeneralCovariance, <:Covariance{<:Any, <:Any, <:FullMoment}}, state::CovarianceState)
cor(ce::Union{<:GeneralCovariance, <:Covariance{<:Any, <:Any, <:FullMoment}}, state::CovarianceState)
port_opt_view(x::CovarianceState, i, args...)
merge_states(a::CovarianceState, b::CovarianceState)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field the estimator fits each pair on the observations that pair shares, and [`PortfolioOptimisers.coverage_covariance`](@ref) routes between that arm and the Coverage Universe one.

```@docs
partial_fit!(state::CovarianceState, x::VecNum, ::Nothing, ::Option{<:AbstractVector{<:Bool}})
partial_fit!(state::CovarianceState, x::VecNum, cvg::CoveragePolicy, active_mask::Option{<:AbstractVector{<:Bool}})
fold_inactive!(::ResetCoverage, state::CovarianceState, ni::AbstractVector{<:Bool})
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
