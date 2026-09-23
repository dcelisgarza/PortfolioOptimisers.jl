```@meta
Description = "Simple covariance, public API of PortfolioOptimisers.jl: GeneralCovariance, cov, cor, Covariance, partial_fit!, port_opt_view, merge_states, fold_inactive!."
```

# [Simple covariance](@id api-covariance)

The covariance matrix measures how the returns of the assets vary together. The mean-variance portfolio of Markowitz [markowitz1952](@cite) takes the portfolio variance as its risk, which is `w' Σ w` for the weights `w` and the covariance matrix `Σ`. This page has the sample covariance and correlation estimators.

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

Like the sample mean, the sample covariance with `FullMoment` updates with each new block of observations. `GeneralCovariance` and `Covariance` use the same state, `CovarianceState`, and update it in the same way. [`partial_fit!`](@ref) stores the state in the `cache` field of the estimator it returns, and `cov(ce)` computes the estimate from that state, with no data.

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

With a [`CoveragePolicy`](@ref) in its `cvg` field, the estimator fits each pair of assets on the observations where both assets are finite and active. Without a policy, it fits on the coverage universe, the assets that are finite and active at every observation. [`PortfolioOptimisers.coverage_covariance`](@ref) chooses between the two fits.

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
