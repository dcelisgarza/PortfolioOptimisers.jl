```@meta
Description = "Simple variance and standard deviation, public API of PortfolioOptimisers.jl: SimpleVariance, std, var, partial_fit!, port_opt_view, merge_states, …"
```

# Simple variance and standard deviation

The library uses the variance of each asset in several places: in some expected returns estimators, in covariance estimation, in performance analysis and in constraint generation. The estimators below compute the variance, and the standard deviation, which is its square root.

```@docs
SimpleVariance
std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
std(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
std(ve::SimpleVariance, X::VecNum; mean = nothing)
var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
var(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
var(ve::SimpleVariance, X::VecNum; mean = nothing)
```

## Incremental fit

Like the sample mean, the sample variance updates with each new block of observations. [`partial_fit!`](@ref) stores the running state in the `cache` field of the estimator it returns, and `var(ve)` computes the estimate from that state, with no data.

```@docs
partial_fit!(state::SimpleVarianceState, x::VecNum)
partial_fit!(ve::SimpleVariance, X::MatNum; dims::Int = 1)
partial_fit!(ve::SimpleVariance, x::VecNum)
var(ve::SimpleVariance, state::SimpleVarianceState)
port_opt_view(x::SimpleVarianceState, i, args...)
merge_states(a::SimpleVarianceState, b::SimpleVarianceState)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field, the variance of each asset uses only the observations where that asset is finite and active. Without a policy, it fits on the coverage universe, the assets that are finite and active at every observation. [`PortfolioOptimisers.coverage_variance`](@ref) chooses between the two fits.

```@docs
partial_fit!(state::SimpleVarianceState, x::VecNum, ::Nothing, ::Option{<:AbstractVector{<:Bool}})
partial_fit!(state::SimpleVarianceState, x::VecNum, cvg::CoveragePolicy, active_mask::Option{<:AbstractVector{<:Bool}})
fold_inactive!(::ResetCoverage, state::SimpleVarianceState, ni::AbstractVector{<:Bool})
```
