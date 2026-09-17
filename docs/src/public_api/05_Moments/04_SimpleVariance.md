```@meta
Description = "Simple variance and standard deviation, public API of PortfolioOptimisers.jl: SimpleVariance, std, var, partial_fit!, port_opt_view, merge_states, …"
```

# Simple variance and standard deviation

The variance is used throughout the library, it can be used as part of the expected return, covariance estimation, performance analysis, and constraint generation. It is trivial to compute the standard deviation from the variance, so we provide those too.

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

The sample variance folds one observation at a time, so a long history need not be held or re-read. [`partial_fit!`](@ref) returns a new estimator whose `cache` field carries the state, and `var` reads the fit off the estimator alone.

```@docs
partial_fit!(state::SimpleVarianceState, x::VecNum)
partial_fit!(ve::SimpleVariance, X::MatNum; dims::Int = 1)
partial_fit!(ve::SimpleVariance, x::VecNum)
var(ve::SimpleVariance, state::SimpleVarianceState)
port_opt_view(x::SimpleVarianceState, i, args...)
merge_states(a::SimpleVarianceState, b::SimpleVarianceState)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field the estimator fits each asset on that asset's own finite and active observations, and [`PortfolioOptimisers.coverage_variance`](@ref) routes between that arm and the Coverage Universe one.

```@docs
partial_fit!(state::SimpleVarianceState, x::VecNum, ::Nothing, ::Option{<:AbstractVector{<:Bool}})
partial_fit!(state::SimpleVarianceState, x::VecNum, cvg::CoveragePolicy, active_mask::Option{<:AbstractVector{<:Bool}})
fold_inactive!(::ResetCoverage, state::SimpleVarianceState, ni::AbstractVector{<:Bool})
```
