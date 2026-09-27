```@meta
Description = "Simple expected returns, public API of PortfolioOptimisers.jl: SimpleExpectedReturns, mean, partial_fit!, port_opt_view, merge_states, fold_inactive!."
```

# Simple expected returns

The simplest estimate of the expected returns is the mean return of each asset. The types and functions below compute it, with or without observation weights.

```@docs
SimpleExpectedReturns
mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
```

## Incremental fit

The sample mean updates with each new block of observations, and does not read the earlier observations again. [`partial_fit!`](@ref) returns a new estimator whose `cache` field holds the running state, and `mean(me)` returns the estimate from the estimator alone.

```@docs
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum)
partial_fit!(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1)
partial_fit!(me::SimpleExpectedReturns, x::VecNum)
mean(me::SimpleExpectedReturns, state::SimpleExpectedReturnsState)
port_opt_view(x::SimpleExpectedReturnsState, i, args...)
merge_states(a::SimpleExpectedReturnsState, b::SimpleExpectedReturnsState)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field, the estimator fits each asset on the observations where that asset is finite and active. Without a policy, it fits on the coverage universe, the assets that are finite and active at every observation. [`PortfolioOptimisers.coverage_mean`](@ref) chooses between the two fits.

```@docs
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum, ::Nothing, ::Option{<:AbstractVector{<:Bool}})
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum, cvg::CoveragePolicy, active_mask::Option{<:AbstractVector{<:Bool}})
fold_inactive!(::ResetCoverage, state::SimpleExpectedReturnsState, ni::AbstractVector{<:Bool})
```
