```@meta
Description = "Simple expected returns, public API of PortfolioOptimisers.jl: SimpleExpectedReturns, mean, partial_fit!, port_opt_view, merge_states, fold_inactive!."
```

# Simple expected returns

The most basic moment is the simple expected return. These types and functions implement it.

```@docs
SimpleExpectedReturns
mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
```

## Incremental fit

The sample mean folds one observation at a time, so a long history need not be held or re-read. [`partial_fit!`](@ref) returns a new estimator whose `cache` field carries the state, and `mean` reads the fit off the estimator alone.

```@docs
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum)
partial_fit!(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1)
partial_fit!(me::SimpleExpectedReturns, x::VecNum)
mean(me::SimpleExpectedReturns, state::SimpleExpectedReturnsState)
port_opt_view(x::SimpleExpectedReturnsState, i, args...)
merge_states(a::SimpleExpectedReturnsState, b::SimpleExpectedReturnsState)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field the estimator fits each asset on that asset's own finite and active observations, and [`PortfolioOptimisers.coverage_mean`](@ref) routes between that arm and the Coverage Universe one.

```@docs
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum, ::Nothing, ::Option{<:AbstractVector{<:Bool}})
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum, cvg::CoveragePolicy, active_mask::Option{<:AbstractVector{<:Bool}})
fold_inactive!(::ResetCoverage, state::SimpleExpectedReturnsState, ni::AbstractVector{<:Bool})
```
