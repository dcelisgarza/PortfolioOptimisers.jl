# Simple expected returns

The most basic moment is the simple expected return. These types and functions implement it.

```@docs
SimpleExpectedReturns
show_fields(::SimpleExpectedReturns)
mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
```

## Incremental fit

The sample mean folds one observation at a time, so a long history need not be held or re-read. [`partial_fit!`](@ref) returns a new estimator whose `cache` field carries the state, and `mean` reads the fit off the estimator alone.

```@docs
SimpleExpectedReturnsState
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum)
partial_fit!(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1)
partial_fit!(me::SimpleExpectedReturns, x::VecNum)
expected_returns_state_seed
mean(me::SimpleExpectedReturns, state::SimpleExpectedReturnsState)
merge_states(a::SimpleExpectedReturnsState, b::SimpleExpectedReturnsState)
Base.copy(x::SimpleExpectedReturnsState)
port_opt_view(x::SimpleExpectedReturnsState, i, args...)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field the estimator fits each asset on that asset's own finite and active observations, and [`PortfolioOptimisers.coverage_mean`](@ref) routes between that arm and the Coverage Universe one.

```@docs
PortfolioOptimisers.coverage_mean
PortfolioOptimisers.coverage_mean(me::SimpleExpectedReturns, ::Nothing, X::MatNum)
PortfolioOptimisers.coverage_mean(me::SimpleExpectedReturns, cvg::CoveragePolicy, X::MatNum)
PortfolioOptimisers.coverage_mean(::SimpleExpectedReturns, ::Nothing, state::SimpleExpectedReturnsState)
PortfolioOptimisers.coverage_mean(::SimpleExpectedReturns, cvg::CoveragePolicy, state::SimpleExpectedReturnsState)
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum, ::Nothing, ::Option{<:AbstractVector{<:Bool}})
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum, cvg::CoveragePolicy, active_mask::Option{<:AbstractVector{<:Bool}})
PortfolioOptimisers.fold_inactive!(::ResetCoverage, state::SimpleExpectedReturnsState, ni::AbstractVector{<:Bool})
```
