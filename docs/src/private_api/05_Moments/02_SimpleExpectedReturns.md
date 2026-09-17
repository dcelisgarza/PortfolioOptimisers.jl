```@meta
Description = "Simple expected returns, private API of PortfolioOptimisers.jl: show_fields, SimpleExpectedReturnsState, expected_returns_state_seed, merge_states, …"
```

# Simple expected returns: private API

The most basic moment is the simple expected return. These types and functions implement it.

```@docs
show_fields(::SimpleExpectedReturns)
```

## Incremental fit

The sample mean folds one observation at a time, so a long history need not be held or re-read. [`partial_fit!`](@ref) returns a new estimator whose `cache` field carries the state, and `mean` reads the fit off the estimator alone.

```@docs
SimpleExpectedReturnsState
expected_returns_state_seed
merge_states(a::SimpleExpectedReturnsState, b::SimpleExpectedReturnsState)
Base.copy(x::SimpleExpectedReturnsState)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field the estimator fits each asset on that asset's own finite and active observations, and [`PortfolioOptimisers.coverage_mean`](@ref) routes between that arm and the Coverage Universe one.

```@docs
PortfolioOptimisers.coverage_mean
PortfolioOptimisers.coverage_mean(me::SimpleExpectedReturns, ::Nothing, X::MatNum)
PortfolioOptimisers.coverage_mean(me::SimpleExpectedReturns, cvg::CoveragePolicy, X::MatNum)
PortfolioOptimisers.coverage_mean(::SimpleExpectedReturns, ::Nothing, state::SimpleExpectedReturnsState)
PortfolioOptimisers.coverage_mean(::SimpleExpectedReturns, cvg::CoveragePolicy, state::SimpleExpectedReturnsState)
PortfolioOptimisers.fold_inactive!(::ResetCoverage, state::SimpleExpectedReturnsState, ni::AbstractVector{<:Bool})
```
