```@meta
Description = "Simple expected returns, private API of PortfolioOptimisers.jl: show_fields, SimpleExpectedReturnsState, expected_returns_state_seed, Base.copy, …"
```

# Simple expected returns: private API

The simplest estimate of the expected return of an asset is its sample mean, which [`SimpleExpectedReturns`](@ref) computes. The entries below support it.

```@docs
show_fields(::SimpleExpectedReturns)
```

## Incremental fit

The sample mean can take the observations one at a time, so you do not need to keep a long history or read it again. [`partial_fit!`](@ref) returns a new estimator whose `cache` field holds the running count and mean. `mean(me)` then computes the estimate from that field, with no data.

```@docs
SimpleExpectedReturnsState
expected_returns_state_seed
Base.copy(x::SimpleExpectedReturnsState)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field, the estimator computes the mean of each asset from the rows where that asset's return is finite and the asset is active. This is the available-case fit. Without a policy, the estimator computes the ordinary sample mean over the assets that have no gap in the window. [`PortfolioOptimisers.coverage_mean`](@ref) selects one of the two fits from the type of the `cvg` field.

```@docs
PortfolioOptimisers.coverage_mean
PortfolioOptimisers.coverage_mean(me::SimpleExpectedReturns, ::Nothing, X::MatNum)
PortfolioOptimisers.coverage_mean(me::SimpleExpectedReturns, cvg::CoveragePolicy, X::MatNum)
PortfolioOptimisers.coverage_mean(::SimpleExpectedReturns, ::Nothing, state::SimpleExpectedReturnsState)
PortfolioOptimisers.coverage_mean(::SimpleExpectedReturns, cvg::CoveragePolicy, state::SimpleExpectedReturnsState)
```
