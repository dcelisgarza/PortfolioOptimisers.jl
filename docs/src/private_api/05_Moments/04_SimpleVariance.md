```@meta
Description = "Simple variance and standard deviation, private API of PortfolioOptimisers.jl: show_fields, simple_variance_kernel, SimpleVarianceState, …"
```

# Simple variance and standard deviation: private API

The variance is used throughout the library, it can be used as part of the expected return, covariance estimation, performance analysis, and constraint generation. It is trivial to compute the standard deviation from the variance, so we provide those too.

```@docs
show_fields(::SimpleVariance)
simple_variance_kernel
```

## Incremental fit

The sample variance folds one observation at a time, so a long history need not be held or re-read. [`partial_fit!`](@ref) returns a new estimator whose `cache` field carries the state, and `var` reads the fit off the estimator alone.

```@docs
SimpleVarianceState
variance_state_seed
Base.copy(x::SimpleVarianceState)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field the estimator fits each asset on that asset's own finite and active observations, and [`PortfolioOptimisers.coverage_variance`](@ref) routes between that arm and the Coverage Universe one.

```@docs
PortfolioOptimisers.coverage_variance
PortfolioOptimisers.coverage_variance(f, ve::SimpleVariance, ::Nothing, me::AbstractExpectedReturnsEstimator, X::MatNum)
PortfolioOptimisers.coverage_variance(f, ve::SimpleVariance, cvg::CoveragePolicy, ::AbstractExpectedReturnsEstimator, X::MatNum)
PortfolioOptimisers.coverage_variance(f, ve::SimpleVariance, ::Nothing, X::VecNum)
PortfolioOptimisers.coverage_variance(f, ve::SimpleVariance, cvg::CoveragePolicy, X::VecNum)
PortfolioOptimisers.coverage_variance(ve::SimpleVariance, ::Nothing, state::SimpleVarianceState)
PortfolioOptimisers.coverage_variance(ve::SimpleVariance, cvg::CoveragePolicy, state::SimpleVarianceState)
PortfolioOptimisers.coverage_moment_map
```
