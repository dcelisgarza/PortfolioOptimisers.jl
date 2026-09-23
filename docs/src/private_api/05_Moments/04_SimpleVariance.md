```@meta
Description = "Simple variance and standard deviation, private API of PortfolioOptimisers.jl: show_fields, simple_variance_kernel, SimpleVarianceState, …"
```

# Simple variance and standard deviation: private API

The library uses the variance of each asset in several places, such as some estimators of the expected returns and of the covariance, performance analysis, and the constraints it generates. [`SimpleVariance`](@ref) computes the sample variance, and the standard deviation as its square root.

```@docs
show_fields(::SimpleVariance)
simple_variance_kernel
```

## Incremental fit

The sample variance can take the observations one at a time as well. [`partial_fit!`](@ref) returns a new estimator whose `cache` field holds the running state, and `var(ve)` computes the estimate from that field, with no data.

```@docs
SimpleVarianceState
variance_state_seed
Base.copy(x::SimpleVarianceState)
```

## Available-case fit

With a [`CoveragePolicy`](@ref) in its `cvg` field, the estimator computes the variance of each asset from the rows where that asset's return is finite and the asset is active. Without a policy, it computes the ordinary sample variance over the assets that have no gap in the window. [`PortfolioOptimisers.coverage_variance`](@ref) selects one of the two fits from the type of the `cvg` field.

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
