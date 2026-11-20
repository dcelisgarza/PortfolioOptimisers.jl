```@meta
Description = "Abstract covariance variance, private API of PortfolioOptimisers.jl: VarianceCount, variance_series, variance_count."
```

# Abstract covariance variance: private API

## Types

```@docs
VarianceCount
```

## Functions

```@docs
variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
variance_count
```
