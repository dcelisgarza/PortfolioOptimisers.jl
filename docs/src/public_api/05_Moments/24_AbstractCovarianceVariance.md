```@meta
Description = "Abstract covariance variance, public API of PortfolioOptimisers.jl: var, std, cov, cor."
```

# Abstract covariance variance

```@docs
var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
cov(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
cor(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
std(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
```
