```@meta
Description = "Base robust covariance, public API of PortfolioOptimisers.jl: AbstractRobustCovarianceEstimator, RobustCovarianceResult, cov, cor."
```

# Base robust covariance

```@docs
AbstractRobustCovarianceEstimator
RobustCovarianceResult
cov(ce::AbstractRobustCovarianceEstimator, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::AbstractRobustCovarianceEstimator, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
```
