# Variance from covariance

```@docs
var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                        kwargs...)
std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                        kwargs...)
variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                         kwargs...)
cov(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
cor(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
std(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
```
