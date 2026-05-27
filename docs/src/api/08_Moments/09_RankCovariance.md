# Rank Covariances

```@docs
KendallCovariance
factory(ce::KendallCovariance, w::ObsWeights)
cov(::KendallCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor(::KendallCovariance, X::MatNum; dims::Int = 1, kwargs...)
moment_view(ce::KendallCovariance, i)
SpearmanCovariance
factory(ce::SpearmanCovariance, w::ObsWeights)
cov(::SpearmanCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor(::SpearmanCovariance, X::MatNum; dims::Int = 1, kwargs...)
moment_view(ce::SpearmanCovariance, i)
RankCovarianceEstimator
```
