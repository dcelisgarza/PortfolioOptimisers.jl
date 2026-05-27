# Mutual Information Covariance

```@docs
MutualInfoCovariance
factory(ce::MutualInfoCovariance, w::ObsWeights)
cov(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)
moment_view(ce::MutualInfoCovariance, i)
```
