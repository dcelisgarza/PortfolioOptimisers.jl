# Detone covariance

```@docs
DetoneCovariance
factory(ce::DetoneCovariance, w::ObsWeights)
cov(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)
cor(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)
moment_view(ce::DetoneCovariance, i)
```
