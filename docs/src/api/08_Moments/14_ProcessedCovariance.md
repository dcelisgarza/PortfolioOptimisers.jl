# Processed covariance

```@docs
ProcessedCovariance
factory(ce::ProcessedCovariance, w::ObsWeights)
cov(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)
cor(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)
moment_view(ce::ProcessedCovariance, i)
```
