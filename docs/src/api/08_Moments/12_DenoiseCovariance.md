# Denoise covariance

```@docs
DenoiseCovariance
factory(ce::DenoiseCovariance, w::ObsWeights)
cov(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)
cor(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)
port_opt_view(ce::DenoiseCovariance, i, args...)
```
