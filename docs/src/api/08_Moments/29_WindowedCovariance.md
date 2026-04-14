# Windowed covariance

```@docs
WindowedCovariance
factory(ce::WindowedCovariance, w::ObsWeights)
cov(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
cor(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
```
