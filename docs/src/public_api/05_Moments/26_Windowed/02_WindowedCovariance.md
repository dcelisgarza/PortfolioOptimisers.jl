```@meta
Description = "Windowed covariance, public API of PortfolioOptimisers.jl: WindowedCovariance, cov, cor."
```

# [Windowed covariance](@id api-windowed-covariance)

```@docs
WindowedCovariance
cov(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing, kwargs...)
cor(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing, kwargs...)
```
