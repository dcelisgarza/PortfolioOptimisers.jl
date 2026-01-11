# [Covariance](@id api-covariance)

The classic Markowitz portfolio minimises the portfolio variance. This is computed from the covariance matrix and portfolio weights. Here we define the most basic covariance/correlation estimator.

```@docs
GeneralCovariance
cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
Covariance
cov(ce::Covariance{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
```
