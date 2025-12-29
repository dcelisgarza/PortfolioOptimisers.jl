# Covariance

```@docs
GeneralCovariance
cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
Covariance
cov(ce::Covariance{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
```
