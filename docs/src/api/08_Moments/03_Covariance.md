# Covariance

```@docs
GeneralCovariance
cov(ce::GeneralCovariance, X::NumMat; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GeneralCovariance, X::NumMat; dims::Int = 1, mean = nothing, kwargs...)
Covariance
cov(ce::Covariance{<:Any, <:Any, <:Full}, X::NumMat; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Full}, X::NumMat; dims::Int = 1, mean = nothing, kwargs...)
```
