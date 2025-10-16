# Covariance

```@docs
GeneralCovariance
cov(ce::GeneralCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GeneralCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
Covariance
cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
```
