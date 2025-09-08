# Covariance

```@docs
GeneralWeightedCovariance
cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
Covariance
cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
```
