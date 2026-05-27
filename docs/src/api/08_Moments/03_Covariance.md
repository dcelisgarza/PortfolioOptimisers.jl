# [Simple covariance](@id api-covariance)

The covariance is an important measure of risk used in portfolio selection and performance analysis. The classic Markowitz [markowitz1952](@cite) portfolio uses the portfolio variance as its risk measure, which is computed from the covariance matrix and portfolio weights. Here we define the most basic covariance/correlation estimator.

## General covariance

```@docs
GeneralCovariance
factory(ce::GeneralCovariance, w::ObsWeights)
cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
moment_view(ce::GeneralCovariance, i)
```

## [Covariance](@id api-covariance)

```@docs
Covariance
factory(ce::Covariance, w::ObsWeights)
cov(ce::Covariance{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
moment_view(ce::Covariance, i)
```
