# Moments

## Base Moments

```@docs
Full
Semi
PortfolioOptimisers.AbstractExpectedReturnsEstimator
PortfolioOptimisers.AbstractExpectedReturnsAlgorithm
PortfolioOptimisers.AbstractCovarianceEstimator
PortfolioOptimisers.AbstractMomentAlgorithm
PortfolioOptimisers.AbstractVarianceEstimator
PortfolioOptimisers.robust_cov
PortfolioOptimisers.robust_cor
```

## Mean

```@docs
SimpleExpectedReturns
SimpleExpectedReturns()
mean(me::SimpleExpectedReturns, X::AbstractArray; dims::Int = 1, kwargs...)
```

## Covariance and Correlation

```@docs
GeneralWeightedCovariance
GeneralWeightedCovariance()
cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
Covariance
Covariance()
cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
```

## Variance and Standard Deviation

```@docs
SimpleVariance
SimpleVariance()
std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)
std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)
var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)
var(ve::SimpleVariance, X::AbstractVector; mean = nothing)
```
