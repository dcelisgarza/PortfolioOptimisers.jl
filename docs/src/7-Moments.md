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
```
