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
PortfolioOptimisers.BaseGerberCovariance
PortfolioOptimisers.GerberCovarianceAlgorithm
PortfolioOptimisers.UnNormalisedGerberCovarianceAlgorithm
PortfolioOptimisers.NormalisedGerberCovarianceAlgorithm
Gerber0
Gerber1
Gerber2
NormalisedGerber0
NormalisedGerber0()
NormalisedGerber1
NormalisedGerber1()
NormalisedGerber2
NormalisedGerber2()
GerberCovariance
GerberCovariance()
cov(ce::GerberCovariance{<:Any, <:Any, <:Any, <:PortfolioOptimisers.UnNormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cov(ce::GerberCovariance{<:Any, <:Any, <:Any, <:PortfolioOptimisers.NormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GerberCovariance{<:Any, <:Any, <:Any, <:PortfolioOptimisers.UnNormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GerberCovariance{<:Any, <:Any, <:Any, <:PortfolioOptimisers.NormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix,
                std_vec::AbstractArray)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0},
                X::AbstractMatrix)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix,
                std_vec::AbstractArray)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1},
                X::AbstractMatrix)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix,
                std_vec::AbstractArray)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2},
                X::AbstractMatrix)
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
