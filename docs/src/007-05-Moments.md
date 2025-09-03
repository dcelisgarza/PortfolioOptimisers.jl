# Gerber Covariance

```@docs
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
