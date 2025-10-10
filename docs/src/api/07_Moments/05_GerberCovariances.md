# Gerber Covariance

```@docs
Gerber0
Gerber1
Gerber2
NormalisedGerber0
NormalisedGerber1
NormalisedGerber2
GerberCovariance
cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:PortfolioOptimisers.UnStandardisedGerberCovarianceAlgorithm}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:PortfolioOptimisers.UnStandardisedGerberCovarianceAlgorithm}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
PortfolioOptimisers.BaseGerberCovariance
PortfolioOptimisers.GerberCovarianceAlgorithm
PortfolioOptimisers.UnStandardisedGerberCovarianceAlgorithm
PortfolioOptimisers.StandardisedGerberCovarianceAlgorithm
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
