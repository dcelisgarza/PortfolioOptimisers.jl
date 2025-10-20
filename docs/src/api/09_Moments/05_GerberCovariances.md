# Gerber Covariance

```@docs
Gerber0
Gerber1
Gerber2
StandardisedGerber0
StandardisedGerber1
StandardisedGerber2
GerberCovariance
cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:PortfolioOptimisers.UntandardisedGerberCovarianceAlgorithm}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:PortfolioOptimisers.UntandardisedGerberCovarianceAlgorithm}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
PortfolioOptimisers.BaseGerberCovariance
PortfolioOptimisers.GerberCovarianceAlgorithm
PortfolioOptimisers.UntandardisedGerberCovarianceAlgorithm
PortfolioOptimisers.StandardisedGerberCovarianceAlgorithm
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix,
                std_vec::AbstractArray)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber0},
                X::AbstractMatrix)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix,
                std_vec::AbstractArray)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber1},
                X::AbstractMatrix)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix,
                std_vec::AbstractArray)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber2},
                X::AbstractMatrix)
```
