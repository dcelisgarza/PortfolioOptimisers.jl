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
                        <:PortfolioOptimisers.UnstandardisedGerberCovarianceAlgorithm}, X::NumMat; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:PortfolioOptimisers.UnstandardisedGerberCovarianceAlgorithm}, X::NumMat; dims::Int = 1, mean = nothing, kwargs...)
PortfolioOptimisers.BaseGerberCovariance
PortfolioOptimisers.GerberCovarianceAlgorithm
PortfolioOptimisers.UnstandardisedGerberCovarianceAlgorithm
PortfolioOptimisers.StandardisedGerberCovarianceAlgorithm
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::NumMat,
                std_vec::NumArr)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber0},
                X::NumMat)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::NumMat,
                std_vec::NumArr)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber1},
                X::NumMat)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::NumMat,
                std_vec::NumArr)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber2},
                X::NumMat)
```
