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
                        <:PortfolioOptimisers.UnstandardisedGerberCovarianceAlgorithm}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:PortfolioOptimisers.UnstandardisedGerberCovarianceAlgorithm}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
PortfolioOptimisers.BaseGerberCovariance
PortfolioOptimisers.GerberCovarianceAlgorithm
PortfolioOptimisers.UnstandardisedGerberCovarianceAlgorithm
PortfolioOptimisers.StandardisedGerberCovarianceAlgorithm
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::MatNum,
                std_vec::ArrNum)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber0},
                X::MatNum)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::MatNum,
                std_vec::ArrNum)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber1},
                X::MatNum)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::MatNum,
                std_vec::ArrNum)
PortfolioOptimisers.gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber2},
                X::MatNum)
```
