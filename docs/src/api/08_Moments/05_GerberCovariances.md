# Gerber covariance

```@docs
Gerber0
Gerber1
Gerber2
StandardisedGerber0
StandardisedGerber1
StandardisedGerber2
GerberCovariance
cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:UnstandardisedGerberCovarianceAlgorithm}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:UnstandardisedGerberCovarianceAlgorithm}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
BaseGerberCovariance
GerberCovarianceAlgorithm
UnstandardisedGerberCovarianceAlgorithm
StandardisedGerberCovarianceAlgorithm
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::MatNum,
                std_vec::ArrNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber0},
                X::MatNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::MatNum,
                std_vec::ArrNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber1},
                X::MatNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::MatNum,
                std_vec::ArrNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber2},
                X::MatNum)
```
