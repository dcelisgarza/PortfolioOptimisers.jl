# Gerber covariance

The Gerber statistic is a vote-based robust co-movement measure. It ignores fluctuations below a threshold while limiting the effect of extreme movements. It extends Kendall's Tau coefficient by counting the proportion of concordant and discordant movements within the window defined by the upper and lower limits [gerber](@cite).

Three variants have been published and all three have been implemented because each has unique characteristics [gerber_analysis](@cite). We have also implemented extensions which Z-normalise the data and thus treat the thresholds as relative rather than absolute values.

## Abstract Gerber covariance types

These serve as the scaffolding for defining Gerber covariance estimators and algorithms.

```@docs
BaseGerberCovariance
GerberCovarianceAlgorithm
UnstandardisedGerberCovarianceAlgorithm
StandardisedGerberCovarianceAlgorithm
```

## Concrete Gerber covariance implementations

These define the concrete implementations of the Gerber covariance estimators and algorithms.

```@docs
Gerber0
Gerber1
Gerber2
StandardisedGerber0
StandardisedGerber1
StandardisedGerber2
GerberCovariance
factory(ce::GerberCovariance, w::StatsBase.AbstractWeights)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::MatNum, sd::ArrNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber0}, X::MatNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::MatNum, sd::ArrNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber1}, X::MatNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::MatNum, sd::ArrNum)
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:StandardisedGerber2}, X::MatNum)
cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:UnstandardisedGerberCovarianceAlgorithm}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                        <:UnstandardisedGerberCovarianceAlgorithm}, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
```
