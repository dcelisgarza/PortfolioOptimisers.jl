# Shrunk Expected Returns

```@docs
GrandMean
VolatilityWeighted
MeanSquaredError
JamesStein
BayesStein
BodnarOkhrinParolya
ShrunkExpectedReturns
mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::MatNum; dims::Int = 1, kwargs...)
mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::MatNum; dims::Int = 1, kwargs...)
mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}, X::MatNum; dims::Int = 1, kwargs...)
port_opt_view(me::ShrunkExpectedReturns, i)
AbstractShrunkExpectedReturnsEstimator
AbstractShrunkExpectedReturnsAlgorithm
AbstractShrunkExpectedReturnsTarget
target_mean
```
