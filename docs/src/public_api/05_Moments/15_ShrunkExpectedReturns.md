```@meta
Description = "Shrunk Expected Returns, public API of PortfolioOptimisers.jl: GrandMean, VolatilityWeighted, MeanSquaredError, AbstractShrunkExpectedReturnsIntensity, …"
```

# Shrunk Expected Returns

```@docs
GrandMean
VolatilityWeighted
MeanSquaredError
AbstractShrunkExpectedReturnsIntensity
UnitIntervalIntensity
NonNegativeIntensity
UnclampedIntensity
shrinkage_intensity
JamesStein
BayesStein
BodnarOkhrinParolya
ShrunkExpectedReturns
mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::MatNum; dims::Int = 1, kwargs...)
mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::MatNum; dims::Int = 1, kwargs...)
mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}, X::MatNum; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
