```@meta
Description = "Ordered Weights Array (a), public API of PortfolioOptimisers.jl: AbstractOrderedWeightsArrayFunction, ExponentialConeEntropy, RelativeEntropy, …"
```

# Ordered Weights Array (a)

Several risk measures are ordered weighted averages. Such a measure sorts the portfolio returns, and weights each return by its rank [owa1,owa3](@cite). The higher L-moments are linear combinations of such averages [owa2](@cite).

The types and functions below build the weight vectors of these measures, and the formulations that optimise them.

```@docs
AbstractOrderedWeightsArrayFunction
ExponentialConeEntropy
RelativeEntropy
MaximumEntropy
MinimumSquaredDistance
MinimumSumSquares
NormalisedConstantRelativeRiskAversion
LinearMoment
OrderedWeightsArrayConditionalValueatRisk
OrderedWeightsArrayTailGini
OrderedWeightsArrayConditionalValueatRiskRange
OrderedWeightsArrayTailGiniRange
ExactOrderedWeightsArray
ApproxOrderedWeightsArray
OrderedWeightsArray
OWAJuMP
factory(x::OrderedWeightsArray, pr::AbstractPriorResult, args...; kwargs...)
owa_l_moment_crm
owa_l_moment
owa_gmd
owa_cvar
owa_wcvar
owa_tg
owa_wr
owa_rg
owa_cvarrg
owa_wcvarrg
owa_tgrg
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
