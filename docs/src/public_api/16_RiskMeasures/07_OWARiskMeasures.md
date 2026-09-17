```@meta
Description = "Ordered Weights Array, public API of PortfolioOptimisers.jl: AbstractOrderedWeightsArrayFunction, ExponentialConeEntropy, RelativeEntropy, MaximumEntropy, …"
```

# Ordered Weights Array

Certain risk measures can be expressed as ordered weights arrays [owa1,owa3](@cite). It is also possible to express higher Linear moments (l-moments) as linear combinations of ordered weights arrays [owa2](@cite).

These types and functions implement the various existing formulations and risk measures representable by ordered weights arrays.

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
OrderedWeightsArrayRange
OWAJuMP
factory(x::OrderedWeightsArray, pr::AbstractPriorResult, args...; kwargs...)
factory(x::OrderedWeightsArrayRange, pr::AbstractPriorResult, args...; kwargs...)
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
