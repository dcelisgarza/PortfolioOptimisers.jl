```@meta
Description = "Ordered Weights Array, private API of PortfolioOptimisers.jl: AbstractOrderedWeightsArrayEstimator, AbstractOrderedWeightsArrayAlgorithm, …"
```

# Ordered Weights Array: private API

```@docs
AbstractOrderedWeightsArrayEstimator
AbstractOrderedWeightsArrayAlgorithm
AbstractOrderedWeightsArrayFunction
SquaredOrderedWeightsArrayAlgorithm
UnionAllSOCRiskExpr
UnionSOCRiskExpr
UnionRSOCSOCRiskExpr
EntropyFormulation
OWA_Func_VecNum
OrderedWeightsArrayFormulation
OWA_RevFunc
OWA_CalOccupant
resolve_deferred_quantities(x::ComposedFunction{typeof(reverse), <:AbstractOrderedWeightsArrayFunction}, pr::AbstractPriorResult)
owa_l_moment_crm_sumsq_obj
ncrra_weights
owa_model_setup
owa_model_solve
owa_l_moment_crm_entropy
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
