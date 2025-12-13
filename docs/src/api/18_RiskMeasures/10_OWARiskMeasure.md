# Ordered Weights Array

Certain risk measures can be expressed as ordered weights arrays [owa1,owa3](@cite). It is also possible to express higher Linear moments (l-moments) as linear combinations of ordered weights arrays [owa2](@cite).

These types and functions implement the various existing formulations and risk measures representable by ordered weights arrays.

```@docs
AbstractOrderedWeightsArrayEstimator
AbstractOrderedWeightsArrayAlgorithm
SquaredOrderedWeightsArrayAlgorithm
UnionAllSOCRiskExpr
UnionSOCRiskExpr
UnionRSOCSOCRiskExpr
MaximumEntropy
MinimumSquaredDistance
MinimumSumSquares
NormalisedConstantRelativeRiskAversion
OWAJuMP
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
ncrra_weights
owa_model_setup
owa_model_solve
```
