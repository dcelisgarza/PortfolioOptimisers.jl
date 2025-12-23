# Variance

```@docs
QuadRiskExpr
SquaredSOCRiskExpr
RSOCRiskExpr
SOCRiskExpr
Variance
StandardDeviation
UncertaintySetVariance
factory(r::Variance, pr::AbstractPriorResult, args...; kwargs...)
factory(r::StandardDeviation, pr::AbstractPriorResult, args...; kwargs...)
factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ::Any,
                 ucs::Option{<:UcSE_UcS} = nothing, args...;
                 kwargs...)
SecondMomentFormulation
VarianceFormulation
```
