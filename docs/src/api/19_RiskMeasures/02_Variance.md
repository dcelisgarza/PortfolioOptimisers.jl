# Variance

```@docs
QuadRiskExpr
SquaredSOCRiskExpr
RSOCRiskExpr
SOCRiskExpr
Variance
StandardDeviation
UncertaintySetVariance
factory(r::Variance, prior::AbstractPriorResult, args...; kwargs...)
factory(r::StandardDeviation, prior::AbstractPriorResult, args...; kwargs...)
factory(r::UncertaintySetVariance, prior::AbstractPriorResult, ::Any,
                 ucs::Option{<:UcSE_UcS} = nothing, args...;
                 kwargs...)
SecondMomentFormulation
VarianceFormulation
```
