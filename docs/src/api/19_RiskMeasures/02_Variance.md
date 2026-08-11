# Variance

```@docs
QuadRiskExpr
SquaredSOCRiskExpr
RSOCRiskExpr
SOCRiskExpr
NSkeQuadFormulations
QuadSecondMomentFormulations
Variance
StandardDeviation
UncertaintySetVariance
resolve_deferred_quantities(r::Variance, pr::AbstractPriorResult)
resolve_deferred_quantities(r::StandardDeviation, pr::AbstractPriorResult)
resolve_deferred_quantities(r::UncertaintySetVariance, pr::AbstractPriorResult)
factory(r::Variance, pr::AbstractPriorResult, args...; kwargs...)
factory(r::StandardDeviation, pr::AbstractPriorResult, args...; kwargs...)
factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ::Any,
                 ucs::Option{<:UcSE_UcS} = nothing, args...;
                 kwargs...)
factory(r::UncertaintySetVariance, pr::AbstractPriorResult,
                 ucs::Option{<:UcSE_UcS} = nothing; kwargs...)
factory(r::UncertaintySetVariance, ucs::UcSE_UcS,
                 pr::Option{<:AbstractPriorResult} = nothing; kwargs...)
SecondMomentFormulation
VarianceFormulation
_no_bounds_risk_measure(r::UncertaintySetVariance, ::Union{Val{true}, Nothing})
_no_bounds_no_risk_expr_risk_measure(r::UncertaintySetVariance, ::Union{Val{true}, Nothing})
ucs_risk_measure
ucs_variance
```
