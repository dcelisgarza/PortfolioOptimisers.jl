```@meta
Description = "Variance, public API of PortfolioOptimisers.jl: QuadRiskExpr, SquaredSOCRiskExpr, RSOCRiskExpr, SOCRiskExpr, RatioFormulation, SquareRootRatio, LinearRatio, …"
```

# Variance

```@docs
QuadRiskExpr
SquaredSOCRiskExpr
RSOCRiskExpr
SOCRiskExpr
RatioFormulation
SquareRootRatio
LinearRatio
Variance
StandardDeviation
UncertaintySetVariance
factory(r::Variance, pr::AbstractPriorResult, args...; kwargs...)
factory(r::StandardDeviation, pr::AbstractPriorResult, args...; kwargs...)
factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ::Any, ucs::Option{<:UcSE_UcS} = nothing, args...; kwargs...)
factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ucs::Option{<:UcSE_UcS} = nothing; kwargs...)
factory(r::UncertaintySetVariance, ucs::UcSE_UcS, pr::Option{<:AbstractPriorResult} = nothing; kwargs...)
obs_weights_view(r::UncertaintySetVariance, i)
no_bounds_risk_measure(r::UncertaintySetVariance, ::Union{Val{true}, Nothing})
no_bounds_no_risk_expr_risk_measure(r::UncertaintySetVariance, ::Union{Val{true}, Nothing})
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
