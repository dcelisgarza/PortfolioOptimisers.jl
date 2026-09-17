```@meta
Description = "Variance, public API of PortfolioOptimisers.jl: QuadRiskExpr, SquaredSOCRiskExpr, RSOCRiskExpr, SOCRiskExpr, Variance, StandardDeviation, …"
```

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
factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ::Any, ucs::Option{<:UcSE_UcS} = nothing, args...; kwargs...)
factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ucs::Option{<:UcSE_UcS} = nothing; kwargs...)
factory(r::UncertaintySetVariance, ucs::UcSE_UcS, pr::Option{<:AbstractPriorResult} = nothing; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
