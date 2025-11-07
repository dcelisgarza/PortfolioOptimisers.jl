# Variance

```@docs
QuadRiskExpr
SquaredSOCRiskExpr
RSOCRiskExpr
SOCRiskExpr
Variance
StandardDeviation
UncertaintySetVariance
factory(r::Variance, prior::PortfolioOptimisers.AbstractPriorResult, args...; kwargs...)
factory(r::StandardDeviation, prior::PortfolioOptimisers.AbstractPriorResult, args...; kwargs...)
factory(r::UncertaintySetVariance, prior::PortfolioOptimisers.AbstractPriorResult, ::Any,
                 ucs::Option{<:Union{<:PortfolioOptimisers.AbstractUncertaintySetResult,
                            <:PortfolioOptimisers.AbstractUncertaintySetEstimator}} = nothing, args...;
                 kwargs...)
PortfolioOptimisers.SecondMomentFormulation
PortfolioOptimisers.VarianceFormulation
```
