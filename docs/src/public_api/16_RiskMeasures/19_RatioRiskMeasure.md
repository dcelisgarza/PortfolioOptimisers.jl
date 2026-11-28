```@meta
Description = "Ratio Risk Measure, public API of PortfolioOptimisers.jl: RiskRatio, NonOptimisationRiskRatio, factory, bigger_is_better, needs_previous_weights, …"
```

# [Ratio Risk Measure](@id api-ratio-risk-measure)

```@docs
RiskRatio
NonOptimisationRiskRatio
factory(r::RiskRatio, args...; kwargs...)
factory(r::RiskRatio, w::VecNum)
factory(r::NonOptimisationRiskRatio, args...; kwargs...)
bigger_is_better(r::Union{RiskRatio, NonOptimisationRiskRatio})
needs_previous_weights(r::RkRatioRM)
supports_precomputed_returns(r::RkRatioRM)
```
