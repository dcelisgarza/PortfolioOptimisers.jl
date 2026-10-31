```@meta
Description = "Ratio Risk Measure, public API of PortfolioOptimisers.jl: RiskRatio, NonOptimisationRiskRatio, factory, bigger_is_better."
```

# [Ratio Risk Measure](@id api-ratio-risk-measure)

```@docs
RiskRatio
NonOptimisationRiskRatio
factory(r::RiskRatio, args...; kwargs...)
factory(r::RiskRatio, w::VecNum)
factory(r::NonOptimisationRiskRatio, args...; kwargs...)
bigger_is_better(r::Union{RiskRatio, NonOptimisationRiskRatio})
```
