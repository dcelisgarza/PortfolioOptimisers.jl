```@meta
Description = "Turnover risk measure, public API of PortfolioOptimisers.jl: TurnoverRiskMeasure, port_opt_view, factory, needs_previous_weights."
```

# [Turnover risk measure](@id api-turnover-risk-measure)

```@docs
TurnoverRiskMeasure
port_opt_view(r::TurnoverRiskMeasure, i, args...)
factory(r::TurnoverRiskMeasure, w::VecNum)
factory(r::TurnoverRiskMeasure, ::Any, ::Any, ::Any, w::Option{<:VecNum} = nothing, args...; kwargs...)
needs_previous_weights(r::TurnoverRiskMeasure)
```
