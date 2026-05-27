# Turnover risk measure

```@docs
TurnoverRiskMeasure
risk_measure_view(r::TurnoverRiskMeasure, i, args...)
needs_previous_weights(r::TurnoverRiskMeasure)
factory(r::TurnoverRiskMeasure, w::VecNum)
factory(r::TurnoverRiskMeasure, ::Any, ::Any, ::Any, w::Option{<:VecNum} = nothing, args...; kwargs...)
```
