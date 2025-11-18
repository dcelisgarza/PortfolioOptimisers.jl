# Expected Returns

## Public

```@docs
ReturnRiskMeasure
RatioRiskMeasure
```

## Private

```@docs
expected_return
expected_ratio
expected_risk_ret_ratio
expected_sric
expected_risk_ret_sric
factory(r::ReturnRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
factory(r::ReturnRiskMeasure, args...; kwargs...)
factory(r::RatioRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
factory(r::RatioRiskMeasure{<:Any, <:Union{<:RiskTrackingRiskMeasure, <:TrackingRiskMeasure}}, w::VecNum)
```
