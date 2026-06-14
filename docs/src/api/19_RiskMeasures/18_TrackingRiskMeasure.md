# Tracking risk measure

```@docs
RiskTrackingError
port_opt_view(::Nothing, ::Any)
port_opt_view(tr::RiskTrackingError, i, X::MatNum, args...)
factory(tr::RiskTrackingError, pr::AbstractPriorResult, slv::Any, ucs::Any,
                 w::Option{<:VecNum} = nothing, args...; kwargs...)
needs_previous_weights(tr::RiskTrackingError)
factory(tr::RiskTrackingError, w::VecNum)
TrackingRiskMeasure
port_opt_view(r::TrackingRiskMeasure, i, args...)
needs_previous_weights(r::TrackingRiskMeasure)
factory(r::TrackingRiskMeasure, w::VecNum)
factory(r::TrackingRiskMeasure, ::Any, ::Any, ::Any, w::VecNum, args...; kwargs...)
RiskTrackingRiskMeasure
port_opt_view(r::RiskTrackingRiskMeasure, i, X::MatNum, args...)
needs_previous_weights(r::RiskTrackingRiskMeasure)
factory(r::RiskTrackingRiskMeasure, w::VecNum)
factory(r::RiskTrackingRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
TrRM
supports_precomputed_returns(::TrackingRiskMeasure{<:WeightsTracking})
supports_precomputed_returns(::TrackingRiskMeasure{<:ReturnsTracking})
```
