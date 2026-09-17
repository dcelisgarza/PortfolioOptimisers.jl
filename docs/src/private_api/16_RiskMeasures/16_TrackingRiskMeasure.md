```@meta
Description = "Tracking risk measure, private API of PortfolioOptimisers.jl: needs_previous_weights, supports_precomputed_returns."
```

# Tracking risk measure: private API

```@docs
needs_previous_weights(tr::RiskTrackingError)
needs_previous_weights(r::TrackingRiskMeasure)
needs_previous_weights(r::RiskTrackingRiskMeasure)
supports_precomputed_returns(::TrackingRiskMeasure{<:Any, <:WeightsTracking})
supports_precomputed_returns(::TrackingRiskMeasure{<:Any, <:ReturnsTracking})
supports_precomputed_returns(::RiskTrackingRiskMeasure)
```
