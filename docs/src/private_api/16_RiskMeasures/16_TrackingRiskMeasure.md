```@meta
Description = "Tracking risk measure, private API of PortfolioOptimisers.jl: supports_precomputed_returns, bind_data_benchmark, reader_frames."
```

# Tracking risk measure: private API

```@docs
supports_precomputed_returns(::TrackingRiskMeasure{<:Any, <:WeightsTracking})
supports_precomputed_returns(::TrackingRiskMeasure{<:Any, <:ReturnsTracking})
supports_precomputed_returns(::TrackingRiskMeasure{<:Any, <:DataBenchmarkTracking})
supports_precomputed_returns(::RiskTrackingRiskMeasure)
PortfolioOptimisers.bind_data_benchmark(::TrackingRiskMeasure, ::Any)
PortfolioOptimisers.reader_frames(::Union{<:RiskTrackingError, <:RiskTrackingRiskMeasure}, ::AbstractString, ::Bool)
```
