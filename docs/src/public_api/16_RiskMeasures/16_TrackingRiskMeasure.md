```@meta
Description = "Tracking risk measure, public API of PortfolioOptimisers.jl: RiskTrackingError, TrackingRiskMeasure, RiskTrackingRiskMeasure, port_opt_view, factory."
```

# Tracking risk measure

```@docs
RiskTrackingError
TrackingRiskMeasure
RiskTrackingRiskMeasure
port_opt_view(::Nothing, ::Any)
port_opt_view(tr::RiskTrackingError, i, X::MatNum, args...)
factory(tr::RiskTrackingError, pr::AbstractPriorResult, slv::Any, ucs::Any, w::Option{<:VecNum} = nothing, args...; kwargs...)
factory(tr::RiskTrackingError, w::VecNum)
factory(r::TrackingRiskMeasure, w::VecNum)
factory(r::TrackingRiskMeasure, ::Any, ::Any, ::Any, w::VecNum, args...; kwargs...)
port_opt_view(r::RiskTrackingRiskMeasure, i, X::MatNum, args...)
factory(r::RiskTrackingRiskMeasure, w::VecNum)
factory(r::RiskTrackingRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
