```@meta
Description = "Tracking, public API of PortfolioOptimisers.jl: AbstractTrackingAlgorithm, IndependentVariableTracking, DependentVariableTracking, WeightsTracking, …"
```

# Tracking

```@docs
AbstractTrackingAlgorithm
IndependentVariableTracking
DependentVariableTracking
WeightsTracking
ReturnsTracking
TrackingError
factory(tr::WeightsTracking, w::VecNum)
tracking_benchmark
needs_previous_weights(tr::TrackingError)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
