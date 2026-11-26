```@meta
Description = "Tracking, public API of PortfolioOptimisers.jl: AbstractTrackingAlgorithm, IndependentVariableTracking, DependentVariableTracking, WeightsTracking, …"
```

# Tracking

```@docs
AbstractTrackingAlgorithm
IndependentVariableTracking
DependentVariableTracking
WeightsTracking
port_opt_view(tr::WeightsTracking, i::PortfolioOptimisers.InvestableIndex, args...)
ReturnsTracking
DataBenchmarkTracking
TrackingError
factory(tr::WeightsTracking, w::VecNum)
tracking_benchmark
obs_weights_view(tr::ReturnsTracking, i)
needs_previous_weights(tr::TrackingError)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
