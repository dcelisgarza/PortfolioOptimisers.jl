# Tracking

## Public

```@docs
IndependentVariableTracking
DependentVariableTracking
WeightsTracking
ReturnsTracking
TrackingError
```

## Private

```@docs
AbstractTracking
AbstractTrackingAlgorithm
VecTr
Tr_VecTr
VariableTracking
tracking_benchmark
factory(tr::WeightsTracking, w::VecNum)
needs_previous_weights(tr::TrackingError)
narrow_optimiser_vector
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
