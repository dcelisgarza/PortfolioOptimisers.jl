# Tracking

## Public

```@docs
SOCTracking
NOCTracking
IndependentVariableTracking
DependentVariableTracking
WeightsTracking
ReturnsTracking
TrackingError
TrackingFormulation
```

## Private

```@docs
AbstractTracking
AbstractTrackingAlgorithm
VecTr
Tr_VecTr
NormTracking
VariableTracking
norm_tracking
tracking_benchmark
tracking_view
factory(tr::WeightsTracking, w::VecNum)
factory(tr::ReturnsTracking, ::Any)
```
