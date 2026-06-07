# Base Risk Measures

```@docs
RiskMeasure
HierarchicalRiskMeasure
Frontier
RiskMeasureSettings
HierarchicalRiskMeasureSettings
SumScalariser
MaxScalariser
MinScalariser
LogSumExpScalariser
AbstractBaseRiskMeasure
NonOptimisationRiskMeasure
OptimisationRiskMeasure
AbstractRiskMeasureSettings
JuMPRiskMeasureSettings
FrontierBoundEstimator
LinearBound
SquareRootBound
SquaredBound
Scalariser
NonHierarchicalScalariser
HierarchicalScalariser
nothing_scalar_array_selector
risk_measure_nothing_scalar_array_view
solver_selector
VecBaseRM
VecOptRM
OptRM_VecOptRM
VecRM
RM_VecRM
RkRtBounds
Front_NumVec
bigger_is_better
needs_previous_weights(::AbstractBaseRiskMeasure)
factory(rs::AbstractBaseRiskMeasure, args...; kwargs...)
factory(rs::VecBaseRM, args...; kwargs...)
risk_measure_view(rs::AbstractBaseRiskMeasure, ::Any, ::Any)
_Frontier
```
