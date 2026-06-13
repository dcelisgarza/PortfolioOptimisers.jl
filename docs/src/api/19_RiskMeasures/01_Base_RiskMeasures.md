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
RiskInputKind
NetReturnsInput
WeightsReturnsFeesInput
WeightsInput
risk_input_kind
supports_precomputed_returns(r::AbstractBaseRiskMeasure)
supports_precomputed_returns(::NetReturnsInput, ::Any)
supports_precomputed_returns(::WeightsInput, ::Any)
supports_precomputed_returns(::WeightsReturnsFeesInput, r::AbstractBaseRiskMeasure)
weight_independent_target(::Nothing)
weight_independent_target(::Number)
weight_independent_target(::Any)
factory(rs::AbstractBaseRiskMeasure, args...; kwargs...)
factory(rs::VecBaseRM, args...; kwargs...)
port_opt_view(rs::AbstractBaseRiskMeasure, ::Any, ::Any)
_Frontier
```
