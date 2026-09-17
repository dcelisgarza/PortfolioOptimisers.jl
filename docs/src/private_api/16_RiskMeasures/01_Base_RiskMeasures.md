```@meta
Description = "Base Risk Measures, private API of PortfolioOptimisers.jl: AbstractBaseRiskMeasure, NonOptimisationRiskMeasure, OptimisationRiskMeasure, …"
```

# Base Risk Measures: private API

```@docs
AbstractBaseRiskMeasure
NonOptimisationRiskMeasure
OptimisationRiskMeasure
AbstractRiskMeasureSettings
JuMPRiskMeasureSettings
FrontierBoundEstimator
Scalariser
NonHierarchicalScalariser
HierarchicalScalariser
DeferredQuantity
MuSlot
SigmaSlot
KtSlot
SkSlot
VecBaseRM
BaseRM_VecBaseRM
VecOptRM
OptRM_VecOptRM
VecRM
RM_VecRM
RkRtBounds
Front_NumVec
RiskInputKind
NetReturnsInput
WeightsReturnsFeesInput
WeightsInput
scalarise
scalarise_combine
scalarise_map
scalarise_logsumexp
nothing_scalar_array_selector
deferred_factors
fit_deferred_quantity
coskewness_processor
deferred_centre
centring_target
fit_deferred_moment
deferred_quantity
fan_out_slot
deferred_derived_quantity
resolve_slot
deferred_slots
functor_slots
resolve_calibration_slots(x, pr::AbstractPriorResult)
resolve_deferred_quantities(x, pr::AbstractPriorResult)
resolve_deferred_child
rebuild_with_slots
assert_declared_slot_resolver
assert_resolved_slots
sigma_chol_selector
assert_derived_slot_has_source
risk_measure_nothing_scalar_array_view
solver_selector
bigger_is_better
needs_previous_weights(::AbstractBaseRiskMeasure)
risk_input_kind
range_tails
supports_precomputed_returns(r::AbstractBaseRiskMeasure)
supports_precomputed_returns(rs::VecBaseRM)
supports_precomputed_returns(::NetReturnsInput, ::Any)
supports_precomputed_returns(::WeightsInput, ::Any)
supports_precomputed_returns(::WeightsReturnsFeesInput, r::AbstractBaseRiskMeasure)
weight_independent_target(::Nothing)
weight_independent_target(::Number)
weight_independent_target(::Any)
_Frontier
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
