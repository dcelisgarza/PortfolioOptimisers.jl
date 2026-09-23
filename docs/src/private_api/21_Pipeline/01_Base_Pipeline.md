```@meta
Description = "Base Pipeline, private API of PortfolioOptimisers.jl: AbstractPipelineEstimator, AbstractPipelineResult, PIPELINE_DATA_SLOTS, PIPELINE_SLOTS, …"
```

# Base Pipeline: private API

The abstract types of the preprocessing estimators and of their results, the price-level result [`PricesResult`](@ref), and the functions that fit and apply preprocessing are on the [preprocessing](../03_InputData/04_Preprocessing.md) page. They work without a pipeline, so this page does not document them.

```@docs
AbstractPipelineEstimator
AbstractPipelineResult
PIPELINE_DATA_SLOTS
PIPELINE_SLOTS
PIPELINE_INVALIDATES
PIPELINE_ROUTING_TARGETS
PIPELINE_OPTIONAL_TARGETS
PIPELINE_ACCUMULATING_TARGETS
PIPELINE_STEP_TARGETS
PIPELINE_THRESHOLD_TARGETS
PIPELINE_ASSET_SETS_MATRIX_TARGETS
TD_OptE_Opt_Inferable
PipelineContext
TargetedConstraint
PipelineUncertaintySets
unroutable_target
assert_opt_last
pipe_reads
pipe_writes
```
