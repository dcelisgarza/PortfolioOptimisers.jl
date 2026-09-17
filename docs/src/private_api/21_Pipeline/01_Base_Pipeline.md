```@meta
Description = "Base Pipeline, private API of PortfolioOptimisers.jl: AbstractPipelineEstimator, AbstractPipelineResult, PIPELINE_DATA_SLOTS, PIPELINE_SLOTS, …"
```

# Base Pipeline: private API

The preprocessing estimator and result hierarchies, the price-level [`PricesResult`](@ref) container, and their fit/apply verbs are not pipeline concepts — they are documented under [Preprocessing](../03_InputData/04_Preprocessing.md).

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
pipe_constraint_targets
```
