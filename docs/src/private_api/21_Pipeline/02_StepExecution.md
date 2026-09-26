```@meta
Description = "Step execution, private API of PortfolioOptimisers.jl: run_step, require_slot, set_slot, run_uncertainty_step, uncertainty_step_source, run_constraint_step, …"
```

# Step execution: private API

A `run_step` method is the only code that connects an estimator family to a pipeline. It reads the slots of the [`PipelineContext`](@ref) that its estimator needs, and calls the function that the family uses outside a pipeline: [`prior`](@ref) for a prior estimator, [`clusterise`](@ref) for a clustering estimator, [`optimise`](@ref) for an optimiser, and [`fit_preprocessing`](@ref) and [`apply_preprocessing`](@ref) for a preprocessing estimator. It then writes the slot that the family produces.

The estimators do not depend on the pipeline code, and their docstrings are with their own families. The preprocessing estimators, for example, are on the [preprocessing](@ref private-api-preprocessing) page.

```@docs
run_step
require_slot
set_slot
run_uncertainty_step
uncertainty_step_source
run_constraint_step
resolve_constraint_target
pipeline_asset_sets
add_constraint_result
```

A [`TrainTestSplit`](@ref) is the one step whose written slot does not follow from its type. It narrows whichever data slot the pipeline input filled, the prices or the returns. [`pipe_writes`](@ref) returns `:split` for it, which is not a slot, so the pipeline treats the step as one that writes nothing. The docstrings of `run_step`, [`pipe_reads`](@ref) and [`pipe_writes`](@ref) cover the `run_step` method of the split and its slot declarations.
