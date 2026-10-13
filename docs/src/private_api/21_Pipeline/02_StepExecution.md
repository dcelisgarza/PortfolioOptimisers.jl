```@meta
Description = "Step execution, private API of PortfolioOptimisers.jl: run_step, run_step_under, require_slot, set_slot, run_uncertainty_step, uncertainty_step_source, …"
```

# Step execution: private API

A `run_step` method is the only code that connects an estimator family to a pipeline. It reads the slots of the [`PipelineContext`](@ref) that its estimator needs, and calls the function that the family uses outside a pipeline: [`prior`](@ref) for a prior estimator, [`clusterise`](@ref) for a clustering estimator, [`optimise`](@ref) for an optimiser, and [`fit_preprocessing`](@ref) and [`apply_preprocessing`](@ref) for a preprocessing estimator. It then writes the slot that the family produces.

The estimators do not depend on the pipeline code, and their docstrings are with their own families. The preprocessing estimators, for example, are on the [preprocessing](@ref private-api-preprocessing) page.

```@docs
run_step
run_step_under
require_slot
set_slot
run_uncertainty_step
uncertainty_step_source
run_constraint_step
resolve_constraint_target
pipeline_asset_sets
add_constraint_result
```

## The frame of a prior step

A prior step fits on the raw returns, or on the returns in excess of the benchmark when its [`PipelineStep`](@ref) sets `brt = true`. The optimiser that receives the prior must read it in the same frame, and the [`Pipeline`](@ref) constructor checks this.

```@docs
assert_step_brt
pipe_prior_brt
step_prior_brt
pipe_terminal_optimiser
assert_benchmark_frame
```

A [`TrainTestSplit`](@ref) is the one step whose written slot does not follow from its type. It narrows whichever data slot the pipeline input filled, the prices or the returns. [`pipe_writes`](@ref) returns `:split` for it, which is not a slot, so the pipeline treats the step as one that writes nothing. The docstrings of `run_step`, [`pipe_reads`](@ref) and [`pipe_writes`](@ref) cover the `run_step` method of the split and its slot declarations.
