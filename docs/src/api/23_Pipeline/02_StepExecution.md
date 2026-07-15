# Step execution

The step execution contract is the only pipeline-aware layer over the estimator families. Each `run_step` method reads the [`PipelineContext`](@ref) slots its estimator needs, dispatches to that family's **native verb** — [`prior`](@ref) for prior estimators, [`clusterise`](@ref) for clustering, [`optimise`](@ref) for optimisers, [`fit_preprocessing`](@ref)/[`apply_preprocessing`](@ref) for preprocessing estimators — and writes the slot the family produces.

The estimators themselves live with their own families and know nothing about pipelines; the preprocessing estimators, for instance, are documented under [Pre-processing](../03_Preprocessing.md).

```@docs
run_step
require_slot
set_slot
run_uncertainty_step
pipeline_asset_sets
add_constraint_result
```

A [`TrainTestSplit`](@ref) is the one step whose written slot is not a property of its type: it narrows whichever data slot the pipeline input filled, and declares the sentinel `:split` (see [`pipe_writes`](@ref)) so that the generic constructor machinery treats it as writing nothing. Its `run_step` method and slot declarations are documented with `run_step`, [`pipe_reads`](@ref), and [`pipe_writes`](@ref).
