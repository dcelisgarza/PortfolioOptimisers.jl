# Base Pipeline

A pipeline reifies an end-to-end workflow — price preprocessing, prices-to-returns conversion, returns preprocessing, prior estimation, phylogeny, uncertainty sets, constraint generation, and optimisation — as an ordered list of steps executed left-to-right over an accumulating context. Pipelines widen the cross-validation and hyperparameter-tuning boundary to the entire workflow, data preparation included. See `docs/adr/0028-pipeline-workflow-estimator.md` for the design rationale.

## Abstract types

```@docs
AbstractPipelineEstimator
AbstractPipelineResult
```

The preprocessing estimator and result hierarchies, the price-level [`PricesResult`](@ref) container, and their fit/apply verbs are not pipeline concepts — they are documented under [Pre-processing](../03_Preprocessing.md).

## Context and slots

```@docs
PipelineContext
PIPELINE_SLOTS
pipe_reads
pipe_writes
PipelineStep
PipelineUncertaintySets
```
