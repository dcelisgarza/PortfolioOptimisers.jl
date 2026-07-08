# Base Pipeline

A pipeline reifies an end-to-end workflow — price preprocessing, prices-to-returns conversion, returns preprocessing, prior estimation, phylogeny, uncertainty sets, constraint generation, and optimisation — as an ordered list of steps executed left-to-right over an accumulating context. Pipelines widen the cross-validation and hyperparameter-tuning boundary to the entire workflow, data preparation included. See `docs/adr/0028-pipeline-workflow-estimator.md` for the design rationale.

## Abstract types

```@docs
AbstractPipelineEstimator
AbstractPipelineResult
AbstractPreprocessingEstimator
AbstractPricesPreprocessingEstimator
AbstractReturnsPreprocessingEstimator
AbstractPreprocessingResult
AbstractPricesResult
```

## Price-level data

```@docs
PricesResult
prices_view
```

## Context and slots

```@docs
PipelineContext
PIPELINE_SLOTS
pipe_reads
pipe_writes
PipelineStep
PipelineUncertaintySets
```
