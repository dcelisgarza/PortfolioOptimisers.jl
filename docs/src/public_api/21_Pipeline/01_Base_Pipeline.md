```@meta
Description = "Base Pipeline, public API of PortfolioOptimisers.jl: PipelineStep."
```

# Base Pipeline

A pipeline reifies an end-to-end workflow — price preprocessing, prices-to-returns conversion, returns preprocessing, prior estimation, phylogeny, uncertainty sets, constraint generation, and optimisation — as an ordered list of steps executed left-to-right over an accumulating context. Pipelines widen the cross-validation and hyperparameter-tuning boundary to the entire workflow, data preparation included. See `docs/adr/0028-pipeline-workflow-estimator.md` for the design rationale.

```@docs
PipelineStep
```
