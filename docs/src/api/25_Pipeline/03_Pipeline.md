# Pipeline

The `Pipeline` estimator reifies an end-to-end workflow — data preparation, prior estimation, phylogeny, uncertainty sets, constraint generation, and optimisation — as an ordered list of steps fitted as a single unit. Computed slots override the terminal optimiser's internal configuration; absent steps fall back to what the optimiser computes internally.

## Pipeline symbols

```@docs
Pipeline
PipelineResult
PortfolioOptimisers.fit
```

## Injection

```@docs
inject_context
inject_config
inject_sigma_ucs
constraint_results
```
