```@meta
Description = "Base Pipeline, public API of PortfolioOptimisers.jl: PipelineStep, pipe_constraint_targets."
```

# Base Pipeline

A pipeline holds a whole workflow as a list of steps that run from left to right. The steps can be price preprocessing, the conversion of prices to returns, returns preprocessing, prior estimation, phylogeny, uncertainty sets, constraint generation and optimisation. Each step can read what the steps before it computed. Cross-validation and hyperparameter tuning then cover the whole workflow, data preparation included, not the optimiser alone.

```@docs
PipelineStep
pipe_constraint_targets
```
