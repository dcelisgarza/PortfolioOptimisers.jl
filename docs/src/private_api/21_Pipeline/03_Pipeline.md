```@meta
Description = "PortfolioOptimisers pipeline, private API of PortfolioOptimisers.jl: first_duplicate, inject_context, constraint_results, implicit_constraint_target, …"
```

# PortfolioOptimisers pipeline: private API

```@docs
first_duplicate
```

## Injection

The pipeline resolves its computed slots into [routing targets](@ref PIPELINE_ROUTING_TARGETS) and hands each one to the optimiser, which owns the decision of where it lands. See [`pipe_route`](@ref) for the optimiser-owned half of the seam.

```@docs
inject_context
constraint_results
implicit_constraint_target
constraint_target_of
constraint_value_of
accumulate_constraint_values
constraint_targets
maybe_inject_step
pipe_required_targets
assert_routable
assert_constraint_targets
```

## Prediction

Predicting with a fitted pipeline replays the fitted preprocessing steps — the training universe, the training imputation parameters, the returns conversion — on an unseen data window, then delegates to the existing weights-level prediction machinery. Cross-validation folds can be computed directly on price-level data ([`Prices_RR`](@ref)), so the whole workflow is fitted per fold with no test-window leakage into stateful preprocessing.

```@docs
apply_fitted_step
apply_fitted_steps
assert_universe_aligned
```
