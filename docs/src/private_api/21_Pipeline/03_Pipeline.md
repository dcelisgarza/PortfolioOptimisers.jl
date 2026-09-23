```@meta
Description = "PortfolioOptimisers pipeline, private API of PortfolioOptimisers.jl: first_duplicate, inject_context, constraint_results, constraint_target_of, …"
```

# PortfolioOptimisers pipeline: private API

```@docs
first_duplicate
```

## Injection

The pipeline turns every slot it computed into [routing targets](@ref PIPELINE_ROUTING_TARGETS) and passes them to the optimiser. The optimiser chooses the field that receives the value, through [`pipe_route`](@ref).

```@docs
inject_context
constraint_results
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

To predict with a fitted pipeline, the pipeline applies its fitted preprocessing steps to the new data window. The steps keep what they learned on the training window: the universe, the imputation parameters and the conversion to returns. The pipeline then predicts with the same functions as an optimisation result. Cross-validation also accepts price data, through [`Prices_RR`](@ref). Each fold then fits the whole pipeline on its own training window, so no preprocessing step learns anything from the test window.

```@docs
apply_fitted_step
apply_fitted_steps
assert_universe_aligned
```
