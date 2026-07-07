# Preprocessing steps

Preprocessing estimators transform price or returns data inside a pipeline under a fit/apply contract: fitting on a training window produces a result carrying any fitted state — imputation parameters, thresholds, and the selected asset universe — which is then applied to unseen windows so train and test data are transformed consistently, with no information flowing from test to train.

## Step execution contract

```@docs
run_step
fit_step
apply_step
require_slot
set_slot
is_missing_value
run_uncertainty_step
pipeline_asset_sets
add_constraint_result
```

## Preprocessing estimators

```@docs
PricesToReturns
MissingDataFilter
MissingDataFilterResult
Imputer
ImputerResult
```
