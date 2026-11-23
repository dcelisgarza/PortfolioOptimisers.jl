```@meta
Description = "Base Cross Validation (b), private API of PortfolioOptimisers.jl: Fold, fit_and_predict, reconstruct_rd, sort_predictions, cv_sequential_info, …"
```

# Base Cross Validation (b): private API

```@docs
Fold
fit_and_predict
reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult, X::VecNum)
sort_predictions(test_idx::VecVecInt, predictions::VecPredRes)
cv_sequential_info
parallel_folds
run_folds
advance_previous_fold
FoldEvaluation
fold_evaluation
folds_are_stepped
fold_loop
assert_unshuffled_folds
assert_prediction_series
```
