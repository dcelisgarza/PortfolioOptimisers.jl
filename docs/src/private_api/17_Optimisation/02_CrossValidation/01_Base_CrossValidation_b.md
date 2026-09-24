```@meta
Description = "Base Cross Validation (b), private API of PortfolioOptimisers.jl: Fold, fit_and_predict, sort_predictions!, cv_sequential_info, parallel_folds, run_folds, …"
```

# Base Cross Validation (b): private API

```@docs
Fold
fit_and_predict
sort_predictions!(test_idx::VecVecInt, predictions::VecPredRes)
cv_sequential_info
parallel_folds
run_folds
advance_previous_fold
folds_are_time_ordered
fold_evaluation
folds_are_stepped
fold_loop
assert_unshuffled_folds
```
