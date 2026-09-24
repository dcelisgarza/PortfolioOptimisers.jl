```@meta
Description = "Base Cross Validation (a), private API of PortfolioOptimisers.jl: CrossValidationResult, CrossValidationAlgorithm, CVER, …"
```

# Base Cross Validation (a): private API

```@docs
CrossValidationResult
CrossValidationAlgorithm
CVER
OptimisationCrossValidationEstimator
SequentialCrossValidationEstimator
NonSequentialCrossValidationEstimator
OptimisationCrossValidationResult
SequentialCrossValidationResult
NonSequentialCrossValidationResult
OptCVER
NonSeqCVER
SeqCVER
NonOptimisationCrossValidationEstimator
NonOptimisationSequentialCrossValidationEstimator
NonOptimisationNonSequentialCrossValidationEstimator
NonOptimisationCrossValidationResult
NonOptimisationSequentialCrossValidationResult
NonOptimisationNonSequentialCrossValidationResult
AbstractPredictionResult
VecPredRes
PredRes_MultiPredRes
VecMPredRes
VecVecPredRes
VecPredRes_MultiPredRes
mapreduce_RetMtx(rd::AbstractVector{<:PredictionReturnsResult{<:Any, <:VecNum}}, sym = :X)
quantile_by_measure
reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult, X::VecNum)
investable_fold_view
fold_fees
threads_weights
fold_solved
held_weight_members
held_start_weights
fold_factor_returns
collapse_benchmark(B::Nothing, w::VecNum_VecVecNum, hw)
ruined_retcodes
mark_ruined_members
warn_ruined_members
cv_nobs
cv_live_assets
cv_timestamps
```
