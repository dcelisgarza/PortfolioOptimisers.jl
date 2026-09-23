```@meta
Description = "The online arm of the fold loop, private API of PortfolioOptimisers.jl: fit_fold_result, thread_online_folds!, online_folds, online_step_fold, step_context."
```

# The online arm of the fold loop: private API

A scheme built by [`OnlineIndexWalkForward`](@ref), [`OnlineDateWalkForward`](@ref) or [`OnlineHindsightSplit`](@ref) is an online scheme. With an online scheme, [`PortfolioOptimisers.fold_loop`](@ref) does not refit the estimator on each training window. It fits one estimator on the first training window. At each later fold it adds to that estimator only the rows that the training window gained, by default with [`partial_fit!`](@ref).

The loop then gives the callback a [`PortfolioOptimisers.Fold`](@ref) whose `train` field is `nothing`, because the estimator already holds the training rows. Where a batch fold refits, this fold calls `optimise(opt)` with no returns. At fold `i` the estimator holds every row up to the end of the training window. The batch expanding-window fold reads the same rows. The weights of each fold are therefore those of the batch walk-forward, to the tolerance of the moment estimates and of the solver.

```@docs
PortfolioOptimisers.fit_fold_result
PortfolioOptimisers.thread_online_folds!
PortfolioOptimisers.online_folds
PortfolioOptimisers.online_step_fold(::Any, ::Nothing, ::ReturnsResult)
PortfolioOptimisers.step_context
```
