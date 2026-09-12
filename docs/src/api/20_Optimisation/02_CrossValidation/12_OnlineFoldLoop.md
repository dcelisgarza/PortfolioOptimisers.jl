# The online arm of the fold loop

A walk-forward that declares a Fold Fit of [`OnlineStep`](@ref) sends [`PortfolioOptimisers.fold_loop`](@ref) down a third arm. It warms one estimator up on the first training window, folds each fold's new observations into it, and hands the callback a [`PortfolioOptimisers.Fold`](@ref) whose `train` is `nothing` — *the estimator holds its window* — so the fold reads the estimator out through `optimise(opt)` where a refit would have run. The run reaches the weights of the batch expanding-window walk-forward fold for fold. ADR 0140 records the decision.

```@docs
PortfolioOptimisers.fit_fold_result
PortfolioOptimisers.cv_online_info
PortfolioOptimisers.thread_online_folds!
PortfolioOptimisers.online_folds
```
