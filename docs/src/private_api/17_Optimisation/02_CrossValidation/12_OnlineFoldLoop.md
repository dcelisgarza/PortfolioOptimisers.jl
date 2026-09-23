```@meta
Description = "The online arm of the fold loop, private API of PortfolioOptimisers.jl: fit_fold_result, thread_online_folds!, online_folds, online_step_fold, step_context."
```

# The online arm of the fold loop: private API

An Online Scheme — a walk-forward wrapped in `Online` by [`OnlineIndexWalkForward`](@ref), [`OnlineDateWalkForward`](@ref) or [`OnlineHindsightSplit`](@ref) — sends [`PortfolioOptimisers.fold_loop`](@ref) down a third arm. It warms one estimator up on the first training window, folds each fold's new observations into it, and hands the callback a [`PortfolioOptimisers.Fold`](@ref) whose `train` is `nothing` — *the estimator holds its window* — so the fold reads the estimator out through `optimise(opt)` where a refit would have run. The run reaches the weights of the batch expanding-window walk-forward fold for fold.

```@docs
PortfolioOptimisers.fit_fold_result
PortfolioOptimisers.thread_online_folds!
PortfolioOptimisers.online_folds
PortfolioOptimisers.online_step_fold(::Any, ::Nothing, ::ReturnsResult)
PortfolioOptimisers.step_context
```
