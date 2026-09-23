```@meta
Description = "Log-wealth regret against a comparator, public API of PortfolioOptimisers.jl: LogWealthRegretResult, log_wealth_regret, BudgetedHindsightPath, …"
```

# Log-wealth regret against a comparator

[`log_wealth_regret`](@ref) compares two prediction results scored over the same timestamps, a strategy and a comparator. It returns the difference in log final wealth between the comparator and the strategy, which is positive when the comparator ends with more. It also returns the difference at each row, and a Diebold-Mariano-West test, with a Newey-West variance, that the two have the same expected log growth, in the same form as [`covariance_forecast_compare`](@ref). It was written for the online selection rules, and it accepts any two prediction results over one sequence of rows.

The comparator can be the same estimator run causally. It can also be a hindsight comparator, an estimator fitted on the rows it is scored on and predicted on those same rows. [`BestConstantRebalancedPortfolio`](@ref) gives the best constant rebalanced portfolio. A [`ScoreSelector`](@ref) that keeps the top asset, composed with [`EqualWeighted`](@ref), gives the best single asset.

Run through a [`HindsightSplit`](@ref), the same estimators give the comparators of dynamic regret, which can change at each row. These are be-the-leader, the best constant rebalanced portfolio over the rows up to and including each row, and the best portfolio of each period. The result reports the path length of these comparators next to the regret. [`BudgetedHindsightPath`](@ref) is the third comparator of dynamic regret, the best sequence of weights whose path length stays within a budget. Like the others, it is an estimator. Its fit is one optimisation over the whole panel, and its prediction result has one fold per row.

```@docs
LogWealthRegretResult
log_wealth_regret
BudgetedHindsightPath
BudgetedHindsightPathResult
factory(res::BudgetedHindsightPathResult, fb::Option{<:FbChain})
_optimise(est::BudgetedHindsightPath, rd::ReturnsResult; dims::Int = 1, kwargs...)
predict(res::BudgetedHindsightPathResult, rd::ReturnsResult)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
