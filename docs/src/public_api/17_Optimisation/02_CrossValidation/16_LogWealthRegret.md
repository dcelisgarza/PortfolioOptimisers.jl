```@meta
Description = "Log-wealth regret against a comparator, public API of PortfolioOptimisers.jl: LogWealthRegretResult, log_wealth_regret, BudgetedHindsightPath, …"
```

# Log-wealth regret against a comparator

The evaluation surface of the online selection family, usable by any two prediction results over one sequence of rows. [`log_wealth_regret`](@ref) reads two prediction results scored over the same timestamps and answers the gap in log terminal wealth between the comparator and the strategy, positive when the comparator wins, with the per-row difference and a Newey–West test of equal expected log growth in the shape of [`covariance_forecast_compare`](@ref). The comparator is any prediction result over the rows: the same estimator run causally, or a Hindsight Comparator fit on the rows it is scored on and predicted in sample — [`BestConstantRebalancedPortfolio`](@ref) for the best constant rebalanced portfolio, or a top-1 [`ScoreSelector`](@ref) composed with [`EqualWeighted`](@ref) for the best stock.

Run through a [`HindsightSplit`](@ref) the same estimators are the per-row comparators of dynamic regret, be-the-leader and the per-period minimiser, and the Result reports their path length beside the regret. [`BudgetedHindsightPath`](@ref) is the third comparator of dynamic regret, the best sequence under a path-length budget: an estimator under the same rule, whose fit is one programme over the panel and whose prediction result is one fold per row.

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
