```@meta
Description = "Online selection rules: the third set, public API of PortfolioOptimisers.jl: AbstractSampleSelector, select_rows, Prefix, LastRows, …"
```

# Online selection rules: the third set

The rules that solve an optimisation at each period. `FollowTheLeader` solves an optimisation estimator again on the past rows that a sample selector names, and moves the weights part of the way to the solution. The selectors are `Prefix`, which takes every row so far, `LastRows`, which takes the last `W` rows, and five pattern matchers, `HistogramMatch`, `KernelMatch`, `NearestNeighbourMatch`, `CorrelationMatch` and `ClusterMatch`. A pattern matcher takes the rows whose preceding window resembles the latest one. `AllocationSetConstraint` adds the set of allowed weights of `OnlinePortfolioSelection` to that optimisation. `ShortTermLossControlPortfolio` is a follow-the-leader rule over the rank-one covariance, and `LowDimensionEnsemblePortfolio` is one over the low-dimension ensemble prior. `FollowTheLeadingHistory` mixes copies of one rule started at different periods.

```@docs
PortfolioOptimisers.AbstractSampleSelector
PortfolioOptimisers.select_rows
Prefix
LastRows
PortfolioOptimisers.AbstractPatternMatchSelector
PortfolioOptimisers.matched_rows
HistogramMatch
KernelMatch
NearestNeighbourMatch
CorrelationMatch
ClusterMatch
PortfolioOptimisers.AllocationSetConstraint
FollowTheLeader
ShortTermLossControlPortfolio
LowDimensionEnsemblePortfolio
FollowTheLeadingHistory
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
