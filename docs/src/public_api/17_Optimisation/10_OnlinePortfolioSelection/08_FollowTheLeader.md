```@meta
Description = "Online selection rules: the third set, public API of PortfolioOptimisers.jl: AbstractSampleSelector, select_rows, Prefix, LastRows, …"
```

# Online selection rules: the third set

The solved rows of the literature: a follow-the-leader rule that re-solves an optimisation estimator on the rows a Sample Selector names, the selectors — every row so far, the last `W`, and the histogram, kernel, nearest-neighbour, correlation and cluster pattern matchers — the Allocation Set Constraint through which the re-solve takes the head's Allocation Set as its feasible region, the short-term loss-control portfolio it constructs over the rank-one covariance, the low-dimension ensemble portfolio it constructs over the ensemble prior, and follow the leading history, a mixture over copies of one rule started at different periods.

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
