```@meta
Description = "Dependency exposure types, public API of PortfolioOptimisers.jl: DependencyExposure, AbstractDependencyReach, ShortestHops, WalkShares, …"
```

# Dependency exposure types

The dependency exposure of a portfolio is the weight of its held assets that depends, directly
or indirectly, on each node of an exogenous graph. The configuration states the reach and the
direction, and the Result holds the long and the short parts of each node.

```@docs
DependencyExposure
PortfolioOptimisers.AbstractDependencyReach
ShortestHops
WalkShares
DependencyExposureResult
DataFrames.DataFrame(res::DependencyExposureResult)
exposure_contributions
```
