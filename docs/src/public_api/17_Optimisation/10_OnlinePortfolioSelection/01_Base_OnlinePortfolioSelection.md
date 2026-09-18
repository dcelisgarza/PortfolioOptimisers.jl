```@meta
Description = "Online portfolio selection: the family, the geometry, the set and the state, public API of PortfolioOptimisers.jl: AbstractOnlinePortfolioSelectionAlgorithm, …"
```

# Online portfolio selection: the family, the geometry, the set and the state

The shared vocabulary of the online portfolio selection family (ADRs 0155, 0157, 0159): the abstract rule type every Online Selection Rule subtypes, the Projection Geometries a rule projects its raw step in, the Allocation Set the head holds, the one verb a rule writes, and the Partial Fit State the head carries.

```@docs
PortfolioOptimisers.AbstractOnlinePortfolioSelectionAlgorithm
PortfolioOptimisers.AbstractProjectionGeometry
EuclideanProjection
EntropicProjection
PortfolioOptimisers.AbstractAllocationSet
BoundedAllocationSet
PortfolioOptimisers.resolve_allocation_set
PortfolioOptimisers.project
PortfolioOptimisers.online_update!
PortfolioOptimisers.rule_state_seed
PortfolioOptimisers.projection_geometry
PortfolioOptimisers.rows_needed(::PortfolioOptimisers.AbstractOnlinePortfolioSelectionAlgorithm)
PortfolioOptimisers.merge_states(::PortfolioOptimisers.OnlinePortfolioSelectionState, ::PortfolioOptimisers.OnlinePortfolioSelectionState)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.OnlinePortfolioSelectionState, i, args...)
```
