```@meta
Description = "Online portfolio selection: the family, the geometry, the set and the state, public API of PortfolioOptimisers.jl: AbstractOnlinePortfolioSelectionAlgorithm, …"
```

# Online portfolio selection: the family, the geometry, the set and the state

This page has the types and functions that every online portfolio selection rule shares. A rule goes in the `alg` field of `OnlinePortfolioSelection`, and it updates the weights after each observation. The page has the abstract type of every rule, and the projection geometries, the measures of distance in which a rule moves its raw step back into the set of allowed weights. `BoundedAllocationSet` is a set of allowed weights for the `set` field. A first-order rule can hold a learning-rate schedule in place of a fixed step size, and the abstract type of the schedules is also on this page. `online_update!` is the one function a new rule must write. The other functions start and update the state of a rule and of a schedule.

```@docs
PortfolioOptimisers.AbstractOnlinePortfolioSelectionAlgorithm
PortfolioOptimisers.AbstractProjectionGeometry
EuclideanProjection
EntropicProjection
GramProjection
PortfolioOptimisers.AbstractLearningRateSchedule
BoundedAllocationSet
PortfolioOptimisers.resolve_allocation_set
PortfolioOptimisers.project
PortfolioOptimisers.online_update!
PortfolioOptimisers.rule_state_seed
PortfolioOptimisers.projection_geometry
PortfolioOptimisers.rows_needed(::PortfolioOptimisers.AbstractOnlinePortfolioSelectionAlgorithm)
PortfolioOptimisers.learning_rate
PortfolioOptimisers.restart
PortfolioOptimisers.schedule_state_seed
PortfolioOptimisers.schedule_update!
PortfolioOptimisers.mixing_share
PortfolioOptimisers.reads_period_row
PortfolioOptimisers.schedule_state_view
PortfolioOptimisers.merge_states(::PortfolioOptimisers.OnlinePortfolioSelectionState, ::PortfolioOptimisers.OnlinePortfolioSelectionState)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.OnlinePortfolioSelectionState, i, args...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
