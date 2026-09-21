```@meta
Description = "Online portfolio selection: the family, the geometry, the set and the state, private API of PortfolioOptimisers.jl: simplex_bounds, assert_feasible_bounds, …"
```

# Online portfolio selection: the family, the geometry, the set and the state: private API

```@docs
PortfolioOptimisers.simplex_bounds
PortfolioOptimisers.assert_feasible_bounds
PortfolioOptimisers.bounded_quadratic_projection
PortfolioOptimisers.bounded_root
PortfolioOptimisers.project_simplex
PortfolioOptimisers.gram_geometry
PortfolioOptimisers.HeldStep
PortfolioOptimisers.ProjectionStep
PortfolioOptimisers.PROJECTION_STEP
PortfolioOptimisers.with_projection_step
PortfolioOptimisers.record_held_step!
PortfolioOptimisers.projection_step_rows
PortfolioOptimisers.projection_step_names
PortfolioOptimisers.projection_step_strict
PortfolioOptimisers.assert_rule_admits_set
PortfolioOptimisers.rows_needed_max
PortfolioOptimisers.rule_state_view
PortfolioOptimisers.price_adjusted_allocation
PortfolioOptimisers.OnlinePortfolioSelectionState
Base.copy(x::PortfolioOptimisers.OnlinePortfolioSelectionState)
PortfolioOptimisers.renormalised_view
PortfolioOptimisers.statistic_before_rate
PortfolioOptimisers.statistic_after_step
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
