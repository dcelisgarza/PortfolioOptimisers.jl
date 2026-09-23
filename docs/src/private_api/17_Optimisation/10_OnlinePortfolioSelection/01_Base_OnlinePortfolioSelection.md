```@meta
Description = "Online portfolio selection: the family, the geometry, the set and the state, private API of PortfolioOptimisers.jl: simplex_bounds, assert_feasible_bounds, …"
```

# Online portfolio selection: the family, the geometry, the set and the state: private API

```@docs
PortfolioOptimisers.simplex_bounds
PortfolioOptimisers.assert_feasible_bounds
PortfolioOptimisers.assert_finite_raw_step
PortfolioOptimisers.budget_or_held_step
PortfolioOptimisers.bounded_quadratic_projection
PortfolioOptimisers.breakpoint_root
PortfolioOptimisers.breakpoint_segment
PortfolioOptimisers.bounded_root
PortfolioOptimisers.bisection_cap
PortfolioOptimisers.project_simplex
PortfolioOptimisers.gram_geometry
PortfolioOptimisers.HeldStep
PortfolioOptimisers.ProjectionStep
PortfolioOptimisers.PROJECTION_STEP
PortfolioOptimisers.with_projection_step
PortfolioOptimisers.price_relative
PortfolioOptimisers.record_held_step!
PortfolioOptimisers.projection_step_rows
PortfolioOptimisers.projection_step_strict
PortfolioOptimisers.assert_rule_admits_set
PortfolioOptimisers.rows_needed_max
PortfolioOptimisers.rule_state_view
PortfolioOptimisers.price_adjusted_allocation
PortfolioOptimisers.project_start
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
