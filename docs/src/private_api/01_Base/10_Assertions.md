```@meta
Description = "Assertions, private API of PortfolioOptimisers.jl: assert_resource_cap, assert_ep_grid_size, resolve_rng, assert_nonempty, assert_finite, assert_nonneg, …"
```

# [Assertions: private API](@id private-api-assertions)

The library checks each input where a constructor or a function receives it, a practice called [defensive programming](https://en.wikipedia.org/wiki/Defensive_programming). Most of the functions below do one such check. A failed check throws an error that names the argument and the condition it failed, and [`assert_resource_cap`](@ref) also names the limit to raise.

```@docs
assert_resource_cap
assert_ep_grid_size
resolve_rng
assert_nonempty
assert_finite
assert_nonneg
assert_gt0
assert_nonempty_nonneg_finite_val
assert_nonempty_gt0_finite_val
assert_nonempty_finite_val
assert_matrix_issquare
assert_unit_interval
assert_closed_unit_interval
assert_all_finite
assert_source_selector
```
