# Assertions

## Configuration

```@docs
assert_resource_cap
assert_ep_grid_size
```

## Utilities

```@docs
resolve_rng
```

## Assertions

In order to increase correctness, robustness, and safety, we make extensive use of [defensive programming](https://en.wikipedia.org/wiki/Defensive_programming). The following functions perform some of these validations and are usually called at variable instantiation.

```@docs
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
assert_returns_result_dims
```
