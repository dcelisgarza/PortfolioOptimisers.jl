```@meta
Description = "EW beta descriptors (a), private API of PortfolioOptimisers.jl: EWBlockState, ew_active_returns, ew_agg_series, ew_agg_vector, ew_beta_expand, Base.copy, …"
```

# EW beta descriptors (a): private API

## Types

```@docs
EWBlockState
```

## Functions

```@docs
ew_active_returns
ew_agg_series
ew_agg_vector
ew_beta_expand
Base.copy(x::PortfolioOptimisers.EWBlockState)
ew_block_copy
ew_block_rows
ew_beta_residual_variance
ew_beta_group_prior
ew_beta_shrink
ew_masked_mean
ew_masked_weighted_mean
assert_ew_agg_obs
assert_ew_shrinkage_bounds
ew_beta_output
```
