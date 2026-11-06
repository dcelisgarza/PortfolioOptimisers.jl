```@meta
Description = "Regime adjusted exp weighted covariance (a), private API of PortfolioOptimisers.jl: RegimeAdjustedCovarianceState, pair_weighted_correlation, …"
```

# Regime adjusted exp weighted covariance (a): private API

## Types

```@docs
RegimeAdjustedCovarianceState
```

## Functions

```@docs
pair_weighted_correlation
update_var_cor!
bias_corrected_covariance
update_regime!
process_observation!(cache::RegimeAdjustedCovarianceState, ce::RegimeAdjustedExpWeightedCovariance, X::VecNum, estimation_mask::Option{<:AbstractVector{<:Bool}}, active_mask::Option{<:AbstractVector{<:Bool}})
apply_active_mask!
assert_regime_target
regime_adjusted_covariance_pass!
regime_adjusted_covariance
regime_adjusted_correlation
```
