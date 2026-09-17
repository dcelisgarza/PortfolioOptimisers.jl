```@meta
Description = "Regime Adjusted Exponential Weighted Covariance, private API of PortfolioOptimisers.jl: RegimeAdjustedCovarianceState, min_active_assets, …"
```

# Regime Adjusted Exponential Weighted Covariance: private API

## Types

```@docs
RegimeAdjustedCovarianceState
```

## Functions

```@docs
min_active_assets
has_separate_cor_decay
regime_kappa
regime_denom
get_regime_state(::RootMeanSquaredAdjusted, ::RegimeAdjustedTarget, stats::VecNum, n::Integer, ::Any)
get_regime_state(method::FirstMomentRegimeAdjusted, target::RegimeAdjustedTarget, stats::VecNum, n::Integer, ::Any)
get_regime_state(method::LogRegimeAdjusted, target::RegimeAdjustedTarget, stats::VecNum, n::Integer, min_val::Number)
safe_regime_cholesky
regime_statistic
hac_outer_product!
update_var_cor!
bias_corrected_covariance
regime_covariance_block
update_regime!
process_observation!(cache::RegimeAdjustedCovarianceState, ce::RegimeAdjustedExpWeightedCovariance, X::VecNum, estimation_mask::Option{<:AbstractVector{<:Bool}}, active_mask::Option{<:AbstractVector{<:Bool}})
assert_regime_target
regime_adjusted_covariance_pass!
regime_adjusted_covariance
regime_adjusted_correlation
gap_fill_value(::RegimeAdjustedExpWeightedCovariance)
PortfolioOptimisers.merge_states(a::RegimeAdjustedCovarianceState, b::RegimeAdjustedCovarianceState)
Base.copy(x::PortfolioOptimisers.RegimeAdjustedCovarianceState)
variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```
