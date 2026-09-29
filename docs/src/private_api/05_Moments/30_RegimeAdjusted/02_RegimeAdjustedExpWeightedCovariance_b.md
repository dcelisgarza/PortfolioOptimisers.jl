```@meta
Description = "Regime adjusted exp weighted covariance (b), private API of PortfolioOptimisers.jl: regime_target_statistic, diagonal_law_factor, diagonal_law_spectrum, …"
```

# Regime adjusted exp weighted covariance (b): private API

## Functions

```@docs
regime_target_statistic
diagonal_law_factor
diagonal_law_spectrum
regime_law_log_laplace
regime_law_grid
regime_law_factor
exp_weight_cross_sum
inverse_volatility_bias
regime_bias_state
safe_regime_cholesky
mahalanobis_bias
mahalanobis_bias_sums
gap_fill_value(::RegimeAdjustedExpWeightedCovariance)
Base.copy(x::PortfolioOptimisers.RegimeAdjustedCovarianceState)
variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```
