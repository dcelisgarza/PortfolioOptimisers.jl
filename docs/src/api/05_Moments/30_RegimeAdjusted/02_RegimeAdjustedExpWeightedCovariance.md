# Regime Adjusted Exponential Weighted Covariance

## Types

```@docs
RegimeAdjustedTarget
MahalanobisTarget
DiagonalTarget
PortfolioTarget
RegimeAdjustedExpWeightedCovariance
RegimeAdjustedCovarianceState
min_active_assets
has_separate_cor_decay
regime_kappa
regime_denom
get_regime_state(::RootMeanSquaredAdjusted, ::RegimeAdjustedTarget, stats::VecNum,
                    n::Integer, ::Any)
get_regime_state(method::FirstMomentRegimeAdjusted, target::RegimeAdjustedTarget,
                    stats::VecNum, n::Integer, ::Any)
get_regime_state(method::LogRegimeAdjusted, target::RegimeAdjustedTarget, stats::VecNum,
                    n::Integer, min_val::Number)
safe_regime_cholesky
regime_statistic
hac_outer_product!
update_var_cor!
bias_corrected_covariance
regime_covariance_block
update_regime!
process_observation!(cache::RegimeAdjustedCovarianceState,
                    ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                    estimation_mask::Option{<:AbstractVector{<:Bool}},
                    active_mask::Option{<:AbstractVector{<:Bool}})
assert_regime_target
regime_adjusted_covariance_pass!
regime_adjusted_covariance
regime_adjusted_correlation
gap_fill_value(::RegimeAdjustedExpWeightedCovariance)
cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                    estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                    estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                    estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, x::VecNum;
                    estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                    kwargs...)
cov(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState;
                    kwargs...)
cov(ce::RegimeAdjustedExpWeightedCovariance; kwargs...)
cor(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState;
                    kwargs...)
cor(ce::RegimeAdjustedExpWeightedCovariance; kwargs...)
PortfolioOptimisers.merge_states(a::RegimeAdjustedCovarianceState, b::RegimeAdjustedCovarianceState)
Base.copy(x::PortfolioOptimisers.RegimeAdjustedCovarianceState)
cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                         estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```
