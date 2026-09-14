# Regime Adjusted Exponential Weighted Variance

## Types

```@docs
RegimeAdjustedMethod
LogRegimeAdjusted
FirstMomentRegimeAdjusted
RootMeanSquaredAdjusted
RegimeAdjustedExpWeightedVariance
RegimeAdjustedVarianceState
regime_multiplier
get_regime_state(::RootMeanSquaredAdjusted, z2_valid::VecNum, ::Any)
get_regime_state(method::FirstMomentRegimeAdjusted, z2_valid::VecNum, ::Any)
get_regime_state(method::LogRegimeAdjusted, z2_valid::VecNum,
                 min_val::Number = sqrt(eps(eltype(z2_valid))))
hac_squared_returns!
process_observation!(cache::RegimeAdjustedVarianceState,
                    ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                    estimation_mask::Option{<:AbstractVector{<:Bool}},
                    active_mask::Option{<:AbstractVector{<:Bool}})
regime_adjusted_variance_pass!
regime_adjusted_variance
var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                    estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
variance_series(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                    estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                    estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
partial_fit!(ce::RegimeAdjustedExpWeightedVariance, x::VecNum;
                    estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                    kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance; kwargs...)
std(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                    estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
std(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)
std(ce::RegimeAdjustedExpWeightedVariance; kwargs...)
PortfolioOptimisers.merge_states(a::RegimeAdjustedVarianceState, b::RegimeAdjustedVarianceState)
Base.copy(x::PortfolioOptimisers.RegimeAdjustedVarianceState)
```
