# Regime Adjusted Exponential Weighted Variance

## Types

```@docs
RegimeAdjustedMethod
LogRegimeAdjusted
FirstMomentRegimeAdjusted
RootMeanSquaredAdjusted
RegimeAdjustedExpWeightedVariance
RegimeAdjustedVarianceCache
regime_multiplier
get_regime_state
hac_squared_returns!
process_observation!
regime_adjusted_variance_pass!
regime_adjusted_variance
var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                    estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
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
var(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceCache; kwargs...)
var(ce::RegimeAdjustedExpWeightedVariance; kwargs...)
PortfolioOptimisers.merge_states(a::RegimeAdjustedVarianceCache, b::RegimeAdjustedVarianceCache)
```
