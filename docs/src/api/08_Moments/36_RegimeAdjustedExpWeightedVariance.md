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
var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                    estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
```
