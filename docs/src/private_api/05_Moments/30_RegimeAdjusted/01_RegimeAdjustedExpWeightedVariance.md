```@meta
Description = "Regime adjusted exp weighted variance, private API of PortfolioOptimisers.jl: RegimeAdjustedVarianceState, RegimeTermMoments, assert_regime_debias, …"
```

# Regime adjusted exp weighted variance: private API

## Types

```@docs
RegimeAdjustedVarianceState
RegimeTermMoments
```

## Functions

```@docs
assert_regime_debias
regime_bias_factor
regime_log_variance
regime_bias!
regime_bias_open
hac_squared_returns!
blend_regime_state
process_observation!(cache::RegimeAdjustedVarianceState, ce::RegimeAdjustedExpWeightedVariance, X::VecNum, estimation_mask::Option{<:AbstractVector{<:Bool}}, active_mask::Option{<:AbstractVector{<:Bool}})
regime_adjusted_variance_pass!
regime_adjusted_variance
variance_series(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
Base.copy(x::PortfolioOptimisers.RegimeAdjustedVarianceState)
```
