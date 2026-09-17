```@meta
Description = "Regime Adjusted Exponential Weighted Variance, private API of PortfolioOptimisers.jl: RegimeAdjustedMethod, RegimeAdjustedVarianceState, regime_multiplier, …"
```

# Regime Adjusted Exponential Weighted Variance: private API

## Types

```@docs
RegimeAdjustedMethod
RegimeAdjustedVarianceState
```

## Functions

```@docs
regime_multiplier
get_regime_state(::RootMeanSquaredAdjusted, z2_valid::VecNum, ::Any)
get_regime_state(method::FirstMomentRegimeAdjusted, z2_valid::VecNum, ::Any)
get_regime_state(method::LogRegimeAdjusted, z2_valid::VecNum, min_val::Number = sqrt(eps(eltype(z2_valid))))
hac_squared_returns!
process_observation!(cache::RegimeAdjustedVarianceState, ce::RegimeAdjustedExpWeightedVariance, X::VecNum, estimation_mask::Option{<:AbstractVector{<:Bool}}, active_mask::Option{<:AbstractVector{<:Bool}})
regime_adjusted_variance_pass!
regime_adjusted_variance
variance_series(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
PortfolioOptimisers.merge_states(a::RegimeAdjustedVarianceState, b::RegimeAdjustedVarianceState)
Base.copy(x::PortfolioOptimisers.RegimeAdjustedVarianceState)
```
