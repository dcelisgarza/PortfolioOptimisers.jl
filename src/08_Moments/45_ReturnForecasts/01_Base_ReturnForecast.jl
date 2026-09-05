"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Return Forecast Estimator types.

A Return Forecast Estimator produces a Return Forecast: the part of an asset's expected return the caller states or fits from Descriptors, over and above the factor mean. It reads an Asset Panel and a fitted [`CrossSectionalFactorModel`](@ref), so it answers [`return_forecast`](@ref) and never `mean(me, X)`: a Return Forecast is not an expected returns estimator, and the two families are kept apart so that no `me` slot admits one.

All concrete types producing a Return Forecast should be subtypes of `AbstractReturnForecastEstimator`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractReturnForecastEstimator` and implement the following methods:

## `return_forecast`

  - [`return_forecast(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult, csfm::CrossSectionalFactorModel)`](@ref): Computes the Return Forecast of a carrier and a factor-model block.

### Arguments

  - `rfe`: The concrete subtype instance.
  - `rd`: The returns result that carries the Asset Panel.
  - `csfm`: The fitted factor-model block.

### Returns

  - `rf::AbstractReturnForecastResult`: The member's own Result.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractReturnForecastResult`](@ref)
  - [`return_forecast`](@ref)
  - [`CustomValueReturnForecast`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
  - [`DescriptorScores`](@ref)
"""
abstract type AbstractReturnForecastEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Return Forecast Result types.

Each member of the Return Forecast family carries its own fitted state, so each answers its own Result type. This root states the two reads every one of them answers:

  - `mu`: The latest Return Forecast, in return units, one entry per asset of the coverage universe. It is `NaN` for an asset the member forecasts nothing for.
  - `hist`: The Return Forecast history, `observations × assets` and in return units, or `nothing` for a member that computes none.

A consumer reads those two fields off any member, and reads the member's own fields only when it knows which member it holds.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractReturnForecastEstimator`](@ref)
  - [`return_forecast`](@ref)
  - [`CustomValueReturnForecastResult`](@ref)
  - [`FixedWeightedReturnForecastResult`](@ref)
"""
abstract type AbstractReturnForecastResult <: AbstractResult end
"""
    return_forecast(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult,
                    csfm::CrossSectionalFactorModel) -> AbstractReturnForecastResult

Compute the Return Forecast of a carrier and a fitted factor-model block.

This is the verb every Return Forecast Estimator answers. The block carries the exposure history, the idiosyncratic variance history and the factor axis the members read, so a caller fits a forecast on a stored prior result without refitting the prior.

Every member follows two conventions: the value at an observation uses information up to and including that observation, and `mu` is in return units whatever the Forecast Unit the member scores in.

# Arguments

  - `rfe`: Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block.

# Returns

  - `rf::AbstractReturnForecastResult`: The member's own Result.

# Related

  - [`AbstractReturnForecastEstimator`](@ref)
  - [`AbstractReturnForecastResult`](@ref)
  - [`CustomValueReturnForecast`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function return_forecast end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Forecast Unit tags.

A Forecast Unit says what the Descriptors of a fitted member forecast, before the member converts the answer to return units. The unit is a tag rather than a flag, so the conversion is a method and no member writes a branch over it.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`IdiosyncraticReturnUnit`](@ref)
  - [`IdiosyncraticSharpeUnit`](@ref)
  - [`forecast_return_units`](@ref)
"""
abstract type AbstractForecastUnit <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The Descriptors forecast the idiosyncratic return itself.

This is the default unit. The forecast is already in return units, so [`forecast_return_units`](@ref) returns it unchanged and the member reads no idiosyncratic variance.

# Related

  - [`AbstractForecastUnit`](@ref)
  - [`IdiosyncraticSharpeUnit`](@ref)
  - [`forecast_return_units`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
"""
struct IdiosyncraticReturnUnit <: AbstractForecastUnit end
"""
$(DocStringExtensions.TYPEDEF)

The Descriptors forecast the idiosyncratic return divided by the idiosyncratic volatility.

The forecast is a Sharpe ratio, so [`forecast_return_units`](@ref) multiplies it by the idiosyncratic volatility of the same observation. The block must then carry its idiosyncratic variance history in `vs`.

# Related

  - [`AbstractForecastUnit`](@ref)
  - [`IdiosyncraticReturnUnit`](@ref)
  - [`forecast_return_units`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
"""
struct IdiosyncraticSharpeUnit <: AbstractForecastUnit end
"""
    forecast_return_units(unit::IdiosyncraticReturnUnit, F::MatNum,
                          vs::Option{<:MatNum}) -> MatNum
    forecast_return_units(unit::IdiosyncraticSharpeUnit, F::MatNum, vs::Nothing) -> MatNum
    forecast_return_units(unit::IdiosyncraticSharpeUnit, F::MatNum, vs::MatNum) -> MatNum

Convert a Return Forecast history from its Forecast Unit to return units.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`IdiosyncraticReturnUnit`](@ref): the forecast is already in return units, so it is returned unchanged and `vs` is not read.
 2. [`IdiosyncraticSharpeUnit`](@ref) with no `vs`: the conversion needs the idiosyncratic volatility, so the absent history is refused.
 3. [`IdiosyncraticSharpeUnit`](@ref) with a `vs`: every cell is multiplied by the square root of the idiosyncratic variance of the same observation and asset. A `NaN` variance gives a `NaN` forecast, and a negative one raises a `DomainError` from `sqrt`.

# Arguments

  - `unit`: The Forecast Unit the member scores in.
  - `F`: The Return Forecast history in that unit, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`, or `nothing`.

# Validation

  - `vs` is given when the unit is [`IdiosyncraticSharpeUnit`](@ref). Raises an [`IsNothingError`](@ref).
  - `size(vs) == size(F)`. Raises a `DimensionMismatch`.

# Returns

  - `F::MatNum`: The Return Forecast history in return units.

# Examples

```jldoctest
julia> PortfolioOptimisers.forecast_return_units(IdiosyncraticSharpeUnit(), [1.0 2.0], [0.04 0.25])
1×2 Matrix{Float64}:
 0.2  1.0
```

# Related

  - [`AbstractForecastUnit`](@ref)
  - [`IdiosyncraticReturnUnit`](@ref)
  - [`IdiosyncraticSharpeUnit`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function forecast_return_units(::IdiosyncraticReturnUnit, F::MatNum,
                               ::Option{<:MatNum})::MatNum
    return F
end
function forecast_return_units(::IdiosyncraticSharpeUnit, ::MatNum, ::Nothing)::MatNum
    return throw(IsNothingError("a Sharpe unit forecast is converted to return units with the idiosyncratic volatility, and the factor model block carries no vs. Fit the block with an idiosyncratic variance history, or score in return units with IdiosyncraticReturnUnit()"))
end
function forecast_return_units(::IdiosyncraticSharpeUnit, F::MatNum, vs::MatNum)::MatNum
    @argcheck(size(vs) == size(F),
              DimensionMismatch("vs ($(size(vs, 1))×$(size(vs, 2))) must match the Return Forecast history ($(size(F, 1))×$(size(F, 2)))"))
    return F .* sqrt.(vs)
end
"""
    return_forecast_weights(rd::ReturnsResult) -> Matrix{Float64}

Return the cross-sectional weights the transforms of a Return Forecast are weighted by.

The weights are the estimation mask of the Asset Panel read as numbers, so an asset that does not enter the cross-sectional estimate of an observation carries no weight there. This is the one weighting the Return Forecast family uses, and it is why a member states no benchmark weight field: a Descriptor score is standardised over the estimation universe, not over the benchmark.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`descriptor_asset_panel`](@ref).
  - The Asset Panel is time-varying, so it carries an estimation mask. Raises an [`IsNothingError`](@ref).

# Returns

  - `w::Matrix{Float64}`: The cross-sectional weights, `observations × assets`, one where the asset enters the estimate and zero where it does not.

# Related

  - [`DescriptorScores`](@ref)
  - [`descriptor_scores`](@ref)
  - [`AssetPanel`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
function return_forecast_weights(rd::ReturnsResult)::Matrix{Float64}
    pnl = descriptor_asset_panel(rd)
    emsk = pnl.emsk
    @argcheck(!isnothing(emsk),
              IsNothingError("a Return Forecast scores its Descriptors over the estimation universe of each observation, and this Asset Panel is static, so it carries no estimation mask"))
    return Float64.(emsk)
end

export return_forecast, IdiosyncraticReturnUnit, IdiosyncraticSharpeUnit
