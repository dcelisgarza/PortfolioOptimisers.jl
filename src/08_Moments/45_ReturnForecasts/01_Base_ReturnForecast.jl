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

"""
    forecast_unit_target(unit::IdiosyncraticReturnUnit, y::MatNum,
                         vs::Option{<:MatNum}) -> MatNum
    forecast_unit_target(unit::IdiosyncraticSharpeUnit, y::MatNum, vs::MatNum) -> MatNum

Convert a forward idiosyncratic return into the Forecast Unit a fitted member scores in.

This is the inverse of [`forecast_return_units`](@ref). A fitted member regresses its Descriptor scores on a target, and the target must stand in the unit the scores are read in, so the two conversions are one pair of methods on the tag.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`IdiosyncraticReturnUnit`](@ref): the target is already the idiosyncratic return, so it is returned unchanged and `vs` is not read.
 2. [`IdiosyncraticSharpeUnit`](@ref): every cell is divided by the square root of the idiosyncratic variance of the same observation and asset. A cell whose variance is zero leaves an infinite target, which is not finite, so the pair drops out of the fit.

# Arguments

  - `unit`: The Forecast Unit the member scores in.
  - `y`: Forward idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.

# Returns

  - `y::MatNum`: The target in the Forecast Unit.

# Examples

```jldoctest
julia> PortfolioOptimisers.forecast_unit_target(IdiosyncraticSharpeUnit(), [0.2 1.0], [0.04 0.25])
1×2 Matrix{Float64}:
 1.0  2.0
```

# Related

  - [`AbstractForecastUnit`](@ref)
  - [`forecast_return_units`](@ref)
  - [`forward_mean_returns`](@ref)
"""
function forecast_unit_target(::IdiosyncraticReturnUnit, y::MatNum,
                              ::Option{<:MatNum})::MatNum
    return y
end
function forecast_unit_target(::IdiosyncraticSharpeUnit, y::MatNum, vs::MatNum)::MatNum
    return y ./ sqrt.(vs)
end
"""
    forecast_idiosyncratic_returns(csfm::CrossSectionalFactorModel) -> MatNum

Return the idiosyncratic return history a fitted Return Forecast builds its target from.

The history lives on the cross-sectional fit the block nests, which is optional, so this is the one place that states the refusal.

# Arguments

  - `csfm`: The fitted factor-model block.

# Validation

  - `csfm.csr` is given. Raises an [`IsNothingError`](@ref).

# Returns

  - `eps::MatNum`: Idiosyncratic returns, `observations × assets`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalRegression`](@ref)
  - [`forward_mean_returns`](@ref)
"""
function forecast_idiosyncratic_returns(csfm::CrossSectionalFactorModel)::MatNum
    csr = csfm.csr
    @argcheck(!isnothing(csr),
              IsNothingError("a fitted Return Forecast regresses its Descriptor scores on the forward idiosyncratic return, and the factor model block carries no cross-sectional fit in csr"))
    return csr.eps
end
"""
    forecast_idiosyncratic_variances(csfm::CrossSectionalFactorModel) -> MatNum

Return the idiosyncratic variance history a fitted Return Forecast weighs its fit by.

A fitted member reads the variances whatever its Forecast Unit: in the return unit they are the regression weights, and in the Sharpe unit they scale the target and the forecast. A finite variance that is not strictly positive is refused rather than carried, because it is a weight of infinity in the first reading and a division by zero in the second.

# Arguments

  - `csfm`: The fitted factor-model block.

# Validation

  - `csfm.vs` is given. Raises an [`IsNothingError`](@ref).
  - Every finite entry of `csfm.vs` is strictly positive. Raises a `DomainError`.

# Returns

  - `vs::MatNum`: Idiosyncratic variances, `observations × assets`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`forecast_unit_target`](@ref)
  - [`forecast_return_units`](@ref)
"""
function forecast_idiosyncratic_variances(csfm::CrossSectionalFactorModel)::MatNum
    vs = csfm.vs
    @argcheck(!isnothing(vs),
              IsNothingError("a fitted Return Forecast weighs its fit by the idiosyncratic variance, and the factor model block carries no variance history in vs"))
    for idx in CartesianIndices(vs)
        v = vs[idx]
        @argcheck(!isfinite(v) || v > zero(v),
                  DomainError(v,
                              "every finite idiosyncratic variance weighs a fit, so it must be strictly positive, got vs[$(idx[1]), $(idx[2])] = $v"))
    end
    return vs
end
"""
    forward_mean_returns(X::MatNum, horizon::Integer, lag::Integer) -> Matrix{<:Real}

Return the forward mean of a return history, one target per observation and asset.

The target of observation `t` is the mean of the returns over the observations `t + lag` to `t + lag + horizon - 1`. It is the one target both fitted members of the Return Forecast family regress on, and it is why a member states a `horizon` and a `lag` rather than a single offset.

The mean skips a cell that is not finite, so a window with one missing return still gives a target. A window with no finite return, and the last `lag + horizon - 1` observations, give `NaN`.

# Mathematical definition

```math
\\begin{align}
y_{t,i} &= \\frac{1}{\\lvert \\mathcal{W}_{t,i} \\rvert} \\sum_{s \\in \\mathcal{W}_{t,i}} x_{s,i}\\,, &
\\mathcal{W}_{t,i} &= \\left\\{s \\in [t + \\ell,\\, t + \\ell + h - 1] : x_{s,i} \\text{ is finite}\\right\\}\\,.
\\end{align}
```

Where:

  - ``x_{s,i}``: return of asset ``i`` at observation ``s``.
  - ``\\ell``: the lag.
  - ``h``: the horizon.
  - ``\\mathcal{W}_{t,i}``: the finite returns of the forward window of asset ``i`` at observation ``t``.

# Arguments

  - `X`: Return history, `observations × assets`.
  - $(arg_dict[:rf_horizon])
  - $(arg_dict[:rf_lag])

# Returns

  - `Y::Matrix{<:Real}`: Forward mean returns, `observations × assets`.

# Examples

```jldoctest
julia> PortfolioOptimisers.forward_mean_returns([1.0; 2.0; NaN; 4.0; 5.0;;], 2, 1)
5×1 Matrix{Float64}:
   2.0
   4.0
   4.5
 NaN
 NaN
```

# Related

  - [`forecast_idiosyncratic_returns`](@ref)
  - [`forecast_unit_target`](@ref)
  - [`return_forecast`](@ref)
"""
function forward_mean_returns(X::MatNum, horizon::Integer, lag::Integer)::Matrix{<:Real}
    Tf = float(real(eltype(X)))
    T = size(X, 1)
    Y = fill(Tf(NaN), T, size(X, 2))
    gap = lag + horizon - 1
    for i in axes(X, 2), t in 1:(T - gap)
        s = zero(Tf)
        n = 0
        for k in (t + lag):(t + gap)
            x = X[k, i]
            if isfinite(x)
                s += x
                n += 1
            end
        end
        if n > 0
            Y[t, i] = s / n
        end
    end
    return Y
end

export return_forecast, IdiosyncraticReturnUnit, IdiosyncraticSharpeUnit
