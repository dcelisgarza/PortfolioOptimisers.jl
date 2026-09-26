"""
    return_forecast(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult,
                    csfm::CrossSectionalFactorModel) -> AbstractReturnForecastResult

Compute the Return Forecast of a carrier and a fitted factor-model block.

This is the verb every Return Forecast Estimator answers. The block carries the exposure history, the idiosyncratic variance history and the factor axis the members read, so a caller fits a forecast on a stored prior result without refitting the prior.

Every member follows two conventions. The value at an observation uses information up to and including that observation. And `mu` is in return units, whatever the Forecast Unit that the member scores in.

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

# Mathematical definition

```math
\\begin{align}
\\alpha_{ti} &= g_{ti} \\, f_{ti}\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_ti_fc])
  - ``f_{ti}``: Return Forecast of asset ``i`` at observation ``t`` in the Forecast Unit.
  - $(math_dict[:g_ti_unit])
  - $(math_dict[:v_ti_idio])

A `NaN` variance gives a `NaN` forecast.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`IdiosyncraticReturnUnit`](@ref): return `F` unchanged. The method does not read `vs`.
 2. [`IdiosyncraticSharpeUnit`](@ref) with no `vs`: refuse the call, because the conversion needs the idiosyncratic volatility.
 3. [`IdiosyncraticSharpeUnit`](@ref) with a `vs`: check the size of `vs`, and multiply every cell of `F` by its unit factor. A negative variance raises a `DomainError` from `sqrt`.

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
    return_forecast_weights(rd::ReturnsResult) -> MatNum

Return the cross-sectional weights that the transforms of a Return Forecast use.

The weights are the estimation mask of the Asset Panel read as numbers, so an asset that does not enter the cross-sectional estimate of an observation carries no weight there. The Return Forecast family uses this weighting and no other. A Descriptor score is standardised over the estimation universe, not over the benchmark, so no member has a benchmark weight field.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`descriptor_asset_panel`](@ref).
  - The Asset Panel is time-varying, so it carries an estimation mask. Raises an [`IsNothingError`](@ref).

# Returns

  - `w::MatNum`: The cross-sectional weights, `observations × assets`, one where the asset enters the estimate and zero where it does not, in the number type of `rd.X`.

# Related

  - [`DescriptorScores`](@ref)
  - [`descriptor_scores`](@ref)
  - [`AssetPanel`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
function return_forecast_weights(rd::ReturnsResult)::MatNum
    pnl = descriptor_asset_panel(rd)
    emsk = pnl.emsk
    @argcheck(!isnothing(emsk),
              IsNothingError("a Return Forecast scores its Descriptors over the estimation universe of each observation, and this Asset Panel is static, so it carries no estimation mask"))
    # The mask is read in the number type of the returns it weighs, so a `Float32` carrier
    # is not widened by its own weights.
    T = real(eltype(rd.X))
    return T.(emsk)
end
"""
    return_forecast_history_rows(A::Nothing) -> Nothing
    return_forecast_history_rows(A::MatNum_Arr3Num) -> Integer

Return the observation count of one history of a factor-model block, or `nothing`.

Every history of a block lays its observations on the first axis, whatever its rank, so one method reads a per-asset history and an exposure history. An absent history gives `nothing`. So [`return_forecast_block_observations`](@ref) reads the first history that the block carries with no `isnothing` test of its own.

# Arguments

  - `A`: One history of the block, or `nothing`.

# Returns

  - `T::Option{<:Integer}`: The observation count, or `nothing` when the block carries no such history.

# Related

  - [`return_forecast_block_observations`](@ref)
  - [`return_forecast_rows`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function return_forecast_history_rows(::Nothing)::Nothing
    return nothing
end
function return_forecast_history_rows(A::MatNum_Arr3Num)::Integer
    return size(A, 1)
end
"""
    return_forecast_block_observations(csfm::CrossSectionalFactorModel) -> Option{<:Integer}

Return the observation count of the histories of a factor-model block, or `nothing`.

The constructor of [`CrossSectionalFactorModel`](@ref) pins the three per-asset histories `rw`, `vs` and `bw` to one observation axis, so the first of them that the block carries states the count. A block that carries none of them falls back to the exposure history. A block that carries no history at all states no window, and the answer is `nothing`.

# Arguments

  - `csfm`: The fitted factor-model block.

# Returns

  - `Tb::Option{<:Integer}`: The observation count of the block, or `nothing`.

# Related

  - [`return_forecast_rows`](@ref)
  - [`return_forecast_history_rows`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function return_forecast_block_observations(csfm::CrossSectionalFactorModel)::Option{<:Integer}
    Tb = return_forecast_history_rows(csfm.rw)
    Tb = isnothing(Tb) ? return_forecast_history_rows(csfm.vs) : Tb
    Tb = isnothing(Tb) ? return_forecast_history_rows(csfm.bw) : Tb
    return isnothing(Tb) ? return_forecast_history_rows(csfm.Ms) : Tb
end
"""
    return_forecast_rows(rd::ReturnsResult, csfm::CrossSectionalFactorModel) -> AbstractUnitRange

Return the rows of the carrier the histories of a factor-model block live on.

A [`CrossSectionalFactorPrior`](@ref) drops the leading observations that its Descriptors warm up over, and fits on the observations that remain. So the block is always a suffix of the carrier. The function finds the suffix by size, not by a stored offset. A stored offset must survive every view of the block, and the size arithmetic holds on every view.

A Return Forecast Estimator scores its own Descriptors over the whole carrier, so they warm up on every observation of the panel. Each member then cuts the scores to these rows. A carrier of the same length as the block gives the whole range. A caller who hands the carrier that is already narrowed gets this answer.

# Mathematical definition

```math
\\begin{align}
\\mathcal{R} &= \\left\\{T_{c} - T_{b} + 1, \\dots, T_{c}\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{R}``: The rows of the carrier that the block lives on.
  - ``T_{c}``: Number of observations of the carrier.
  - ``T_{b}``: Number of observations of the block. A block that carries no history states no window, and then ``T_{b} = T_{c}``.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block.

# Validation

  - The rules of [`descriptor_asset_panel`](@ref).
  - The carrier is at least as long as the block. Raises a `DimensionMismatch`.

# Returns

  - `rows::AbstractUnitRange`: The rows of the carrier the block lives on.

# Related

  - [`return_forecast_block_observations`](@ref)
  - [`return_forecast_cut`](@ref)
  - [`return_forecast_pad`](@ref)
  - [`descriptor_scores`](@ref)
"""
function return_forecast_rows(rd::ReturnsResult,
                              csfm::CrossSectionalFactorModel)::AbstractUnitRange
    Tc = size(descriptor_asset_panel(rd).amsk, 1)
    Tb = return_forecast_block_observations(csfm)
    if isnothing(Tb)
        return 1:Tc
    end
    @argcheck(Tb <= Tc,
              DimensionMismatch("the factor model block is fitted on a suffix of the carrier, so the carrier ($Tc observations) cannot be shorter than the block ($Tb observations). Hand the carrier the prior was fitted on."))
    return (Tc - Tb + 1):Tc
end
"""
    return_forecast_cut(A::Nothing, rows::AbstractUnitRange) -> Nothing
    return_forecast_cut(A::MatNum, rows::AbstractUnitRange) -> MatNum
    return_forecast_cut(A::Arr3Num, rows::AbstractUnitRange) -> Arr3Num

Cut a history that lives on the carrier's observation axis down to the block's rows.

The Return Forecast family computes its Descriptor scores over the whole carrier and answers on the block's rows. So this verb cuts each whole-axis history that the family carries beside the scores, once. The cut is a copy and not a view, because a view of an array whose element type is open names no numeric array.

# Arguments

  - `A`: A history on the carrier's observation axis, or `nothing`.
  - `rows`: The rows of the carrier the block lives on.

# Returns

  - `A`: The same history on the block's rows, or `nothing`.

# Related

  - [`return_forecast_rows`](@ref)
  - [`return_forecast_pad`](@ref)
  - [`descriptor_scores`](@ref)
"""
function return_forecast_cut(::Nothing, ::AbstractUnitRange)::Nothing
    return nothing
end
function return_forecast_cut(A::MatNum, rows::AbstractUnitRange)::MatNum
    return A[rows, :]
end
function return_forecast_cut(A::Arr3Num, rows::AbstractUnitRange)::Arr3Num
    return A[rows, :, :]
end
"""
    return_forecast_pad(A::Nothing, rows::AbstractUnitRange, T::Integer) -> Nothing
    return_forecast_pad(A::MatNum, rows::AbstractUnitRange, T::Integer) -> MatNum

Place a history of a factor-model block into the rows of the carrier it was fitted on.

The rows before the block carry no information of the factor model, so they are `NaN`. A member that fits over the whole carrier reads them as it reads any missing cell. Where the history that a pair needs is not finite, the pair drops out of the fit.

# Arguments

  - `A`: A history on the block's observation axis, or `nothing`.
  - `rows`: The rows of the carrier the block lives on.
  - `T`: Number of observations the carrier has.

# Validation

  - The element type of `A` holds `NaN` after [`float_if_integer`](@ref). A `Rational` element type raises an `ArgumentError`.

# Returns

  - `A`: The same history on the carrier's axis, `NaN` before the block, or `nothing`. Its element type is [`float_if_integer`](@ref) of the element type of `A`, so an integer history becomes a floating-point history.

# Related

  - [`return_forecast_rows`](@ref)
  - [`return_forecast_cut`](@ref)
  - [`TargetReturnForecast`](@ref)
"""
function return_forecast_pad(::Nothing, ::AbstractUnitRange, ::Integer)::Nothing
    return nothing
end
function return_forecast_pad(A::MatNum, rows::AbstractUnitRange, T::Integer)::MatNum
    Tf = float_if_integer(real(eltype(A)))
    @argcheck(!(Tf <: Rational),
              ArgumentError("a factor model history is NaN on the observations before the block, and its element type $Tf cannot hold NaN. Convert the history to a floating-point type."))
    B = fill(Tf(NaN), T, size(A, 2))
    B[rows, :] = A
    return B
end
"""
    forecast_unit_target(unit::IdiosyncraticReturnUnit, y::MatNum,
                         vs::Option{<:MatNum}) -> MatNum
    forecast_unit_target(unit::IdiosyncraticSharpeUnit, y::MatNum, vs::MatNum) -> MatNum

Convert a forward idiosyncratic return into the Forecast Unit a fitted member scores in.

This is the inverse of [`forecast_return_units`](@ref). A fitted member regresses its Descriptor scores on a target, and the target must be in the unit of the scores. So the two conversions are one pair of methods on the tag.

# Mathematical definition

```math
\\begin{align}
\\tilde{y}_{ti} &= \\frac{y_{ti}}{g_{ti}}\\,.
\\end{align}
```

Where:

  - ``\\tilde{y}_{ti}``: Target of asset ``i`` at observation ``t`` in the Forecast Unit.
  - $(math_dict[:y_ti_fwd])
  - $(math_dict[:g_ti_unit])
  - $(math_dict[:v_ti_idio])

A zero variance gives an infinite target. That target is not finite, so the pair drops out of the fit.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`IdiosyncraticReturnUnit`](@ref): return `y` unchanged. The method does not read `vs`.
 2. [`IdiosyncraticSharpeUnit`](@ref): check the size of `vs`, and divide every cell of `y` by its unit factor.

# Arguments

  - `unit`: The Forecast Unit the member scores in.
  - `y`: Forward idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.

# Validation

  - `size(vs) == size(y)` when the unit is [`IdiosyncraticSharpeUnit`](@ref). Raises a `DimensionMismatch`.

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
    @argcheck(size(vs) == size(y),
              DimensionMismatch("vs ($(size(vs, 1))×$(size(vs, 2))) must match the forward target ($(size(y, 1))×$(size(y, 2)))"))
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

A fitted member reads the variances whatever its Forecast Unit. In the return unit they are the regression weights, and in the Sharpe unit they scale the target and the forecast. The function refuses a finite variance that is not strictly positive. Such a variance is a weight of infinity in the return unit and a division by zero in the Sharpe unit.

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

The target of observation `t` is the mean of the returns over the observations `t + lag` to `t + lag + horizon - 1`. The fitted members of the Return Forecast family regress on this target, so a member states a `horizon` and a `lag` and not a single offset.

The mean skips a cell that is not finite, so a window with one missing return still gives a target.

# Mathematical definition

```math
\\begin{align}
y_{ti} &= \\begin{cases}
\\dfrac{1}{\\lvert \\mathcal{W}_{ti} \\rvert} \\displaystyle\\sum_{s \\in \\mathcal{W}_{ti}} x_{si} & \\text{if } t + \\ell + h - 1 \\leq T \\text{ and } \\mathcal{W}_{ti} \\neq \\emptyset\\,, \\\\
\\mathrm{NaN} & \\text{otherwise}\\,,
\\end{cases} \\\\
\\mathcal{W}_{ti} &= \\left\\{s \\in \\left\\{t + \\ell, \\dots, t + \\ell + h - 1\\right\\} : x_{si} \\in \\mathbb{R}\\right\\}\\,.
\\end{align}
```

Where:

  - $(math_dict[:y_ti_fwd])
  - ``x_{si}``: Return of asset ``i`` at observation ``s``.
  - ``\\ell``: The lag.
  - ``h``: The horizon.
  - ``\\mathcal{W}_{ti}``: The observations of the forward window of asset ``i`` at observation ``t`` that carry a finite return.
  - $(math_dict[:T])

So the last ``\\ell + h - 1`` observations are `NaN`.

# Arguments

  - `X`: Return history, `observations × assets`.
  - $(arg_dict[:rf_horizon])
  - $(arg_dict[:rf_lag])

# Validation

  - The element type of `X` holds `NaN` after [`float_if_integer`](@ref). A `Rational` element type raises an `ArgumentError`.

# Returns

  - `Y::Matrix{<:Real}`: Forward mean returns, `observations × assets`. Its element type is [`float_if_integer`](@ref) of the element type of `X`, so integer returns give a floating-point mean.

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
    Tf = float_if_integer(real(eltype(X)))
    @argcheck(!(Tf <: Rational),
              ArgumentError("a forward mean is NaN where its window holds no finite return and on the last lag + horizon - 1 observations, and the element type $Tf cannot hold NaN. Convert the returns to a floating-point type."))
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
