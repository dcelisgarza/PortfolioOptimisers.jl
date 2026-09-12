"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the quantities a Return Forecast is scored against out of sample.

A Return Forecast states one number per asset and observation, and an evaluation only means something once that number is paired with what actually happened next. The member of this family says **which** history the forward window is taken over; [`forward_mean_returns`](@ref) turns that history into the target, and the horizon and the lag are the evaluation's own parameters rather than the target's.

All concrete subtypes should subtype `AbstractForecastTarget`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype
`AbstractForecastTarget` and implement the following methods:

## Required method name

  - `forecast_target_history(target::AbstractForecastTarget, rd::ReturnsResult, csfm::CrossSectionalFactorModel) -> MatNum`: Return the history the forward target is taken over, on the block's observation axis.

### Arguments

  - `target`: The concrete subtype instance.
  - $(arg_dict[:rd])
  - `csfm`: The fitted factor-model block.

### Returns

  - `X::MatNum`: The history, `observations × assets`, on the block's rows.

# Related

  - [`IdiosyncraticTarget`](@ref)
  - [`AssetReturnTarget`](@ref)
  - [`PanelFieldTarget`](@ref)
  - [`forecast_target_history`](@ref)
  - [`forecast_evaluation`](@ref)
"""
abstract type AbstractForecastTarget <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Scores a Return Forecast against the forward idiosyncratic return.

This is the default target, and it is the component a Return Forecast actually forecasts: a fitted member regresses its Descriptor scores on the forward idiosyncratic return inside [`CrossSectionalFactorPrior`](@ref), so the idiosyncratic history is what the forecast is answerable for. A block that carries no cross-sectional fit states the refusal, through [`forecast_idiosyncratic_returns`](@ref).

# Examples

```jldoctest
julia> IdiosyncraticTarget()
IdiosyncraticTarget()
```

# Related

  - [`AbstractForecastTarget`](@ref)
  - [`AssetReturnTarget`](@ref)
  - [`PanelFieldTarget`](@ref)
  - [`forecast_idiosyncratic_returns`](@ref)
"""
struct IdiosyncraticTarget <: AbstractForecastTarget end
"""
$(DocStringExtensions.TYPEDEF)

Scores a Return Forecast against the forward asset return.

The asset return carries the factor component as well as the idiosyncratic one, so a forecast that ranks the idiosyncratic return perfectly scores lower here whenever the factors move the cross-section. It is the target of a caller who asks what the forecast is worth on the return a portfolio actually earns, rather than on the component the forecast was fitted to.

# Examples

```jldoctest
julia> AssetReturnTarget()
AssetReturnTarget()
```

# Related

  - [`AbstractForecastTarget`](@ref)
  - [`IdiosyncraticTarget`](@ref)
  - [`PanelFieldTarget`](@ref)
"""
struct AssetReturnTarget <: AbstractForecastTarget end
"""
$(DocStringExtensions.TYPEDEF)

Scores a Return Forecast against the forward mean of a named numeric Panel Field.

The field is read through [`panel_field_values`](@ref), so a cell the panel fill touched comes back as `NaN` and never enters the target. Naming the field rather than passing the matrix keeps the target a value the Result can carry and print, and refuses a field the panel does not hold at read time rather than silently scoring against nothing.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PanelFieldTarget(; name::AbstractString) -> PanelFieldTarget

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(name)`. Raises an [`IsEmptyError`](@ref).

# Examples

```jldoctest
julia> PanelFieldTarget(; name = \"excess_return\")
PanelFieldTarget
  name ┴ String: "excess_return"
```

# Related

  - [`AbstractForecastTarget`](@ref)
  - [`IdiosyncraticTarget`](@ref)
  - [`AssetReturnTarget`](@ref)
  - [`panel_field_values`](@ref)
"""
@concrete struct PanelFieldTarget <: AbstractForecastTarget
    """
    Name of the numeric Panel Field the forward target is taken over.
    """
    name
    function PanelFieldTarget(name::AbstractString)
        @argcheck(!isempty(name),
                  IsEmptyError("a Panel Field target names the field it scores against, so name cannot be empty"))
        return new{typeof(name)}(name)
    end
end
function PanelFieldTarget(; name::AbstractString)
    return PanelFieldTarget(name)
end
"""
    forecast_target_history(target::IdiosyncraticTarget, rd::ReturnsResult,
                            csfm::CrossSectionalFactorModel) -> MatNum
    forecast_target_history(target::AssetReturnTarget, rd::ReturnsResult,
                            csfm::CrossSectionalFactorModel) -> MatNum
    forecast_target_history(target::PanelFieldTarget, rd::ReturnsResult,
                            csfm::CrossSectionalFactorModel) -> MatNum

Return the history a forward target is taken over, on the observation axis of a factor-model block.

It is the one seam of the [`AbstractForecastTarget`](@ref) family, and the one place the two observation axes are reconciled. The idiosyncratic history lives on the block's rows already; the asset returns and the Panel Fields live on the carrier's rows, and the block is a suffix of the carrier, so both are cut through [`return_forecast_rows`](@ref) and [`return_forecast_cut`](@ref). Every member therefore answers on the same axis as the `hist` of a Return Forecast Result.

# Arguments

  - `target`: The forward target.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block.

# Validation

  - `csfm.csr` is given, for [`IdiosyncraticTarget`](@ref). Raises an [`IsNothingError`](@ref).
  - The rules of [`return_forecast_rows`](@ref), for the other two members.
  - The rules of [`panel_field_values`](@ref), for [`PanelFieldTarget`](@ref).

# Returns

  - `X::MatNum`: The history, `observations × assets`, on the block's rows.

# Related

  - [`AbstractForecastTarget`](@ref)
  - [`forward_mean_returns`](@ref)
  - [`forecast_evaluation`](@ref)
"""
function forecast_target_history(::IdiosyncraticTarget, ::ReturnsResult,
                                 csfm::CrossSectionalFactorModel)::MatNum
    return forecast_idiosyncratic_returns(csfm)
end
function forecast_target_history(::AssetReturnTarget, rd::ReturnsResult,
                                 csfm::CrossSectionalFactorModel)::MatNum
    X::MatNum = rd.X
    return return_forecast_cut(X, return_forecast_rows(rd, csfm))
end
function forecast_target_history(target::PanelFieldTarget, rd::ReturnsResult,
                                 csfm::CrossSectionalFactorModel)::MatNum
    return return_forecast_cut(panel_field_values(rd, target.name),
                               return_forecast_rows(rd, csfm))
end
"""
    forecast_evaluation_history(rfr::AbstractReturnForecastResult) -> MatNum

Return the Return Forecast history an evaluation scores, refusing a member that carries none.

`hist` is the `observations × assets` matrix the whole evaluation runs on, and it is optional on the family: [`CustomValueReturnForecastResult`](@ref) states a forecast rather than computing one, and [`TargetReturnForecastResult`](@ref) fits one cross-section. Both carry `nothing`, so the refusal is stated once, here, rather than at each statistic.

# Arguments

  - `rfr`: A fitted Return Forecast Result.

# Validation

  - `rfr.hist` is given. Raises an [`IsNothingError`](@ref) naming the member and the field.

# Returns

  - `alpha::MatNum`: Return Forecast history, `observations × assets`, in return units.

# Related

  - [`AbstractReturnForecastResult`](@ref)
  - [`forecast_evaluation`](@ref)
"""
function forecast_evaluation_history(rfr::AbstractReturnForecastResult)::MatNum
    hist = rfr.hist
    @argcheck(!isnothing(hist),
              IsNothingError("an evaluation scores a Return Forecast at every observation, and $(nameof(typeof(rfr))) carries nothing in hist"))
    return hist
end
"""
    forecast_evaluation_dates(alpha::MatNum, y::MatNum, step::Integer) -> Vector{Int}

Return the observations an evaluation scores, as row indices into the forecast and the target.

An observation is scorable when at least one asset carries a finite forecast **and** a finite target there. The first and the last such observation bound the evaluation, and the dates run between them in strides of `step`. A stride of the horizon gives forward windows that do not overlap, so the scores are independent; a stride of one gives every observation and overlapping windows.

An observation inside the bounds that is not scorable is kept rather than dropped, because dropping it would make the stride mean different things in different parts of the sample. Its statistics are `NaN`.

# Arguments

  - `alpha`: Return Forecast history, `observations × assets`.
  - `y`: Forward target, `observations × assets`.
  - `step`: Number of observations between two evaluation dates.

# Validation

  - At least one observation is scorable. Raises an [`IsEmptyError`](@ref).

# Returns

  - `dates::Vector{Int}`: Row indices of the observations the evaluation scores.

# Examples

```jldoctest
julia> alpha = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0];

julia> y = [NaN NaN; 1.0 2.0; 3.0 4.0; 5.0 6.0];

julia> PortfolioOptimisers.forecast_evaluation_dates(alpha, y, 2)
2-element Vector{Int64}:
 2
 4
```

# Related

  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function forecast_evaluation_dates(alpha::MatNum, y::MatNum, step::Integer)::Vector{Int}
    T, N = size(alpha)
    lo = 0
    hi = 0
    for t in 1:T
        scorable = false
        for i in 1:N
            if isfinite(alpha[t, i]) && isfinite(y[t, i])
                scorable = true
                break
            end
        end
        if scorable
            hi = t
            if lo == 0
                lo = t
            end
        end
    end
    @argcheck(lo > 0,
              IsEmptyError("an evaluation needs one observation at which some asset carries both a finite forecast and a finite target, and no observation does"))
    return collect(lo:step:hi)
end
"""
$(DocStringExtensions.TYPEDEF)

The out-of-sample pairing of a Return Forecast with what happened next.

`ForecastEvaluationResult` is what [`forecast_evaluation`](@ref) returns. It carries the forecast, the forward target it is scored against, the observations it is scored at, and the parameters that produced all three; every statistic of the evaluation is a verb over it rather than a field on it, so a caller re-parameterises a statistic without re-running the pairing.

# The pairing is computed once

Producing `alpha` can cost a rolling refit, so the Result stores it rather than the estimator that made it. That is what separates this Result from the diagnostics of [`factor_model_summary`](@ref), whose verbs read their block directly and are cheap to call twice.

# The dates are row indices

`dates` indexes the rows of `alpha` and `y`, which live on the observation axis of the factor-model block rather than on that of the carrier. A caller who wants timestamps reads them off the carrier's `ts` at the block's rows, through [`return_forecast_rows`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForecastEvaluationResult(
        alpha, y, dates, target, horizon, lag, step, min_count, ppy
    ) -> ForecastEvaluationResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a
Result, so [`forecast_evaluation`](@ref) builds it and a caller reads it; there is no
keyword constructor, and the type validates nothing of its own.

# Related

  - [`forecast_evaluation`](@ref)
  - [`AbstractForecastTarget`](@ref)
  - [`forward_mean_returns`](@ref)
  - [`AbstractReturnForecastResult`](@ref)
"""
@concrete struct ForecastEvaluationResult <: AbstractResult
    """
    Return Forecast history `observations × assets`, in return units.
    """
    alpha
    """
    Forward target `observations × assets`, on the same axis as `alpha`.
    """
    y
    """
    Row indices of the observations the evaluation scores, in increasing order.
    """
    dates
    """
    The [`AbstractForecastTarget`](@ref) the forward target was taken over.
    """
    target
    """
    $(field_dict[:rf_horizon])
    """
    horizon
    """
    $(field_dict[:rf_lag])
    """
    lag
    """
    Number of observations between two evaluation dates.
    """
    step
    """
    Least number of assets a cross-section needs before a statistic of it is reported.
    """
    min_count
    """
    $(field_dict[:ps_ppy]) It defaults to `1`, which reports the statistics per period.
    """
    ppy
end
"""
    forecast_evaluation_pairing(alpha::MatNum, rd::ReturnsResult,
                                csfm::CrossSectionalFactorModel;
                                target::AbstractForecastTarget = IdiosyncraticTarget(),
                                horizon::Integer = 1, lag::Integer = 1,
                                step::Integer = horizon, min_count::Integer = 3,
                                ppy::Number = 1) -> ForecastEvaluationResult

Pair a Return Forecast history with the forward target built from a carrier and a block.

The two methods of [`forecast_evaluation`](@ref) that take a carrier and a block differ only in where the history comes from — a fitted Result publishes one, an Estimator is asked for one through [`forecast_history`](@ref) — so the target is built once, here, and the axis check that catches a carrier the forecast was not fitted on is stated once with it.

# Arguments

  - `alpha`: Return Forecast history `observations × assets`, in return units, on the block's rows.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block.
  - `target`: The [`AbstractForecastTarget`](@ref) the forward target is taken over.
  - $(arg_dict[:rf_horizon])
  - $(arg_dict[:rf_lag])
  - `step`: Number of observations between two evaluation dates.
  - `min_count`: Least number of assets a cross-section needs before a statistic of it is reported.
  - `ppy`: Periods per year.

# Validation

  - The target history has the shape of `alpha`. Raises a `DimensionMismatch`.
  - The rules of [`forecast_target_history`](@ref) and of [`forecast_evaluation`](@ref).

# Returns

  - `fe::ForecastEvaluationResult`: The pairing and the parameters that produced it.

# Related

  - [`forecast_evaluation`](@ref)
  - [`forecast_target_history`](@ref)
  - [`forward_mean_returns`](@ref)
"""
function forecast_evaluation_pairing(alpha::MatNum, rd::ReturnsResult,
                                     csfm::CrossSectionalFactorModel;
                                     target::AbstractForecastTarget = IdiosyncraticTarget(),
                                     horizon::Integer = 1, lag::Integer = 1,
                                     step::Integer = horizon, min_count::Integer = 3,
                                     ppy::Number = 1)::ForecastEvaluationResult
    X = forecast_target_history(target, rd, csfm)
    @argcheck(size(X, 1) == size(alpha, 1) && size(X, 2) == size(alpha, 2),
              DimensionMismatch("the target history ($(size(X, 1))×$(size(X, 2))) must match the Return Forecast history ($(size(alpha, 1))×$(size(alpha, 2))). Hand the carrier and the block the forecast was fitted on."))
    y = forward_mean_returns(X, horizon, lag)
    return forecast_evaluation(alpha, y; target = target, horizon = horizon, lag = lag,
                               step = step, min_count = min_count, ppy = ppy)
end
"""
    forecast_evaluation(alpha::MatNum, y::MatNum;
                        target::AbstractForecastTarget = IdiosyncraticTarget(),
                        horizon::Integer = 1, lag::Integer = 1, step::Integer = horizon,
                        min_count::Integer = 3, ppy::Number = 1) -> ForecastEvaluationResult
    forecast_evaluation(rfr::AbstractReturnForecastResult, rd::ReturnsResult,
                        csfm::CrossSectionalFactorModel;
                        target::AbstractForecastTarget = IdiosyncraticTarget(),
                        horizon::Integer = 1, lag::Integer = 1, step::Integer = horizon,
                        min_count::Integer = 3, ppy::Number = 1) -> ForecastEvaluationResult
    forecast_evaluation(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult,
                        csfm::CrossSectionalFactorModel;
                        target::AbstractForecastTarget = IdiosyncraticTarget(),
                        horizon::Integer = 1, lag::Integer = 1, step::Integer = horizon,
                        min_count::Integer = 3, ppy::Number = 1) -> ForecastEvaluationResult

Pair a Return Forecast with the forward target it is answerable for, out of sample.

This is the bottom of the evaluation hierarchy. The bare method takes the two matrices and computes nothing but the evaluation dates, so every statistic above it is testable without a fit, and a caller scores a forecast the library did not produce. The Result method reads the forecast history off a fitted member and builds the target from the carrier and the block, so a caller who holds a Result writes one call. The Estimator method asks [`forecast_history`](@ref) for the history instead, which refits a member that publishes none, so every member the family ships is evaluable through it.

# Algorithm

 1. For the Result method, read `hist` through [`forecast_evaluation_history`](@ref), which refuses a member that carries none. For the Estimator method, build the history through [`forecast_history`](@ref) at the evaluation's own `step`, so a refit lands on every date the evaluation scores.
 2. For both, build the target history through [`forecast_target_history`](@ref) and take its forward mean with [`forward_mean_returns`](@ref), in [`forecast_evaluation_pairing`](@ref).
 3. Find the evaluation dates with [`forecast_evaluation_dates`](@ref).
 4. Collect the pair, the dates and the parameters into a [`ForecastEvaluationResult`](@ref).

# Arguments

  - `alpha`: Return Forecast history `observations × assets`, in return units.
  - `y`: Forward target `observations × assets`, on the same axis as `alpha`. The bare method takes it already matured, and records `horizon` and `lag` as the parameters that matured it.
  - `rfr`: A fitted Return Forecast Result.
  - `rfe`: A Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block.
  - `target`: The [`AbstractForecastTarget`](@ref) the forward target is taken over. The default is the idiosyncratic return, which is the component a fitted member forecasts.
  - $(arg_dict[:rf_horizon])
  - $(arg_dict[:rf_lag])
  - `step`: Number of observations between two evaluation dates. The default of `horizon` gives forward windows that do not overlap.
  - `min_count`: Least number of assets a cross-section needs before a statistic of it is reported. It is carried rather than applied here, because the pairing is the same whatever the threshold.
  - `ppy`: Periods per year. `252` annualises a daily fit, and the default of `1` reports the statistics per period. It is carried rather than applied here, and the verbs above map it onto `performance_summary`'s `periods_per_year`.

# Validation

  - `!isempty(alpha)`. Raises an [`IsEmptyError`](@ref).
  - `size(alpha) == size(y)`. Raises a `DimensionMismatch`.
  - `horizon >= 1`, `lag >= 0`, `step >= 1`, `min_count >= 1` and `ppy > 0`. Raise a `DomainError`.
  - The rules of [`forecast_evaluation_dates`](@ref).
  - For the Result method, the rules of [`forecast_evaluation_history`](@ref) and of [`forecast_evaluation_pairing`](@ref).
  - For the Estimator method, the rules of [`forecast_history`](@ref) and of [`forecast_evaluation_pairing`](@ref).

# Returns

  - `fe::ForecastEvaluationResult`: The pairing and the parameters that produced it.

# Examples

```jldoctest
julia> alpha = [1.0 2.0; 2.0 1.0; 3.0 4.0; 4.0 3.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> fe = forecast_evaluation(alpha, y);

julia> fe.dates
3-element Vector{Int64}:
 1
 2
 3
```

# Related

  - [`ForecastEvaluationResult`](@ref)
  - [`forecast_evaluation_dates`](@ref)
  - [`forecast_evaluation_history`](@ref)
  - [`forecast_evaluation_pairing`](@ref)
  - [`forecast_history`](@ref)
  - [`forecast_target_history`](@ref)
  - [`forward_mean_returns`](@ref)
"""
function forecast_evaluation(alpha::MatNum, y::MatNum;
                             target::AbstractForecastTarget = IdiosyncraticTarget(),
                             horizon::Integer = 1, lag::Integer = 1,
                             step::Integer = horizon, min_count::Integer = 3,
                             ppy::Number = 1)::ForecastEvaluationResult
    @argcheck(!isempty(alpha), IsEmptyError("alpha cannot be empty"))
    @argcheck(size(alpha, 1) == size(y, 1) && size(alpha, 2) == size(y, 2),
              DimensionMismatch("y ($(size(y, 1))×$(size(y, 2))) must match alpha ($(size(alpha, 1))×$(size(alpha, 2)))"))
    @argcheck(horizon >= one(horizon), DomainError(horizon, "horizon must be >= 1"))
    @argcheck(lag >= zero(lag), DomainError(lag, "lag must be >= 0"))
    @argcheck(step >= one(step), DomainError(step, "step must be >= 1"))
    @argcheck(min_count >= one(min_count), DomainError(min_count, "min_count must be >= 1"))
    @argcheck(ppy > zero(ppy), DomainError(ppy, "ppy must be positive"))
    dates = forecast_evaluation_dates(alpha, y, step)
    return ForecastEvaluationResult(alpha, y, dates, target, horizon, lag, step, min_count,
                                    ppy)
end
function forecast_evaluation(rfr::AbstractReturnForecastResult, rd::ReturnsResult,
                             csfm::CrossSectionalFactorModel;
                             target::AbstractForecastTarget = IdiosyncraticTarget(),
                             horizon::Integer = 1, lag::Integer = 1,
                             step::Integer = horizon, min_count::Integer = 3,
                             ppy::Number = 1)::ForecastEvaluationResult
    return forecast_evaluation_pairing(forecast_evaluation_history(rfr), rd, csfm;
                                       target = target, horizon = horizon, lag = lag,
                                       step = step, min_count = min_count, ppy = ppy)
end
function forecast_evaluation(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult,
                             csfm::CrossSectionalFactorModel;
                             target::AbstractForecastTarget = IdiosyncraticTarget(),
                             horizon::Integer = 1, lag::Integer = 1,
                             step::Integer = horizon, min_count::Integer = 3,
                             ppy::Number = 1)::ForecastEvaluationResult
    return forecast_evaluation_pairing(forecast_history(rfe, rd, csfm; step = step), rd,
                                       csfm; target = target, horizon = horizon, lag = lag,
                                       step = step, min_count = min_count, ppy = ppy)
end

export IdiosyncraticTarget, AssetReturnTarget, PanelFieldTarget, ForecastEvaluationResult,
       forecast_evaluation
