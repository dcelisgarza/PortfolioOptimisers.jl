"""
$(DocStringExtensions.TYPEDEF)

Holds the forecaster of a forecast-reading rule, with the Partial Fit State of the rows that it folded.

A rule seeds this carrier when its forecaster has an exact fold. [`SimpleExpectedReturns`](@ref), [`ExpWeightedExpectedReturns`](@ref), a [`PriceLevelExpectedReturns`](@ref) over a folding statistic and a [`PriorExpectedReturns`](@ref) over a prior that folds all have one. [`supports_partial_fit`](@ref) answers the question once, at the seed.

The rule folds the forecaster on every row and reads the forecast from its state, so the head holds one row for it. The fold reads that row as the head received it, with its gaps and its active mask, and not the finite price relative of the step. The rule fits a forecaster with no exact fold again on the rows that the head holds, and its carrier is then `nothing`.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ForecastReversion`](@ref)
  - [`ForecastTracking`](@ref)
  - [`forecast_relative`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct ForecasterState <: AbstractPartialFitState
    """
    The forecaster, which carries the state of the rows that it folded.
    """
    me
end
function merge_states(::ForecasterState, ::ForecasterState)
    return throw(ArgumentError("a `ForecasterState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::ForecasterState)
    return ForecasterState(copy_forecaster(x.me))
end
function port_opt_view(x::ForecasterState, i, args...)
    return ForecasterState(port_opt_view(x.me, i, args...))
end
"""
    copy_forecaster(me::AbstractExpectedReturnsEstimator)
    copy_forecaster(me::PriorExpectedReturns)

Copies a folded forecaster, so that its state shares no array with the original.

The first method copies the `cache` field through the [`Base.copy`](@ref) of the state, and returns an estimator with no `cache` unchanged. The second method copies the prior of the adapter with [`copy_forecaster_prior`](@ref).

# Related

  - [`ForecasterState`](@ref)
"""
function copy_forecaster(me::AbstractExpectedReturnsEstimator)
    if hasfield(typeof(me), :cache) && !isnothing(getfield(me, :cache))
        return Accessors.@set me.cache = copy(me.cache)
    end
    return me
end
function copy_forecaster(me::PriorExpectedReturns)
    return PriorExpectedReturns(; pe = copy_forecaster_prior(me.pe))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a folded prior, so that its state shares no array with the original.

[`copy_forecaster`](@ref) calls it for the prior of a [`PriorExpectedReturns`](@ref). It copies the `cache` field through the [`Base.copy`](@ref) of the state, and returns a prior with no `cache` unchanged.

# Related

  - [`copy_forecaster`](@ref)
"""
function copy_forecaster_prior(pe::AbstractPriorEstimator)
    if hasfield(typeof(pe), :cache) && !isnothing(getfield(pe, :cache))
        return Accessors.@set pe.cache = copy(pe.cache)
    end
    return pe
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an [`Online`](@ref) wrapper in the forecaster slot of a rule.

The rows buffer of the head already holds the rows that the forecaster reads. A second buffer would hold them twice.

# Validation

  - `me` is not an [`Online`](@ref), and holds none at any depth. An `ArgumentError` is thrown otherwise, and it names the path to the wrapper.

# Returns

  - `nothing`.

# Related

  - [`ForecastReversion`](@ref)
  - [`ForecastTracking`](@ref)
  - [`Online`](@ref)
"""
function assert_forecaster(me::AbstractExpectedReturnsEstimator)::Nothing
    path = online_wrapper_path(me)
    @argcheck(isnothing(path),
              ArgumentError("`$(typeof(me).name.name)` holds an `Online` at `$(path)`, and the forecaster slot of an online selection rule refuses it: the head's rows buffer is the buffer, capped by what the rule tree reads, and a second one would hold the rows twice. Hand the slot the estimator the wrapper holds; one with an exact fold is folded on the Rule State, and one without is refit from the head's rows."))
    return nothing
end
function assert_forecaster(::Online)::Nothing
    return throw(ArgumentError("`Online` does not wrap the forecaster of an online selection rule: the head's rows buffer is the buffer, capped by what the rule tree reads, and a second one would hold the rows twice. Hand the slot the estimator the wrapper holds; one with an exact fold is folded on the Rule State, and one without is refit from the head's rows."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the carrier that a forecast-reading rule seeds.

The carrier is a [`ForecasterState`](@ref) over a forecaster that folds, and `nothing` for a forecaster that the rule fits again from the rows of the head.

# Validation

  - A forecaster that folds carries no partial-fit state at the seed. An `ArgumentError` is thrown otherwise. The head starts cold, and a state folded elsewhere would make the forecast of the first row read rows that the head never saw.

# Related

  - [`ForecasterState`](@ref)
  - [`rule_state_seed`](@ref)
  - [`supports_partial_fit`](@ref)
"""
function forecaster_seed(me::AbstractExpectedReturnsEstimator)
    if !supports_partial_fit(me)
        return nothing
    end
    @argcheck(isnothing(online_entry_state(me)),
              ArgumentError("the forecaster `$(typeof(me).name.name)` enters the online selection head carrying a partial-fit state, and the head starts cold. Hand the slot the estimator with its `cache` at `nothing`."))
    return ForecasterState(me)
end
"""
    forecast_relative(me::AbstractExpectedReturnsEstimator, st::Nothing, x::AbstractVector, rows)
    forecast_relative(me::AbstractExpectedReturnsEstimator, st::ForecasterState, x::AbstractVector, rows)

Advances the forecaster of a rule by the row `x`, and returns its Price Relative Forecast.

The fold arm folds a forecaster on a [`ForecasterState`](@ref) on the row, and reads the forecast from its state. The refit arm fits a forecaster with no carrier again on the rows that the head holds, which include the row. Outside the head, with no rows carrier, the fold arm folds the finite `x .- 1`, and the refit arm fits the row alone.

Both arms read the rows carrier of the head as the batch verb reads it, with the gaps and the active mask. The refit arm reduces to the Coverage Universe of the window. So a plain forecaster gives `NaN`, and the step holds the asset, when the asset has a gap anywhere in the window. A mask-aware forecaster gives the forecast from the rows that it has.

The fold arm folds the last row of the carrier, which is the current row as the head received it, under its active mask. The running statistic of a plain moment forecaster is then `NaN` from the first gap on, which is the Coverage Universe of the prefix. A mask-aware forecaster freezes, resets or admits the asset again by its own policy. A folding price-level statistic resets the asset that the mask turns off, and gives `NaN` until it has folded a level, so a relisted asset starts cold.

# Mathematical definition

```math
\\begin{align}
\\hat{x}_{t+1, i} &= \\begin{cases} 1 + \\hat{\\mu}_i & \\text{if } \\hat{\\mu}_i \\text{ is finite}\\,, \\\\ 1 & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc])
  - ``\\hat{\\mu}_i``: Expected return of asset ``i`` that the forecaster estimates after the row of period ``t``.
  - $(math_dict[:t_period])

A forecast of one in every asset is flat, and the step of every rule holds on a flat forecast. The refit arm also gives a flat forecast while it has fewer rows than [`forecast_min_rows`](@ref).

# Algorithm

With no carrier, the refit arm:

 1. Take the returns `X` and the Asset Panel `pnl` with [`refit_rows`](@ref).
 2. When `X` has fewer rows than [`forecast_min_rows`](@ref), return `nothing` and a flat forecast. Stop.
 3. Estimate the expected returns with `mean(me, X, pnl; dims = 1)`.
 4. Return `nothing` and one plus the expected returns, through [`flat_where_undefined`](@ref).

With a [`ForecasterState`](@ref), the fold arm:

 1. Fold the forecaster `st.me` with [`partial_fit!`](@ref). With a rows carrier, fold the last row of `rows.X` under the mask of [`last_active_mask`](@ref). Without one, fold `x .- 1`.
 2. Read the expected returns from the state of the folded forecaster.
 3. Return the new [`ForecasterState`](@ref) and one plus the expected returns, through [`flat_where_undefined`](@ref).

# Arguments

  - `me`: The forecaster on the slot of the rule.
  - `st`: The carrier of the rule, or `nothing`.
  - `x`: The price relative of the row.
  - `rows`: The rows carrier that the head holds through the row, a [`ReturnsResult`](@ref), or `nothing`.

# Returns

  - `(st', x̂)::Tuple`: The carrier after the row, and the forecast as a new vector.

# Related

  - [`ForecasterState`](@ref)
  - [`ForecastReversion`](@ref)
  - [`ForecastTracking`](@ref)
  - [`refit_rows`](@ref)
"""
function forecast_relative(me::AbstractExpectedReturnsEstimator, ::Nothing,
                           x::AbstractVector, rows::Option{<:ReturnsResult})
    X, pnl = refit_rows(rows, x)
    if size(X, 1) < forecast_min_rows(me)
        return nothing, fill(one(eltype(x)), length(x))
    end
    return nothing,
           flat_where_undefined(one(eltype(x)) .+
                                vec(Statistics.mean(me, X, pnl; dims = 1)))
end
function forecast_relative(::AbstractExpectedReturnsEstimator, st::ForecasterState,
                           x::AbstractVector, ::Nothing)
    me = partial_fit!(st.me, x .- one(eltype(x)))
    return ForecasterState(me),
           flat_where_undefined(one(eltype(x)) .+ vec(Statistics.mean(me)))
end
function forecast_relative(::AbstractExpectedReturnsEstimator, st::ForecasterState,
                           x::AbstractVector, rows::ReturnsResult)
    me = partial_fit!(st.me, view(rows.X, size(rows.X, 1), :);
                      active_mask = last_active_mask(rows.pnl))
    return ForecasterState(me),
           flat_where_undefined(one(eltype(x)) .+ vec(Statistics.mean(me)))
end
"""
    last_active_mask(pnl::Nothing)
    last_active_mask(pnl::AssetPanel)

Returns the active mask of the last row of an Asset Panel, or `nothing` when there is no panel or the panel is static.

The fold arm of [`forecast_relative`](@ref) folds the current row under this mask.

# Related

  - [`forecast_relative`](@ref)
  - [`panel_is_static`](@ref)
"""
function last_active_mask(::Nothing)
    return nothing
end
function last_active_mask(pnl::AssetPanel)
    if panel_is_static(pnl)
        return nothing
    end
    return view(pnl.amsk, size(pnl.amsk, 1), :)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the number of rows that a forecaster needs before the refit arm of [`forecast_relative`](@ref) reads it.

Below this number the refit arm gives a flat forecast.

# Mathematical definition

```math
\\begin{align}
n_{\\min} &= \\max\\left(n_2,\\, n_{\\mathrm{fit}}\\right)\\,.
\\end{align}
```

Where:

  - ``n_{\\min}``: Least number of rows of the refit arm.
  - ``n_2``: Two when the estimator tree holds a covariance, variance or prior estimator, whose second moment is undefined on one row, and one otherwise. [`holds_second_moment`](@ref) decides it.
  - ``n_{\\mathrm{fit}}``: Least number of rows that the estimator states through [`fit_min_rows`](@ref).

# Related

  - [`forecast_relative`](@ref)
  - [`fit_min_rows`](@ref)
"""
function forecast_min_rows(me::AbstractExpectedReturnsEstimator)
    return max(holds_second_moment(me) ? 2 : 1, fit_min_rows(me))
end
"""
    holds_second_moment(est)

Returns whether an estimator tree holds a covariance, variance or prior estimator at any depth.

# Algorithm

 1. When `est` is a covariance, variance or prior estimator, or a `StatsBase.CovarianceEstimator`, return `true`. Stop.
 2. When `est` is another estimator, call the function on each field that [`estimator_fields`](@ref) names, and return `true` when any call returns `true`.
 3. Return `false` for a value that is not an estimator.

# Related

  - [`forecast_min_rows`](@ref)
"""
function holds_second_moment(est::Union{<:AbstractEstimator,
                                        <:StatsBase.CovarianceEstimator})
    if isa(est,
           Union{<:AbstractCovarianceEstimator, <:StatsBase.CovarianceEstimator,
                 <:AbstractVarianceEstimator, <:AbstractPriorEstimator})
        return true
    end
    return any(f -> holds_second_moment(getfield(est, f)), estimator_fields(est))
end
function holds_second_moment(::Any)
    return false
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads each non-finite entry of a Price Relative Forecast as one, so the step holds that asset.

A forecaster whose estimate is undefined over the first rows, such as a variance over one observation, therefore holds until the estimate is defined.

# Related

  - [`forecast_relative`](@ref)
"""
function flat_where_undefined(xhat::AbstractVector)
    return [isfinite(v) ? v : one(v) for v in xhat]
end
"""
    refit_rows(rows::ReturnsResult, x::AbstractVector)
    refit_rows(rows::Nothing, x::AbstractVector)

Returns the returns that the refit arm fits a forecaster on, and the Asset Panel that explains them.

With a rows carrier these are its `X` and its `pnl`. Without one, they are the current row as a matrix of one row, and no panel.

# Related

  - [`forecast_relative`](@ref)
"""
function refit_rows(rows::ReturnsResult, ::AbstractVector)
    return rows.X, rows.pnl
end
function refit_rows(::Nothing, x::AbstractVector)
    return reshape(x .- one(eltype(x)), 1, :), nothing
end
"""
    scale_relative(scale::Nothing, x::AbstractVector, rows)
    scale_relative(scale::AbstractPriceLevelStatistic, x::AbstractVector, rows)

Returns the diagonal of the preconditioner of a [`ForecastReversion`](@ref) step, or `nothing` for the identity.

The diagonal is the Price Relative Forecast of the `scale` statistic, fit on the rows that the head holds, or on the current row alone without a rows carrier. An entry that the statistic cannot give is one, through [`flat_where_undefined`](@ref), so the step on that asset is not scaled.

# Related

  - [`ForecastReversion`](@ref)
  - [`ReweightedPriceRelativeTracking`](@ref)
"""
function scale_relative(::Nothing, ::AbstractVector, ::Any)
    return nothing
end
function scale_relative(scale::AbstractPriceLevelStatistic, x::AbstractVector,
                        rows::Option{<:ReturnsResult})
    me = PriceLevelExpectedReturns(; alg = scale)
    X, pnl = refit_rows(rows, x)
    return flat_where_undefined(one(eltype(x)) .+
                                vec(Statistics.mean(me, X, pnl; dims = 1)))
end
"""
    scale_rows(scale::Nothing)
    scale_rows(scale::AbstractPriceLevelStatistic)

Returns the number of rows that a `scale` statistic reads at a step, and `0` when there is no statistic.

# Related

  - [`ForecastReversion`](@ref)
  - [`rows_needed`](@ref)
"""
function scale_rows(::Nothing)
    return 0
end
function scale_rows(scale::AbstractPriceLevelStatistic)
    return window_rows(scale)
end
"""
$(DocStringExtensions.TYPEDEF)

Moves the allocation by the least amount that gives a forecast return of at least `eps`, then projects.

Five papers share this passive-aggressive step: the moving average reversion of Li and Hoi (2012), the robust median reversion of Huang, Zhou, Li, Hoi and Zhou (2016), the reweighted price relative tracking of Lai, Yang, Fang and Wu (2020), the Gaussian weighting reversion of Cai and Ye (2019) and the local adaptive learning of Guan and An (2019). The papers differ only in the statistic on the `me` slot. [`MovingAverageReversion`](@ref), [`ExponentialMovingAverageReversion`](@ref), [`RobustMedianReversion`](@ref), [`ReweightedPriceRelativeTracking`](@ref), [`GaussianWeightingReversion`](@ref) and [`LocalAdaptiveLearning`](@ref) fill the slot with the statistic of each paper.

Any expected-returns estimator can sit on the slot, for example a mean of past returns, a shrunk mean, or the mean of a Prior through [`PriorExpectedReturns`](@ref). [`forecast_relative`](@ref) advances it. The rule fits a forecaster that does not fold again on every row that the head holds, because [`rows_needed`](@ref) returns `nothing` for it. So each step costs one fit over the whole prefix.

The rule bets on the reversion that the forecast encodes. It gains on a market that reverts and loses on a market that trends.

# Mathematical definition

```math
\\begin{align}
\\lambda_t &= \\max\\left(0, \\frac{\\epsilon - \\langle \\boldsymbol{w}_t, \\hat{\\boldsymbol{x}}_{t+1} \\rangle}{\\lVert \\tilde{\\boldsymbol{x}}_{t+1} \\rVert^2}\\right)\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\boldsymbol{w}_t + \\lambda_t \\mathbf{D}_{t+1} \\tilde{\\boldsymbol{x}}_{t+1} \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc])
  - $(math_dict[:xtilde_fc])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:t_period])
  - ``\\epsilon``: Target of the forecast return ``\\langle \\boldsymbol{w}, \\hat{\\boldsymbol{x}}_{t+1} \\rangle``, the `eps` field.
  - ``\\lambda_t``: Step multiplier, zero when ``\\tilde{\\boldsymbol{x}}_{t+1} = \\boldsymbol{0}``.
  - ``\\mathbf{D}_{t+1}``: Diagonal preconditioner. It is the identity when `scale` is `nothing`, and otherwise the diagonal matrix of the Price Relative Forecast of the `scale` statistic.
  - $(math_dict[:Proj_W_euclid])

Without a preconditioner, the vector inside the projection solves the programme that the papers state, ``\\underset{\\boldsymbol{w}}{\\min} \\; \\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{w}_t \\rVert^2`` subject to ``\\langle \\boldsymbol{w}, \\hat{\\boldsymbol{x}}_{t+1} \\rangle \\geq \\epsilon`` and ``\\boldsymbol{1}^\\intercal \\boldsymbol{w} = 1``. The projection then restores the bounds. The reweighted price relative tracking keeps the same ``\\lambda_t`` with the preconditioner. In general its vector then leaves the budget hyperplane and misses the target ``\\epsilon``, and the projection restores the budget.

The sign of the step is opposite to the sign of [`PassiveAggressiveMeanReversion`](@ref), because ``\\hat{\\boldsymbol{x}}_{t+1}`` forecasts the next price relative, where that rule reads the last one.

# Algorithm

[`online_update!`](@ref) runs these steps at the row `x` of the period, from the allocation `w`.

 1. Advance the forecaster with [`forecast_relative`](@ref), which gives the carrier `st` and the forecast `xhat`.
 2. Centre the forecast, which gives `dev`, and take its squared norm `denom`.
 3. Form the multiplier `lam`: zero when `denom` is zero, and ``\\lambda_t`` otherwise.
 4. Form the diagonal `D` of the preconditioner with [`scale_relative`](@ref), or `nothing`.
 5. Form the raw step `q = w .+ lam .* dev`, or `q = w .+ lam .* D .* dev` with a preconditioner.
 6. Project `q` onto the set with [`project`](@ref), with the [`price_adjusted_allocation`](@ref) of `w` after the row. Return `st` and the new allocation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForecastReversion(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(),
        eps::Real = 10,
        scale::Option{<:AbstractPriceLevelStatistic} = nothing,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ForecastReversion

Keywords correspond to the struct's fields. The rule needs the larger of the rows that the forecaster reads and the rows that the scale reads, so the head holds the rows of both. A forecaster that folds reads the current row alone.

## Validation

  - `eps > 0`. A `DomainError` is thrown otherwise.
  - `me` is not an [`Online`](@ref) wrapper, and holds none. An `ArgumentError` is thrown otherwise, and it names the wrapper.

# Examples

```jldoctest
julia> ForecastReversion()
ForecastReversion
     me ┼ PriceLevelExpectedReturns
        │   alg ┼ MovingAverage
        │       │   window ┴ Int64: 5
    eps ┼ Int64: 10
  scale ┼ nothing
   proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`MovingAverageReversion`](@ref)
  - [`RobustMedianReversion`](@ref)
  - [`ExponentialMovingAverageReversion`](@ref)
  - [`ReweightedPriceRelativeTracking`](@ref)
  - [`GaussianWeightingReversion`](@ref)
  - [`LocalAdaptiveLearning`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)
  - [`ForecastTracking`](@ref)
  - [`forecast_relative`](@ref)

# References

  - $(ref_dict[:lihoi2012])
  - $(ref_dict[:huang2016])
  - $(ref_dict[:lai2018rprt])
  - $(ref_dict[:liluoxu2023])
  - $(ref_dict[:caiye2019])
  - $(ref_dict[:guanan2019])
"""
struct ForecastReversion{T1 <: AbstractExpectedReturnsEstimator, T2 <: Real,
                         T3 <: Option{<:AbstractPriceLevelStatistic},
                         T4 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:forecaster])
    """
    me::T1
    """
    Target of the forecast return. A larger target gives a longer step.
    """
    eps::T2
    """
    The statistic whose Price Relative Forecast preconditions the step, or `nothing` for the identity.
    """
    scale::T3
    """
    $(field_dict[:proj])
    """
    proj::T4
    function ForecastReversion(me::AbstractExpectedReturnsEstimator, eps::Real,
                               scale::Option{<:AbstractPriceLevelStatistic},
                               proj::EuclideanProjection)
        assert_forecaster(me)
        @argcheck(eps > zero(eps), DomainError(eps, "eps must be positive"))
        return new{typeof(me), typeof(eps), typeof(scale), typeof(proj)}(me, eps, scale,
                                                                         proj)
    end
end
function ForecastReversion(me::Online, ::Real, ::Option{<:AbstractPriceLevelStatistic},
                           ::EuclideanProjection)
    return assert_forecaster(me)
end
function ForecastReversion(;
                           me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(),
                           eps::Real = 10,
                           scale::Option{<:AbstractPriceLevelStatistic} = nothing,
                           proj::EuclideanProjection = EuclideanProjection())::ForecastReversion
    return ForecastReversion(me, eps, scale, proj)
end
function port_opt_view(alg::ForecastReversion, i, args...)
    return ForecastReversion(; me = port_opt_view(alg.me, i, args...), eps = alg.eps,
                             scale = alg.scale, proj = alg.proj)
end
function rows_needed(alg::ForecastReversion)
    return rows_needed_max(rows_needed(alg.me), scale_rows(alg.scale))
end
function rule_state_seed(alg::ForecastReversion, ::AbstractVector)
    return forecaster_seed(alg.me)
end
function online_update!(alg::ForecastReversion, st, w::AbstractVector, x::AbstractVector,
                        rows, set::AbstractAllocationSet)
    st, xhat = forecast_relative(alg.me, st, x, rows)
    dev = xhat .- Statistics.mean(xhat)
    denom = sum(abs2, dev)
    lam = if iszero(denom)
        zero(denom)
    else
        max(zero(denom), (alg.eps - LinearAlgebra.dot(w, xhat)) / denom)
    end
    D = scale_relative(alg.scale, x, rows)
    q = isnothing(D) ? w .+ lam .* dev : w .+ lam .* D .* dev
    return st, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
    MovingAverageReversion(; window::Integer = 5, eps::Real = 10, proj::EuclideanProjection = EuclideanProjection())

Builds the on-line moving average reversion of Li and Hoi (2012), OLMAR.

It is a [`ForecastReversion`](@ref) whose forecast is the [`MovingAverage`](@ref) of the last `window` price levels over the last price. The defaults `window = 5` and `eps = 10` are the paper's. `window` is the horizon of the reversion, and a search over `"alg.me.alg.window"` tunes it.

The algorithm of the 2012 paper takes `eps > 1` and `window >= 3`, and the algorithm of Li, Hoi, Sahoo and Liu (2015) takes `window >= 2`. At two levels the average is the mean of the current price and the previous price. The step is defined for any positive `eps`, so the constructor admits `eps > 0` and `window >= 2`. With `eps <= 1` the step moves only when the forecast return of the held allocation is below `eps`, which is a forecast loss.

# Examples

```jldoctest
julia> MovingAverageReversion(; window = 3)
ForecastReversion
     me ┼ PriceLevelExpectedReturns
        │   alg ┼ MovingAverage
        │       │   window ┴ Int64: 3
    eps ┼ Int64: 10
  scale ┼ nothing
   proj ┴ EuclideanProjection()
```

# Related

  - [`ForecastReversion`](@ref)
  - [`MovingAverage`](@ref)
  - [`RobustMedianReversion`](@ref)
  - [`ExponentialMovingAverageReversion`](@ref)

# References

  - $(ref_dict[:lihoi2012])
  - $(ref_dict[:li2015olmar])
"""
function MovingAverageReversion(; window::Integer = 5, eps::Real = 10,
                                proj::EuclideanProjection = EuclideanProjection())::ForecastReversion
    return ForecastReversion(;
                             me = PriceLevelExpectedReturns(;
                                                            alg = MovingAverage(;
                                                                                window = window)),
                             eps = eps, proj = proj)
end
"""
    ExponentialMovingAverageReversion(; alpha::Real = 0.5, eps::Real = 10, proj::EuclideanProjection = EuclideanProjection())

Builds the second form of the on-line moving average reversion of Li, Hoi, Sahoo and Liu (2015), OLMAR-2.

It is a [`ForecastReversion`](@ref) whose forecast is the [`ExponentialMovingAverage`](@ref) of the price levels over the last price. `eps = 10` is the paper's. The paper gives no default `alpha`, and its sensitivity study scans `alpha` from 0 to 1. The library takes `0.5`, the middle of that range.

The average starts at the first price, as the expansion of the paper does, so the forecast after one level is one. The statistic folds, so the rule carries one vector and the head holds only the current row.

# Examples

```jldoctest
julia> ExponentialMovingAverageReversion()
ForecastReversion
     me ┼ PriceLevelExpectedReturns
        │   alg ┼ ExponentialMovingAverage
        │       │   alpha ┴ Float64: 0.5
    eps ┼ Int64: 10
  scale ┼ nothing
   proj ┴ EuclideanProjection()
```

# Related

  - [`ForecastReversion`](@ref)
  - [`ExponentialMovingAverage`](@ref)
  - [`MovingAverageReversion`](@ref)

# References

  - $(ref_dict[:li2015olmar])
"""
function ExponentialMovingAverageReversion(; alpha::Real = 0.5, eps::Real = 10,
                                           proj::EuclideanProjection = EuclideanProjection())::ForecastReversion
    return ForecastReversion(;
                             me = PriceLevelExpectedReturns(;
                                                            alg = ExponentialMovingAverage(;
                                                                                           alpha = alpha)),
                             eps = eps, proj = proj)
end
"""
    RobustMedianReversion(; window::Integer = 5, eps::Real = 5, iters::Integer = 100, tol::Real = 1e-8, proj::EuclideanProjection = EuclideanProjection())

Builds the robust median reversion of Huang, Zhou, Li, Hoi and Zhou (2016), RMR.

It is a [`ForecastReversion`](@ref) whose forecast is the [`SpatialMedian`](@ref) of the last `window` price levels over the last price. The defaults `window = 5` and `eps = 5` are the paper's. One extreme price moves a mean without limit, and it moves a spatial median very little.

The library reads the median on the rebuilt price path, with the last level of each asset at one. Its iteration stops by a rule different from the rule of the paper. [`SpatialMedian`](@ref) states both differences.

# Examples

```jldoctest
julia> RobustMedianReversion(; window = 3)
ForecastReversion
     me ┼ PriceLevelExpectedReturns
        │   alg ┼ SpatialMedian
        │       │   window ┼ Int64: 3
        │       │    iters ┼ Int64: 100
        │       │      tol ┴ Float64: 1.0e-8
    eps ┼ Int64: 5
  scale ┼ nothing
   proj ┴ EuclideanProjection()
```

# Related

  - [`ForecastReversion`](@ref)
  - [`SpatialMedian`](@ref)
  - [`MovingAverageReversion`](@ref)

# References

  - $(ref_dict[:huang2016])
"""
function RobustMedianReversion(; window::Integer = 5, eps::Real = 5, iters::Integer = 100,
                               tol::Real = 1e-8,
                               proj::EuclideanProjection = EuclideanProjection())::ForecastReversion
    return ForecastReversion(;
                             me = PriceLevelExpectedReturns(;
                                                            alg = SpatialMedian(;
                                                                                window = window,
                                                                                iters = iters,
                                                                                tol = tol)),
                             eps = eps, proj = proj)
end
"""
    ReweightedPriceRelativeTracking(; window::Integer = 5, theta::Real = 0.8, eps::Real = 50, proj::EuclideanProjection = EuclideanProjection())

Builds the reweighted price relative tracking of Lai, Yang, Fang and Wu (2020), RPRT.

It is a [`ForecastReversion`](@ref) whose forecast is the [`ReweightedPriceRelative`](@ref) recursion. The [`MovingAverage`](@ref) forecast of the same `window` preconditions its step.

The paper is not open access. Li, Luo and Xu (2023, eqs. 8 to 11) restate its forecast and its step, and the MATLAB code that the authors publish runs the same recursion. The code seeds the forecast at one, and the restatement seeds it at the first price relative, which [`ReweightedPriceRelative`](@ref) follows. The defaults `theta = 0.8`, `eps = 50` and `window = 5` are the values of that code. In its first `window` periods the code preconditions with the last price relative, and the library uses the moving average of the levels that it has.

With `eps = 50` the forecast return is below the target at every step, so the step always moves. The projected answer is then almost one-hot, as the answers of the tracking rules at their defaults are.

# Examples

```jldoctest
julia> ReweightedPriceRelativeTracking()
ForecastReversion
     me ┼ PriceLevelExpectedReturns
        │   alg ┼ ReweightedPriceRelative
        │       │   theta ┴ Float64: 0.8
    eps ┼ Int64: 50
  scale ┼ MovingAverage
        │   window ┴ Int64: 5
   proj ┴ EuclideanProjection()
```

# Related

  - [`ForecastReversion`](@ref)
  - [`ReweightedPriceRelative`](@ref)
  - [`MovingAverage`](@ref)

# References

  - $(ref_dict[:lai2018rprt])
  - $(ref_dict[:liluoxu2023])
"""
function ReweightedPriceRelativeTracking(; window::Integer = 5, theta::Real = 0.8,
                                         eps::Real = 50,
                                         proj::EuclideanProjection = EuclideanProjection())::ForecastReversion
    return ForecastReversion(;
                             me = PriceLevelExpectedReturns(;
                                                            alg = ReweightedPriceRelative(;
                                                                                          theta = theta)),
                             eps = eps, scale = MovingAverage(; window = window),
                             proj = proj)
end
"""
    GaussianWeightingReversion(; tau::Real = 2.8, cutoff::Real = 0.005, eps::Real = 50, proj::EuclideanProjection = EuclideanProjection())

Builds the Gaussian weighting reversion of Cai and Ye (2019), GWR.

It is a [`ForecastReversion`](@ref) whose forecast is the [`GaussianWeightedDoubleEstimate`](@ref) over the last price. The defaults `tau = 2.8`, `cutoff = 0.005` and `eps = 50` are the paper's. The paper names the target ``\\delta`` and the cutoff ``\\epsilon``.

The adaptive variant of the paper, GWR-A, chooses `tau` online with a bandit over the reward. The library does not build it, because the family has no seam for a parameter that the reward chooses online.

# Examples

```jldoctest
julia> GaussianWeightingReversion()
ForecastReversion
     me ┼ PriceLevelExpectedReturns
        │   alg ┼ GaussianWeightedDoubleEstimate
        │       │      tau ┼ Float64: 2.8
        │       │   cutoff ┴ Float64: 0.005
    eps ┼ Int64: 50
  scale ┼ nothing
   proj ┴ EuclideanProjection()
```

# Related

  - [`ForecastReversion`](@ref)
  - [`GaussianWeightedDoubleEstimate`](@ref)

# References

  - $(ref_dict[:caiye2019])
"""
function GaussianWeightingReversion(; tau::Real = 2.8, cutoff::Real = 0.005, eps::Real = 50,
                                    proj::EuclideanProjection = EuclideanProjection())::ForecastReversion
    return ForecastReversion(;
                             me = PriceLevelExpectedReturns(;
                                                            alg = GaussianWeightedDoubleEstimate(;
                                                                                                 tau = tau,
                                                                                                 cutoff = cutoff)),
                             eps = eps, proj = proj)
end
"""
    LocalAdaptiveLearning(; window::Integer = 5, alpha::Real = 0.5, threshold::Real = 0.1, lambda::Real = 0, eps::Real = 10, proj::EuclideanProjection = EuclideanProjection())

Builds the local adaptive learning of Guan and An (2019), LOAD.

It is a [`ForecastReversion`](@ref) whose forecast is a [`TrendSwitch`](@ref) on the [`RegressionSlope`](@ref). An asset whose slope exceeds `threshold` takes its [`WindowPeak`](@ref), and every other asset takes its [`ExponentialMovingAverage`](@ref). The defaults `window = 5`, `alpha = 0.5` and `threshold = 0.1` are the paper's.

The paper states neither the target `eps` nor the ridge weight `lambda`. The defaults are the `eps = 10` of the moving average reversion and plain least squares. The paper fits the slope on the prices. The library fits it on the rebuilt price path, with the last level at one, so `threshold` compares with a slope in units of the current price. The exponential average reads the full history, as the recursion of the paper does, so the head holds every row for this rule.

# Examples

```jldoctest
julia> LocalAdaptiveLearning()
ForecastReversion
     me ┼ PriceLevelExpectedReturns
        │   alg ┼ TrendSwitch
        │       │      test ┼ RegressionSlope
        │       │           │      window ┼ Int64: 5
        │       │           │   threshold ┼ Float64: 0.1
        │       │           │      lambda ┴ Int64: 0
        │       │    rising ┼ WindowPeak
        │       │           │   window ┴ Int64: 5
        │       │      flat ┼ ExponentialMovingAverage
        │       │           │   alpha ┴ Float64: 0.5
        │       │   falling ┼ ExponentialMovingAverage
        │       │           │   alpha ┴ Float64: 0.5
    eps ┼ Int64: 10
  scale ┼ nothing
   proj ┴ EuclideanProjection()
```

# Related

  - [`ForecastReversion`](@ref)
  - [`TrendSwitch`](@ref)
  - [`RegressionSlope`](@ref)

# References

  - $(ref_dict[:guanan2019])
"""
function LocalAdaptiveLearning(; window::Integer = 5, alpha::Real = 0.5,
                               threshold::Real = 0.1, lambda::Real = 0, eps::Real = 10,
                               proj::EuclideanProjection = EuclideanProjection())::ForecastReversion
    ema = ExponentialMovingAverage(; alpha = alpha)
    alg = TrendSwitch(;
                      test = RegressionSlope(; window = window, threshold = threshold,
                                             lambda = lambda),
                      rising = WindowPeak(; window = window), flat = ema, falling = ema)
    return ForecastReversion(; me = PriceLevelExpectedReturns(; alg = alg), eps = eps,
                             proj = proj)
end
"""
$(DocStringExtensions.TYPEDEF)

Moves the allocation a fixed distance `eps` along the centred Price Relative Forecast, then projects.

Three papers share this step: the peak price tracking of Lai, Dai, Ren and Huang (2018), the adaptive input and composite trend representation of the same authors (2018), and the trend promote price tracing of Dai, Liang, Dai, Huang and Adnan (2022). It is the mirror of [`ForecastReversion`](@ref). `eps` is a step length and not a rate, so the forecast sets the direction alone.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\boldsymbol{w}_t + \\epsilon\\, \\frac{\\tilde{\\boldsymbol{x}}_{t+1}}{\\lVert \\tilde{\\boldsymbol{x}}_{t+1} \\rVert} \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc])
  - $(math_dict[:xtilde_fc])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:t_period])
  - ``\\epsilon``: Euclidean length of the step, the `eps` field.
  - $(math_dict[:Proj_W_euclid])

When ``\\tilde{\\boldsymbol{x}}_{t+1} = \\boldsymbol{0}`` the step is zero, and ``\\boldsymbol{w}_{t+1} = \\mathrm{Proj}_{\\mathcal{W}}(\\boldsymbol{w}_t)``.

The vector inside the projection maximises ``\\langle \\boldsymbol{w}, \\hat{\\boldsymbol{x}}_{t+1} \\rangle`` over the ball ``\\lVert \\boldsymbol{w} - \\boldsymbol{w}_t \\rVert \\leq \\epsilon`` in the budget hyperplane ``\\boldsymbol{1}^\\intercal \\boldsymbol{w} = 1``, which is the programme of the papers. The projection then restores the bounds, as it does after the reversion step.

On the simplex, the projection is one-hot on asset ``i`` whenever ``\\epsilon (u_i - u_j) \\geq 2`` for every other asset ``j``, with ``\\boldsymbol{u} = \\tilde{\\boldsymbol{x}}_{t+1} / \\lVert \\tilde{\\boldsymbol{x}}_{t+1} \\rVert``. At ``\\epsilon = 100`` that gap is 0.02, so at the defaults of the papers the step puts all the wealth on the asset with the largest forecast, except when the forecasts of two assets are almost equal.

# Algorithm

[`online_update!`](@ref) runs these steps at the row `x` of the period, from the allocation `w`.

 1. Advance the forecaster with [`forecast_relative`](@ref), which gives the carrier `st` and the forecast `xhat`.
 2. Centre the forecast, which gives `dev`, and take its norm `nrm`.
 3. Form the raw step `q`, which is `w` when `nrm` is zero and `w .+ eps .* dev ./ nrm` otherwise.
 4. Project `q` onto the set with [`project`](@ref), with the [`price_adjusted_allocation`](@ref) of `w` after the row. Return `st` and the new allocation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForecastTracking(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(; alg = WindowPeak()),
        eps::Real = 100,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ForecastTracking

Keywords correspond to the struct's fields. The rule needs the rows that `me` needs.

## Validation

  - `eps > 0`. A `DomainError` is thrown otherwise.
  - `me` is not an [`Online`](@ref) wrapper, and holds none. An `ArgumentError` is thrown otherwise, and it names the wrapper.

# Examples

```jldoctest
julia> ForecastTracking()
ForecastTracking
    me ┼ PriceLevelExpectedReturns
       │   alg ┼ WindowPeak
       │       │   window ┴ Int64: 5
   eps ┼ Int64: 100
  proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`ForecastReversion`](@ref)
  - [`PeakPriceTracking`](@ref)
  - [`AdaptiveInputCompositeTrend`](@ref)
  - [`TrendPromotePriceTracking`](@ref)
  - [`KernelTrendTracking`](@ref)
  - [`WindowPeak`](@ref)
  - [`forecast_relative`](@ref)

# References

  - $(ref_dict[:lai2018ppt])
  - $(ref_dict[:lai2018aictr])
  - $(ref_dict[:dai2022tppt])
"""
struct ForecastTracking{T1 <: AbstractExpectedReturnsEstimator, T2 <: Real,
                        T3 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:forecaster])
    """
    me::T1
    """
    Euclidean length of the step along the centred forecast.
    """
    eps::T2
    """
    $(field_dict[:proj])
    """
    proj::T3
    function ForecastTracking(me::AbstractExpectedReturnsEstimator, eps::Real,
                              proj::EuclideanProjection)
        assert_forecaster(me)
        @argcheck(eps > zero(eps), DomainError(eps, "eps must be positive"))
        return new{typeof(me), typeof(eps), typeof(proj)}(me, eps, proj)
    end
end
function ForecastTracking(me::Online, ::Real, ::EuclideanProjection)
    return assert_forecaster(me)
end
function ForecastTracking(;
                          me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(;
                                                                                                              alg = WindowPeak()),
                          eps::Real = 100,
                          proj::EuclideanProjection = EuclideanProjection())::ForecastTracking
    return ForecastTracking(me, eps, proj)
end
function port_opt_view(alg::ForecastTracking, i, args...)
    return ForecastTracking(; me = port_opt_view(alg.me, i, args...), eps = alg.eps,
                            proj = alg.proj)
end
function rows_needed(alg::ForecastTracking)
    return rows_needed(alg.me)
end
function rule_state_seed(alg::ForecastTracking, ::AbstractVector)
    return forecaster_seed(alg.me)
end
function online_update!(alg::ForecastTracking, st, w::AbstractVector, x::AbstractVector,
                        rows, set::AbstractAllocationSet)
    st, xhat = forecast_relative(alg.me, st, x, rows)
    dev = xhat .- Statistics.mean(xhat)
    nrm = LinearAlgebra.norm(dev)
    q = iszero(nrm) ? w : w .+ alg.eps .* dev ./ nrm
    return st, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
    PeakPriceTracking(; window::Integer = 5, eps::Real = 100, proj::EuclideanProjection = EuclideanProjection())

Builds the peak price tracking of Lai, Dai, Ren and Huang (2018), PPT.

It is a [`ForecastTracking`](@ref) whose forecast is the [`WindowPeak`](@ref) of the last `window` price levels over the last price. The defaults `window = 5` and `eps = 100` are the paper's, and the MATLAB code that the authors publish uses the same step.

At `eps = 100` the answer almost always puts all the wealth on the asset with the highest ratio of peak to last price. The sensitivity study of the paper (its Fig. 4) shows a flat wealth for `eps >= 50`, and this saturation is the cause.

# Examples

```jldoctest
julia> PeakPriceTracking()
ForecastTracking
    me ┼ PriceLevelExpectedReturns
       │   alg ┼ WindowPeak
       │       │   window ┴ Int64: 5
   eps ┼ Int64: 100
  proj ┴ EuclideanProjection()
```

# Related

  - [`ForecastTracking`](@ref)
  - [`WindowPeak`](@ref)

# References

  - $(ref_dict[:lai2018ppt])
"""
function PeakPriceTracking(; window::Integer = 5, eps::Real = 100,
                           proj::EuclideanProjection = EuclideanProjection())::ForecastTracking
    return ForecastTracking(;
                            me = PriceLevelExpectedReturns(;
                                                           alg = WindowPeak(;
                                                                            window = window)),
                            eps = eps, proj = proj)
end
"""
    AdaptiveInputCompositeTrend(; window::Integer = 5, alpha::Real = 0.5, sigma2::Real = 0.0025, eps::Real = 1000, proj::EuclideanProjection = EuclideanProjection())

Builds the adaptive input and composite trend representation of Lai, Dai, Ren and Huang (2018), AICTR.

It is a [`ForecastTracking`](@ref) whose forecast is the [`CompositeTrend`](@ref) of the [`MovingAverage`](@ref), the [`ExponentialMovingAverage`](@ref) and the [`WindowPeak`](@ref) over `window` levels. The defaults `window = 5`, `sigma2 = 0.0025` and `eps = 1000` are the paper's.

The paper gives no smoothing weight for its exponential average, so the library sets `alpha = 0.5`. The exponential average reads the full history, so the head holds every row for this rule.

# Examples

```jldoctest
julia> AdaptiveInputCompositeTrend()
ForecastTracking
    me ┼ PriceLevelExpectedReturns
       │   alg ┼ CompositeTrend
       │       │   trends ┼ 3-element Vector{PortfolioOptimisers.AbstractPriceLevelStatistic}
       │       │          │ MovingAverage ⋯
       │       │          │ ExponentialMovingAverage ⋯
       │       │          │ WindowPeak ⋯
       │       │   window ┼ Int64: 5
       │       │   sigma2 ┴ Float64: 0.0025
   eps ┼ Int64: 1000
  proj ┴ EuclideanProjection()
```

# Related

  - [`ForecastTracking`](@ref)
  - [`CompositeTrend`](@ref)

# References

  - $(ref_dict[:lai2018aictr])
"""
function AdaptiveInputCompositeTrend(; window::Integer = 5, alpha::Real = 0.5,
                                     sigma2::Real = 0.0025, eps::Real = 1000,
                                     proj::EuclideanProjection = EuclideanProjection())::ForecastTracking
    alg = CompositeTrend(;
                         trends = [MovingAverage(; window = window),
                                   ExponentialMovingAverage(; alpha = alpha),
                                   WindowPeak(; window = window)], window = window,
                         sigma2 = sigma2)
    return ForecastTracking(; me = PriceLevelExpectedReturns(; alg = alg), eps = eps,
                            proj = proj)
end
"""
    TrendPromotePriceTracking(; window::Integer = 5, alpha::Real = 0.5, eps::Real = 100, proj::EuclideanProjection = EuclideanProjection())

Builds the trend promote price tracing of Dai, Liang, Dai, Huang and Adnan (2022), TPPT.

It is a [`ForecastTracking`](@ref) whose forecast is a [`TrendSwitch`](@ref) on the [`PairwiseSlopeSum`](@ref). A rising asset takes its [`TruncatedExponentialMovingAverage`](@ref), a flat asset takes its current price, and a falling asset takes its [`WindowPeak`](@ref). The defaults `window = 5`, `alpha = 0.5` and `eps = 100` are the paper's.

The rising branch of the paper cannot be computed as printed, because it reads the price of the next period, which is the quantity that it forecasts. [`TruncatedExponentialMovingAverage`](@ref) states the reading that the library takes, so a parity test against the numbers of the paper is not possible.

# Examples

```jldoctest
julia> TrendPromotePriceTracking()
ForecastTracking
    me ┼ PriceLevelExpectedReturns
       │   alg ┼ TrendSwitch
       │       │      test ┼ PairwiseSlopeSum
       │       │           │   window ┴ Int64: 5
       │       │    rising ┼ TruncatedExponentialMovingAverage
       │       │           │    alpha ┼ Float64: 0.5
       │       │           │   window ┴ Int64: 5
       │       │      flat ┼ LaggedPrice
       │       │           │   lag ┴ Int64: 0
       │       │   falling ┼ WindowPeak
       │       │           │   window ┴ Int64: 5
   eps ┼ Int64: 100
  proj ┴ EuclideanProjection()
```

# Related

  - [`ForecastTracking`](@ref)
  - [`TrendSwitch`](@ref)
  - [`PairwiseSlopeSum`](@ref)

# References

  - $(ref_dict[:dai2022tppt])
"""
function TrendPromotePriceTracking(; window::Integer = 5, alpha::Real = 0.5,
                                   eps::Real = 100,
                                   proj::EuclideanProjection = EuclideanProjection())::ForecastTracking
    alg = TrendSwitch(; test = PairwiseSlopeSum(; window = window),
                      rising = TruncatedExponentialMovingAverage(; alpha = alpha,
                                                                 window = window),
                      flat = LaggedPrice(; lag = 0),
                      falling = WindowPeak(; window = window))
    return ForecastTracking(; me = PriceLevelExpectedReturns(; alg = alg), eps = eps,
                            proj = proj)
end
"""
$(DocStringExtensions.TYPEDEF)

Moves the allocation along the kernel-scaled centred Price Relative Forecast at a fixed rate `eta`, then projects.

This is the step of the kernel-based trend pattern tracking of Lai, Yang, Wu and Fang (2018). Unlike the step of [`ForecastTracking`](@ref), it is not normalised, so `eta` is a rate on the centred forecast and not a step length. [`forecast_relative`](@ref) advances the forecaster on `me`, and [`KernelTrendPatternTracking`](@ref) fills that slot with the [`KernelTrendPattern`](@ref) of the paper.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\boldsymbol{w}}_t &= \\boldsymbol{w}_t - \\bar{w}_t \\boldsymbol{1}\\,, \\\\
K_i &= \\exp\\left( -\\lvert \\tilde{w}_{t, i} - \\tilde{x}_{t+1, i} \\rvert^{1/q} \\right)\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\boldsymbol{w}_t + \\eta\\, \\mathbf{K} \\tilde{\\boldsymbol{x}}_{t+1} \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc])
  - $(math_dict[:xtilde_fc]) ``\\tilde{x}_{t+1, i}`` is the entry of asset ``i``.
  - $(math_dict[:w_t_iter])
  - ``\\tilde{\\boldsymbol{w}}_t``: Centred iterate, with ``\\bar{w}_t`` the mean of the entries of ``\\boldsymbol{w}_t``.
  - ``\\mathbf{K}``: Diagonal kernel matrix with the entries ``K_i`` in ``(0, 1]``.
  - ``q``: Shape of the kernel, the `q` field.
  - ``\\eta``: Rate of the step, the `eta` field.
  - $(math_dict[:t_period])
  - $(math_dict[:Proj_W_euclid])

When ``\\tilde{\\boldsymbol{x}}_{t+1} = \\boldsymbol{0}`` the step is zero, and ``\\boldsymbol{w}_{t+1} = \\mathrm{Proj}_{\\mathcal{W}}(\\boldsymbol{w}_t)``.

The kernel is the similarity of the paper between the allocation and the forecast. ``K_i`` is largest where the centred weight of asset ``i`` already matches its centred forecast. The vector inside the projection maximises ``\\langle \\boldsymbol{w}, \\mathbf{K}^{-1} \\tilde{\\boldsymbol{x}}_{t+1} \\rangle`` over the ellipsoid ``(\\boldsymbol{w} - \\boldsymbol{w}_t)^\\intercal \\mathbf{K}^{-2} (\\boldsymbol{w} - \\boldsymbol{w}_t) \\leq \\eta^2 \\lVert \\tilde{\\boldsymbol{x}}_{t+1} \\rVert^2``, which is the programme of the paper at the radius that it sets. The radius grows with the forecast, so the normalisation of [`ForecastTracking`](@ref) cancels.

On the simplex, the projection is one-hot on asset ``i`` whenever ``\\eta (K_i \\tilde{x}_{t+1, i} - K_j \\tilde{x}_{t+1, j}) \\geq 2`` for every other asset ``j``. At ``\\eta = 1000`` that gap is 0.002. Below that gap the answer can hold two assets. From a book on one asset, a gap of ``1.5 / \\eta`` gives the weights 0.75 and 0.25.

# Algorithm

[`online_update!`](@ref) runs these steps at the row `x` of the period, from the allocation `w`.

 1. Advance the forecaster with [`forecast_relative`](@ref), which gives the carrier `st` and the forecast `xhat`.
 2. Centre the forecast, which gives `dev`.
 3. When every entry of `dev` is zero, take the raw step `q = w` and go to step 5.
 4. Form the kernel `K` from the centred `w` and `dev`, and the raw step `q = w .+ eta .* K .* dev`.
 5. Project `q` onto the set with [`project`](@ref), with the [`price_adjusted_allocation`](@ref) of `w` after the row. Return `st` and the new allocation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KernelTrendTracking(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(; alg = KernelTrendPattern()),
        eta::Real = 1000,
        q::Real = 6,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> KernelTrendTracking

Keywords correspond to the struct's fields, and the defaults `eta = 1000` and `q = 6` are the paper's. The rule needs the rows that `me` needs.

## Validation

  - `eta > 0`. A `DomainError` is thrown otherwise.
  - `q > 0`. A `DomainError` is thrown otherwise.
  - `me` is not an [`Online`](@ref) wrapper, and holds none. An `ArgumentError` is thrown otherwise, and it names the wrapper.

# Examples

```jldoctest
julia> KernelTrendTracking()
KernelTrendTracking
    me ┼ PriceLevelExpectedReturns
       │   alg ┼ KernelTrendPattern
       │       │   window ┼ Int64: 5
       │       │       nu ┼ Float64: 0.5
       │       │     path ┼ ElasticNetPath
       │       │          │   theta ┼ Float64: 0.99
       │       │          │   ratio ┼ Float64: 0.001
       │       │          │   iters ┼ Int64: 10000
       │       │          │     tol ┴ Float64: 1.0e-10
   eta ┼ Int64: 1000
     q ┼ Int64: 6
  proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`ForecastTracking`](@ref)
  - [`KernelTrendPatternTracking`](@ref)
  - [`KernelTrendPattern`](@ref)

# References

  - $(ref_dict[:lai2018ktpt])
"""
struct KernelTrendTracking{T1 <: AbstractExpectedReturnsEstimator, T2 <: Real, T3 <: Real,
                           T4 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:forecaster])
    """
    me::T1
    """
    Rate of the step on the centred forecast after the kernel scales it.
    """
    eta::T2
    """
    Shape of the kernel. The kernel raises the distance between the centred weight and the centred forecast of an asset to the power `1 / q`, so a larger `q` gives a flatter similarity.
    """
    q::T3
    """
    $(field_dict[:proj])
    """
    proj::T4
    function KernelTrendTracking(me::AbstractExpectedReturnsEstimator, eta::Real, q::Real,
                                 proj::EuclideanProjection)
        assert_forecaster(me)
        @argcheck(eta > zero(eta), DomainError(eta, "eta must be positive"))
        @argcheck(q > zero(q), DomainError(q, "q must be positive"))
        return new{typeof(me), typeof(eta), typeof(q), typeof(proj)}(me, eta, q, proj)
    end
end
function KernelTrendTracking(me::Online, ::Real, ::Real, ::EuclideanProjection)
    return assert_forecaster(me)
end
function KernelTrendTracking(;
                             me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(;
                                                                                                                 alg = KernelTrendPattern()),
                             eta::Real = 1000, q::Real = 6,
                             proj::EuclideanProjection = EuclideanProjection())::KernelTrendTracking
    return KernelTrendTracking(me, eta, q, proj)
end
function port_opt_view(alg::KernelTrendTracking, i, args...)
    return KernelTrendTracking(; me = port_opt_view(alg.me, i, args...), eta = alg.eta,
                               q = alg.q, proj = alg.proj)
end
function rows_needed(alg::KernelTrendTracking)
    return rows_needed(alg.me)
end
function rule_state_seed(alg::KernelTrendTracking, ::AbstractVector)
    return forecaster_seed(alg.me)
end
function online_update!(alg::KernelTrendTracking, st, w::AbstractVector, x::AbstractVector,
                        rows, set::AbstractAllocationSet)
    st, xhat = forecast_relative(alg.me, st, x, rows)
    dev = xhat .- Statistics.mean(xhat)
    q = if all(iszero, dev)
        w
    else
        K = exp.(-abs.((w .- Statistics.mean(w)) .- dev) .^ inv(alg.q))
        w .+ alg.eta .* K .* dev
    end
    return st, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
    KernelTrendPatternTracking(; window::Integer = 5, nu::Real = 0.5, theta::Real = 0.99, iters::Integer = 10_000, tol::Real = 1e-10, q::Real = 6, eta::Real = 1000, proj::EuclideanProjection = EuclideanProjection())

Builds the kernel-based trend pattern tracking of Lai, Yang, Wu and Fang (2018), KTPT.

It is a [`KernelTrendTracking`](@ref) whose forecast is the [`KernelTrendPattern`](@ref) over `window` levels. `nu` mixes its initial state, and an [`ElasticNetPath`](@ref) at `theta` gives its intermediate state, with at most `iters` sweeps stopped at `tol`. The defaults `window = 5`, `nu = 0.5`, `theta = 0.99`, `q = 6` and `eta = 1000` are the paper's.

The paper reports that the wealth is robust to `eta` from 800 to 1300 and stable for `q` near 6. The statistic folds with a memory, so the head holds only the current row for this rule.

# Examples

```jldoctest
julia> KernelTrendPatternTracking(; window = 3)
KernelTrendTracking
    me ┼ PriceLevelExpectedReturns
       │   alg ┼ KernelTrendPattern
       │       │   window ┼ Int64: 3
       │       │       nu ┼ Float64: 0.5
       │       │     path ┼ ElasticNetPath
       │       │          │   theta ┼ Float64: 0.99
       │       │          │   ratio ┼ Float64: 0.001
       │       │          │   iters ┼ Int64: 10000
       │       │          │     tol ┴ Float64: 1.0e-10
   eta ┼ Int64: 1000
     q ┼ Int64: 6
  proj ┴ EuclideanProjection()
```

# Related

  - [`KernelTrendTracking`](@ref)
  - [`KernelTrendPattern`](@ref)
  - [`ElasticNetPath`](@ref)
  - [`PeakPriceTracking`](@ref)

# References

  - $(ref_dict[:lai2018ktpt])
"""
function KernelTrendPatternTracking(; window::Integer = 5, nu::Real = 0.5,
                                    theta::Real = 0.99, iters::Integer = 10_000,
                                    tol::Real = 1e-10, q::Real = 6, eta::Real = 1000,
                                    proj::EuclideanProjection = EuclideanProjection())::KernelTrendTracking
    alg = KernelTrendPattern(; window = window, nu = nu,
                             path = ElasticNetPath(; theta = theta, iters = iters,
                                                   tol = tol))
    return KernelTrendTracking(; me = PriceLevelExpectedReturns(; alg = alg), eta = eta,
                               q = q, proj = proj)
end
"""
$(DocStringExtensions.TYPEDEF)

Takes one soft-thresholded, linearised step of the log forecast return from the Price-Adjusted Allocation (TCO).

This is the transaction cost optimisation of Li, Wang, Huang and Hoi (2018). The threshold is a multiple of the cost rate, and it leaves each small trade at zero. The first form of the paper reads the reciprocal of the last price relative, [`LaggedPrice`](@ref) at `lag = 1`, which is the default. The second form reads the moving average of the price levels over the last price. The paper does not give its window, and Moon (2019) uses five levels, the default of [`MovingAverage`](@ref).

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{g} &= \\frac{\\hat{\\boldsymbol{x}}_{t+1}}{\\langle \\hat{\\boldsymbol{w}}_t, \\hat{\\boldsymbol{x}}_{t+1} \\rangle}\\,, \\\\
\\boldsymbol{v} &= \\eta \\left( \\boldsymbol{g} - \\bar{g} \\boldsymbol{1} \\right)\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\hat{\\boldsymbol{w}}_t + \\operatorname{sign}(\\boldsymbol{v}) \\odot \\max\\left(\\lvert \\boldsymbol{v} \\rvert - \\eta \\lambda,\\, 0\\right) \\right)\\,, \\\\
\\lambda &= 10 \\gamma\\,.
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc])
  - $(math_dict[:w_hat_t_padj])
  - ``\\boldsymbol{g}``: Gradient of ``\\log \\langle \\boldsymbol{w}, \\hat{\\boldsymbol{x}}_{t+1} \\rangle`` at ``\\hat{\\boldsymbol{w}}_t``.
  - ``\\bar{g}``: Mean of the entries of ``\\boldsymbol{g}``.
  - ``\\boldsymbol{v}``: Centred step before the threshold.
  - ``\\eta``: Step size of the linearised step, the `eta` field.
  - ``\\lambda``: Weight of the ``L_1`` penalty on the trade.
  - ``\\gamma``: Proportional transaction cost rate, the `gamma` field.
  - $(math_dict[:t_period])
  - $(math_dict[:Proj_W_euclid])

The paper states the programme ``\\underset{\\boldsymbol{w}}{\\min} \\; -\\log \\langle \\boldsymbol{w}, \\hat{\\boldsymbol{x}}_{t+1} \\rangle + \\lambda \\lVert \\boldsymbol{w} - \\hat{\\boldsymbol{w}}_t \\rVert_1`` on the simplex. Its Proposition 4.2 linearises the log at ``\\hat{\\boldsymbol{w}}_t``, adds the proximal term ``\\lVert \\boldsymbol{w} - \\hat{\\boldsymbol{w}}_t \\rVert^2 / (2 \\eta)``, and takes ``\\bar{g}`` as the multiplier of the budget. The vector inside the projection is the exact minimiser of that proximal programme without the bounds, and the projection restores them. When ``\\eta \\lambda`` exceeds every entry of ``\\lvert \\boldsymbol{v} \\rvert``, the rule holds ``\\hat{\\boldsymbol{w}}_t``.

The proof of Proposition 4.2 renames the threshold ``\\eta \\lambda`` as ``\\lambda``, and the experiments of the paper set that renamed threshold to ``10 \\gamma``. Moon (2019) records that the code of the authors applies the threshold ``10 \\eta \\gamma``, and that the published results rest on it. The library takes that threshold, so ``\\lambda = 10 \\gamma`` is the weight of the ``L_1`` penalty.

# Algorithm

[`online_update!`](@ref) runs these steps at the row `x` of the period, from the allocation `w`.

 1. Advance the forecaster with [`forecast_relative`](@ref), which gives the carrier `st` and the forecast `xhat`.
 2. Form the Price-Adjusted Allocation `what` of `w` after the row, with [`price_adjusted_allocation`](@ref).
 3. Form the gradient `g` at `what`.
 4. Form the centred step `v`.
 5. Form the threshold `lam = 10 * eta * gamma`, which is ``\\eta \\lambda``.
 6. Form the raw step `q` by the soft threshold of `v` at `lam`, added to `what`.
 7. Project `q` onto the set with [`project`](@ref), with `what`. Return `st` and the new allocation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TransactionCostOptimisation(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(; alg = LaggedPrice()),
        eta::Real = 10,
        gamma::Real = 0.001,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> TransactionCostOptimisation

Keywords correspond to the struct's fields. The rule needs the rows that `me` needs. `eta = 10` is the paper's. `gamma` is the proportional cost rate that the trader pays, which the paper leaves to the market. The paper tests `gamma = 0.0025` and `gamma = 0.005`.

## Validation

  - `eta > 0`. A `DomainError` is thrown otherwise.
  - `gamma >= 0`. A `DomainError` is thrown otherwise.
  - `me` is not an [`Online`](@ref) wrapper, and holds none. An `ArgumentError` is thrown otherwise, and it names the wrapper.

# Examples

```jldoctest
julia> TransactionCostOptimisation()
TransactionCostOptimisation
     me ┼ PriceLevelExpectedReturns
        │   alg ┼ LaggedPrice
        │       │   lag ┴ Int64: 1
    eta ┼ Int64: 10
  gamma ┼ Float64: 0.001
   proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`LaggedPrice`](@ref)
  - [`MovingAverage`](@ref)
  - [`price_adjusted_allocation`](@ref)
  - [`forecast_relative`](@ref)

# References

  - $(ref_dict[:li2018tco])
  - $(ref_dict[:moon2019])
"""
struct TransactionCostOptimisation{T1 <: AbstractExpectedReturnsEstimator, T2 <: Real,
                                   T3 <: Real, T4 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:forecaster])
    """
    me::T1
    """
    Step size of the linearised step. It also scales the soft threshold.
    """
    eta::T2
    """
    Proportional transaction cost rate. The soft threshold is `10 * eta * gamma`.
    """
    gamma::T3
    """
    $(field_dict[:proj])
    """
    proj::T4
    function TransactionCostOptimisation(me::AbstractExpectedReturnsEstimator, eta::Real,
                                         gamma::Real, proj::EuclideanProjection)
        assert_forecaster(me)
        @argcheck(eta > zero(eta), DomainError(eta, "eta must be positive"))
        @argcheck(gamma >= zero(gamma), DomainError(gamma, "gamma must be non-negative"))
        return new{typeof(me), typeof(eta), typeof(gamma), typeof(proj)}(me, eta, gamma,
                                                                         proj)
    end
end
function TransactionCostOptimisation(me::Online, ::Real, ::Real, ::EuclideanProjection)
    return assert_forecaster(me)
end
function TransactionCostOptimisation(;
                                     me::Union{<:AbstractExpectedReturnsEstimator,
                                               <:Online} = PriceLevelExpectedReturns(;
                                                                                     alg = LaggedPrice()),
                                     eta::Real = 10, gamma::Real = 0.001,
                                     proj::EuclideanProjection = EuclideanProjection())::TransactionCostOptimisation
    return TransactionCostOptimisation(me, eta, gamma, proj)
end
function port_opt_view(alg::TransactionCostOptimisation, i, args...)
    return TransactionCostOptimisation(; me = port_opt_view(alg.me, i, args...),
                                       eta = alg.eta, gamma = alg.gamma, proj = alg.proj)
end
function rows_needed(alg::TransactionCostOptimisation)
    return rows_needed(alg.me)
end
function rule_state_seed(alg::TransactionCostOptimisation, ::AbstractVector)
    return forecaster_seed(alg.me)
end
function online_update!(alg::TransactionCostOptimisation, st, w::AbstractVector,
                        x::AbstractVector, rows, set::AbstractAllocationSet)
    st, xhat = forecast_relative(alg.me, st, x, rows)
    what = price_adjusted_allocation(w, x)
    g = xhat ./ LinearAlgebra.dot(what, xhat)
    v = alg.eta .* (g .- Statistics.mean(g))
    lam = 10 * alg.eta * alg.gamma
    q = what .+ sign.(v) .* max.(abs.(v) .- lam, zero(lam))
    return st, project(alg.proj, set, q, what)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that find the iterate of a short-term sparse portfolio step.

The iterate is the vector ``\\boldsymbol{b}`` whose scaled projection is the next allocation. The members differ in the programme that they solve, and [`ShortTermSparsePortfolio`](@ref) compares them.

# Interfaces

To implement a new algorithm, subtype `AbstractSparsePortfolioAlgorithm` with its parameters as the fields of the struct, and implement this method:

  - `sparse_portfolio_iterate(alg::AbstractSparsePortfolioAlgorithm, phi::AbstractVector, w::AbstractVector) -> AbstractVector`: Returns the iterate for the objective vector `phi`. The entries of the iterate sum to one. `w` is the held allocation, which an iteration reads as its seed.

# Related

  - [`ShortTermSparsePortfolio`](@ref)
  - [`L1Optimum`](@ref)
  - [`HuberOptimum`](@ref)
  - [`AlternatingDirectionMethod`](@ref)
  - [`sparse_portfolio_iterate`](@ref)
"""
abstract type AbstractSparsePortfolioAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Solves a linear programme on the log of the forecast with an ``L_1`` penalty, then projects the scaled solution (SSPO).

This is the short-term sparse portfolio optimisation of Lai, Yang, Fang and Wu (2018). The forecast is the window peak over the last price by default, so every entry of the forecast is at least one. `alg` finds the iterate, and the scale of the iterate before the projection makes the allocation sparse.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\phi} &= -\\left(1.1 \\log \\hat{\\boldsymbol{x}}_{t+1} + \\boldsymbol{1}\\right)\\,, \\\\
&\\underset{\\boldsymbol{b}}{\\min} \\; \\langle \\boldsymbol{b}, \\boldsymbol{\\phi} \\rangle + \\lambda \\lVert \\boldsymbol{b} \\rVert_1 \\quad \\text{s.t.} \\quad \\boldsymbol{1}^\\intercal \\boldsymbol{b} = 1\\,, \\\\
&\\underset{\\boldsymbol{b}, \\boldsymbol{g}}{\\min} \\; \\langle \\boldsymbol{b}, \\boldsymbol{\\phi} \\rangle + \\lambda \\lVert \\boldsymbol{g} \\rVert_1 + \\tfrac{\\lambda}{2 \\gamma} \\lVert \\boldsymbol{b} - \\boldsymbol{g} \\rVert^2 \\quad \\text{s.t.} \\quad \\boldsymbol{1}^\\intercal \\boldsymbol{b} = 1\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left(\\zeta \\boldsymbol{b}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:phi_sspo])
  - $(math_dict[:xhat_fc])
  - $(math_dict[:b_sspo])
  - $(math_dict[:g_aux])
  - $(math_dict[:lambda_l1])
  - $(math_dict[:gamma_st])
  - ``\\zeta``: Scale of the iterate before the projection, the `zeta` field.
  - $(math_dict[:t_period])
  - $(math_dict[:Proj_W_euclid])

The first programme is the programme that the paper states. The paper solves it by an alternating direction iteration, which couples ``\\boldsymbol{b}`` to its soft threshold ``\\boldsymbol{g}`` by a quadratic term with no multiplier. So the fixed point of that iteration solves the second programme. `alg` chooses the iterate ``\\boldsymbol{b}``:

| `alg`                                | The iterate                                              | Parameters                               | Cost of a step                                 |
|:------------------------------------ |:-------------------------------------------------------- |:---------------------------------------- |:---------------------------------------------- |
| [`L1Optimum`](@ref)                  | the optimum of the stated programme                      | none                                     | ``O(N)``                                       |
| [`HuberOptimum`](@ref)               | the fixed point of the paper's iteration, in closed form | `lambda`, `gamma`                        | ``O(N)``, or ``O(N^2)`` when ``N \\gamma > 1`` |
| [`AlternatingDirectionMethod`](@ref) | the paper's iteration at the paper's stop                | `lambda`, `gamma`, `eta`, `iters`, `tol` | up to `iters` iterations of ``O(N)``           |

The first programme puts the whole budget on the largest forecast. The fixed point gives every other asset at most ``\\gamma``. So its scaled projection also puts the whole budget on the largest forecast while ``N \\gamma \\leq 1 - 1 / \\zeta``, with ``N`` the number of assets. At the paper's `gamma = 0.01` and `zeta = 500` that is up to 99 assets. With more assets, the projection of the fixed point can hold a few assets.

The iteration of the paper approaches the fixed point slowly. It stops when the budget residual is below its tolerance, which it can reach at a sign change of the residual while the iterate is still tenths away from the fixed point. So its projection is usually the projection of the fixed point, but not always. [`HuberOptimum`](@ref) is the default, because the method of the paper converges to its answer. Take [`L1Optimum`](@ref) for the programme that the paper states, and [`AlternatingDirectionMethod`](@ref) for the loop of the paper with its stop.

Both programmes have a minimum only when ``\\max \\boldsymbol{\\phi} - \\min \\boldsymbol{\\phi} \\leq 2 \\lambda``. At ``\\lambda = 1/2`` that is ``\\max \\hat{\\boldsymbol{x}}_{t+1} / \\min \\hat{\\boldsymbol{x}}_{t+1} \\leq e^{1 / 1.1}``. Past that bound the objective decreases without limit as the budget moves to the largest forecast. [`L1Optimum`](@ref) and [`HuberOptimum`](@ref) take that limit, and the iteration of the paper returns its last iterate.

# Algorithm

[`online_update!`](@ref) runs these steps at the row `x` of the period, from the allocation `w`.

 1. Advance the forecaster with [`forecast_relative`](@ref), which gives the carrier `st` and the forecast `xhat`.
 2. Check that every entry of `xhat` is positive.
 3. Form the objective vector `phi` from `xhat`.
 4. Find the iterate `b` with [`sparse_portfolio_iterate`](@ref), from `phi` and the seed `w`.
 5. Project `zeta .* b` onto the set with [`project`](@ref), with the [`price_adjusted_allocation`](@ref) of `w` after the row. Return `st` and the new allocation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShortTermSparsePortfolio(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(; alg = WindowPeak()),
        alg::AbstractSparsePortfolioAlgorithm = HuberOptimum(),
        zeta::Real = 500,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ShortTermSparsePortfolio

Keywords correspond to the struct's fields. The defaults of `zeta` and of the parameters of `alg` are the paper's. The rule needs the rows that `me` needs.

## Validation

  - `zeta > 0`. A `DomainError` is thrown otherwise.
  - `me` is not an [`Online`](@ref) wrapper, and holds none. An `ArgumentError` is thrown otherwise, and it names the wrapper.
  - At the update, `x̂ > 0` in every asset, so the log is defined. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ShortTermSparsePortfolio()
ShortTermSparsePortfolio
    me ┼ PriceLevelExpectedReturns
       │   alg ┼ WindowPeak
       │       │   window ┴ Int64: 5
   alg ┼ HuberOptimum
       │   lambda ┼ Float64: 0.5
       │    gamma ┴ Float64: 0.01
  zeta ┼ Int64: 500
  proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`L1Optimum`](@ref)
  - [`HuberOptimum`](@ref)
  - [`AlternatingDirectionMethod`](@ref)
  - [`sparse_portfolio_iterate`](@ref)
  - [`WindowPeak`](@ref)
  - [`ForecastTracking`](@ref)
  - [`forecast_relative`](@ref)

# References

  - $(ref_dict[:lai2018sspo])
"""
struct ShortTermSparsePortfolio{T1 <: AbstractExpectedReturnsEstimator,
                                T2 <: AbstractSparsePortfolioAlgorithm, T3 <: Real,
                                T4 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:forecaster])
    """
    me::T1
    """
    Algorithm that finds the iterate.
    """
    alg::T2
    """
    Scale of the iterate before the projection.
    """
    zeta::T3
    """
    $(field_dict[:proj])
    """
    proj::T4
    function ShortTermSparsePortfolio(me::AbstractExpectedReturnsEstimator,
                                      alg::AbstractSparsePortfolioAlgorithm, zeta::Real,
                                      proj::EuclideanProjection)
        assert_forecaster(me)
        @argcheck(zeta > zero(zeta), DomainError(zeta, "zeta must be positive"))
        return new{typeof(me), typeof(alg), typeof(zeta), typeof(proj)}(me, alg, zeta, proj)
    end
end
function ShortTermSparsePortfolio(me::Online, ::AbstractSparsePortfolioAlgorithm, ::Real,
                                  ::EuclideanProjection)
    return assert_forecaster(me)
end
function ShortTermSparsePortfolio(;
                                  me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(;
                                                                                                                      alg = WindowPeak()),
                                  alg::AbstractSparsePortfolioAlgorithm = HuberOptimum(),
                                  zeta::Real = 500,
                                  proj::EuclideanProjection = EuclideanProjection())::ShortTermSparsePortfolio
    return ShortTermSparsePortfolio(me, alg, zeta, proj)
end
function port_opt_view(alg::ShortTermSparsePortfolio, i, args...)
    return ShortTermSparsePortfolio(; me = port_opt_view(alg.me, i, args...), alg = alg.alg,
                                    zeta = alg.zeta, proj = alg.proj)
end
function rows_needed(alg::ShortTermSparsePortfolio)
    return rows_needed(alg.me)
end
function rule_state_seed(alg::ShortTermSparsePortfolio, ::AbstractVector)
    return forecaster_seed(alg.me)
end
function online_update!(alg::ShortTermSparsePortfolio, st, w::AbstractVector,
                        x::AbstractVector, rows, set::AbstractAllocationSet)
    st, xhat = forecast_relative(alg.me, st, x, rows)
    @argcheck(all(v -> v > zero(v), xhat),
              DomainError(xhat,
                          "the short-term sparse portfolio takes the log of the Price Relative Forecast, which must be positive in every asset"))
    phi = -(1.1 .* log.(xhat) .+ one(eltype(xhat)))
    b = sparse_portfolio_iterate(alg.alg, phi, w)
    return st, project(alg.proj, set, alg.zeta .* b, price_adjusted_allocation(w, x))
end
export ForecastReversion, MovingAverageReversion, ExponentialMovingAverageReversion,
       RobustMedianReversion, ReweightedPriceRelativeTracking, GaussianWeightingReversion,
       LocalAdaptiveLearning, ForecastTracking, PeakPriceTracking,
       AdaptiveInputCompositeTrend, TrendPromotePriceTracking, KernelTrendTracking,
       KernelTrendPatternTracking, TransactionCostOptimisation, ShortTermSparsePortfolio
public ForecasterState, forecast_relative
