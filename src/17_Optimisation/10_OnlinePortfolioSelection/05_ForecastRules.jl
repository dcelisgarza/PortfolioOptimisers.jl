"""
$(DocStringExtensions.TYPEDEF)

The carrier of a forecast-reading rule whose forecaster folds: the expected-returns estimator on the rule's `me` slot, carrying its own Partial Fit State.

A forecaster with an exact fold — [`SimpleExpectedReturns`](@ref), [`ExpWeightedExpectedReturns`](@ref), a [`PriceLevelExpectedReturns`](@ref) over a folding statistic, a [`PriorExpectedReturns`](@ref) over a prior that folds — is folded on every row and read from its state, so the head holds no rows for it; one with no exact fold is refit on the rows the head holds and the rule's carrier is `nothing`. [`supports_partial_fit`](@ref) is the question, asked once at the seed.

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
    The forecaster, carrying the state of the rows folded so far.
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

A copy of a folded forecaster whose state aliases no array of the original: the `cache` field is copied through the state's own [`Base.copy`](@ref), and the adapter recurses into its prior.

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

A copy of a folded prior whose state aliases no array of the original, for the adapter's copy.

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

Refuses an [`Online`](@ref) wrapper in a rule's forecaster slot, by name: the head's rows buffer is the buffer, and a second one would hold the rows twice.

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

The carrier a forecast-reading rule seeds: a [`ForecasterState`](@ref) over the forecaster where it folds, `nothing` where it is refit from the head's rows.

# Validation

  - The forecaster carries no partial-fit state at the seed. An `ArgumentError` is thrown otherwise: the head starts cold, and a state folded elsewhere would make the first row's forecast read rows the head never saw.

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

The Price Relative Forecast `x̂ = 1 .+ mu` of a rule's forecaster after the row `x`, by the fold-or-refit rule: a forecaster on a [`ForecasterState`](@ref) is folded on the row and read from its state; one with none is refit on the rows the head holds, which include the row, or on the row alone when the head holds none. A forecast is **flat** — one in every asset, which every rule's step holds on — where the forecaster has fewer rows than [`forecast_min_rows`](@ref) or answers a non-finite entry ([`flat_where_undefined`](@ref)).

# Arguments

  - `me`: The forecaster on the rule's slot.
  - `st`: The rule's carrier, or `nothing`.
  - `x`: The price relative of the row.
  - `rows`: The returns the head holds through the row, or `nothing`.

# Returns

  - `(st', x̂)::Tuple`: The carrier after the row, and the forecast, a new vector.

# Related

  - [`ForecasterState`](@ref)
  - [`ForecastReversion`](@ref)
  - [`ForecastTracking`](@ref)
  - [`refit_rows`](@ref)
"""
function forecast_relative(me::AbstractExpectedReturnsEstimator, ::Nothing,
                           x::AbstractVector, rows)
    X = refit_rows(rows, x)
    if size(X, 1) < forecast_min_rows(me)
        return nothing, fill(one(eltype(x)), length(x))
    end
    return nothing,
           flat_where_undefined(one(eltype(x)) .+ vec(Statistics.mean(me, X; dims = 1)))
end
function forecast_relative(::AbstractExpectedReturnsEstimator, st::ForecasterState,
                           x::AbstractVector, rows)
    me = partial_fit!(st.me, x .- one(eltype(x)))
    return ForecasterState(me),
           flat_where_undefined(one(eltype(x)) .+ vec(Statistics.mean(me)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The number of rows a stateless forecaster needs before its batch verb is defined: two when its estimator tree holds a covariance, variance or prior estimator, whose second moment is undefined on one row, one otherwise, and the floor any estimator in the tree states through [`fit_min_rows`](@ref) where that is larger. Below it the forecast is flat.

# Related

  - [`forecast_relative`](@ref)
  - [`fit_min_rows`](@ref)
"""
function forecast_min_rows(me::AbstractExpectedReturnsEstimator)
    return max(holds_second_moment(me) ? 2 : 1, fit_min_rows(me))
end
"""
    holds_second_moment(est)

Whether an estimator tree holds a covariance, variance or prior estimator at any depth.

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

Reads a non-finite entry of a Price Relative Forecast as one: the asset carries no forecast, so the step holds it. A forecaster whose estimate is undefined over the first rows — a variance over one observation — therefore holds until it is defined.

# Related

  - [`forecast_relative`](@ref)
"""
function flat_where_undefined(xhat::AbstractVector)
    return [isfinite(v) ? v : one(v) for v in xhat]
end
"""
    refit_rows(rows::AbstractMatrix, x::AbstractVector)
    refit_rows(rows::Nothing, x::AbstractVector)

The returns a stateless forecaster is refit on: the rows the head holds, or the current row alone as a one-row matrix when the head holds none.

# Related

  - [`forecast_relative`](@ref)
"""
function refit_rows(rows::AbstractMatrix, ::AbstractVector)
    return rows
end
function refit_rows(::Nothing, x::AbstractVector)
    return reshape(x .- one(eltype(x)), 1, :)
end
"""
    scale_relative(scale::Nothing, x::AbstractVector, rows)
    scale_relative(scale::AbstractPriceLevelStatistic, x::AbstractVector, rows)

The diagonal preconditioner of a [`ForecastReversion`](@ref) step: `nothing` for the identity, or the Price Relative Forecast of the `scale` statistic read from the rows the head holds.

# Related

  - [`ForecastReversion`](@ref)
  - [`ReweightedPriceRelativeTracking`](@ref)
"""
function scale_relative(::Nothing, ::AbstractVector, ::Any)
    return nothing
end
function scale_relative(scale::AbstractPriceLevelStatistic, x::AbstractVector, rows)
    me = PriceLevelExpectedReturns(; alg = scale)
    return one(eltype(x)) .+ vec(Statistics.mean(me, refit_rows(rows, x); dims = 1))
end
"""
    scale_rows(scale::Nothing)
    scale_rows(scale::AbstractPriceLevelStatistic)

The rows a `scale` statistic reads at a step, `0` for none.

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

The passive-aggressive step toward a Price Relative Forecast: the closest allocation to the current one whose return on the forecast `x̂` is at least `eps`, the one update the moving-average reversion of Li and Hoi (2012), the robust median reversion of Huang, Zhou, Li, Hoi and Zhou (2016), the reweighted price relative tracking of Lai, Yang, Fang and Wu (2018), the Gaussian weighting reversion of Cai and Ye (2019) and the local adaptive learning of Guan and An (2019) share.

The forecast is the expected-returns estimator on `me`, read as `x̂ = 1 .+ mu` and advanced by the fold-or-refit rule of [`forecast_relative`](@ref), so the papers differ only in the statistic on that slot; [`MovingAverageReversion`](@ref), [`RobustMedianReversion`](@ref), [`ExponentialMovingAverageReversion`](@ref), [`ReweightedPriceRelativeTracking`](@ref), [`GaussianWeightingReversion`](@ref) and [`LocalAdaptiveLearning`](@ref) are the constructors that fill it with each paper's own. Any expected-returns estimator may sit there — a mean of past returns, a shrunk mean, a Prior's mean through [`PriorExpectedReturns`](@ref).

# Mathematical definition

```math
\\begin{align}
\\lambda_t &= \\max\\left(0, \\frac{\\epsilon - \\langle \\boldsymbol{w}_t, \\hat{\\boldsymbol{x}}_{t+1} \\rangle}{\\lVert \\hat{\\boldsymbol{x}}_{t+1} - \\bar{x}_{t+1} \\boldsymbol{1} \\rVert^2}\\right)\\,,\\quad
\\boldsymbol{w}_{t+1} = \\mathrm{Proj}\\left( \\boldsymbol{w}_t + \\lambda_t \\boldsymbol{D}_{t+1} \\left( \\hat{\\boldsymbol{x}}_{t+1} - \\bar{x}_{t+1} \\boldsymbol{1} \\right) \\right)\\,,
\\end{align}
```

the step zero when every asset is forecast alike. ``\\boldsymbol{D}_{t+1}`` is the identity when `scale` is `nothing`, and otherwise the diagonal of the Price Relative Forecast of the `scale` statistic read from the head's rows, the preconditioner of the reweighted price relative tracking; ``\\lambda_t`` is the same either way, so the step is not the mirror step of a ``\\boldsymbol{D}``-norm. The sign is opposite to [`PassiveAggressiveMeanReversion`](@ref)'s, because `x̂` forecasts the next return rather than reading the last one. The rule is a total bet on the reversion the forecast encodes: it compounds on a reverting market and collapses on a trending one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForecastReversion(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(),
        eps::Real = 10,
        scale::Option{<:AbstractPriceLevelStatistic} = nothing,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ForecastReversion

Keywords correspond to the struct's fields. `rows_needed` is the larger of the forecaster's and the scale's, so the head holds the rows either reads; a forecaster that folds reads none.

## Validation

  - `eps > 0`. A `DomainError` is thrown otherwise.
  - `me` is not, and holds no, [`Online`](@ref) wrapper: the head's rows buffer is the buffer. An `ArgumentError` is thrown otherwise, by name.

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

# References

  - $(ref_dict[:lihoi2012])
  - $(ref_dict[:huang2016])
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
    Target return on the forecast. Larger is more aggressive.
    """
    eps::T2
    """
    The statistic whose Price Relative Forecast preconditions the direction, or `nothing` for the identity.
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

The on-line moving average reversion of Li and Hoi (2012): a [`ForecastReversion`](@ref) whose forecast is the [`MovingAverage`](@ref) of the last `window` price levels over the last price (OLMAR).

`window` is the reversion horizon, the one parameter that matters, and the natural thing to tune with a search over `"alg.me.alg.window"`. The paper's algorithm takes `eps > 1` and `window >= 3`; the step is defined for any positive `eps` — below one it moves only when the forecast return of the held allocation falls under `eps` — and for two levels, where the average is the mean of the current price and the previous one, so the constructor admits `eps > 0` and `window >= 2`.

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

The second form of the on-line moving average reversion of Li, Hoi, Sahoo and Liu (2015): a [`ForecastReversion`](@ref) whose forecast is the [`ExponentialMovingAverage`](@ref) of the price levels over the last price (OLMAR-2).

The paper states no `alpha`; `0.5` is the middle of the range the literature scans and is the library's choice. The average is seeded at the first price, so the forecast after one level is one, the value the paper's closed expansion gives at its second step; the two readings of the seed agree from then on. The statistic folds, so the rule carries one vector and the head holds no rows for it.

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

The robust median reversion of Huang, Zhou, Li, Hoi and Zhou (2016): a [`ForecastReversion`](@ref) whose forecast is the [`SpatialMedian`](@ref) of the last `window` price levels over the last price (RMR).

The point is the breakdown point: a single extreme print moves a mean without limit and a spatial median almost not at all. The median is read on the reconstructed price path, every asset's last level at one, and its iteration runs to a tighter tolerance than the paper's; [`SpatialMedian`](@ref) states both and their size.

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
    ReweightedPriceRelativeTracking(; window::Integer = 5, theta::Real = 0.7, eps::Real = 50, proj::EuclideanProjection = EuclideanProjection())

The reweighted price relative tracking of Lai, Yang, Fang and Wu (2018), as restated with equations by Li, Luo and Xu (2023): a [`ForecastReversion`](@ref) whose forecast is the [`ReweightedPriceRelative`](@ref) recursion and whose direction is preconditioned by the [`MovingAverage`](@ref) forecast of the same `window` (RPRT).

The paper's own `theta` and `eps` are not read; the defaults are the restating paper's. With `eps = 50` the constraint always binds and the projected answer is near one-hot, as the peak-tracking rules' are.

# Examples

```jldoctest
julia> ReweightedPriceRelativeTracking()
ForecastReversion
     me ┼ PriceLevelExpectedReturns
        │   alg ┼ ReweightedPriceRelative
        │       │   theta ┴ Float64: 0.7
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
function ReweightedPriceRelativeTracking(; window::Integer = 5, theta::Real = 0.7,
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

The Gaussian weighting reversion of Cai and Ye (2019): a [`ForecastReversion`](@ref) whose forecast is the [`GaussianWeightedDoubleEstimate`](@ref) over the last price (GWR).

The paper's bandit over `tau` (GWR-A) is not built: a parameter chosen online from reward is a mechanism the family has no seam for.

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

The local adaptive learning of Guan and An (2019): a [`ForecastReversion`](@ref) whose forecast is a [`TrendSwitch`](@ref) on the [`RegressionSlope`](@ref) — the [`WindowPeak`](@ref) of an asset whose slope exceeds `threshold`, the [`ExponentialMovingAverage`](@ref) of the others (LOAD).

The paper states neither the reversion threshold `eps` nor the ridge weight `lambda`; the defaults are the moving-average reversion's `eps = 10` and plain least squares. The slope is tested on the reconstructed price path, and the exponential average is the paper's full-history recursion, so the head holds every row for this rule. The paper prints its step length with the norm of the centred forecast where the solution of its own programme — the closest allocation whose forecast return reaches `eps` — has the squared norm; the step taken is that solution, the one the moving-average reversion takes, and it differs from the printed one.

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

The tracking step toward a Price Relative Forecast: a step of fixed Euclidean length `eps` along the centred forecast, then the projection, the one update the peak price tracking of Lai, Dai, Ren and Huang (2018), the adaptive input and composite trend representation of Lai, Dai, Ren and Huang (2018) and the trend-promote price tracking of Dai, Liang, Dai, Huang and Adnan (2022) share; the mirror of [`ForecastReversion`](@ref).

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{x}}_{\\perp} &= \\hat{\\boldsymbol{x}}_{t+1} - \\bar{x}_{t+1} \\boldsymbol{1}\\,,\\quad
\\boldsymbol{w}_{t+1} = \\mathrm{Proj}\\left( \\boldsymbol{w}_t + \\epsilon\\, \\frac{\\hat{\\boldsymbol{x}}_{\\perp}}{\\lVert \\hat{\\boldsymbol{x}}_{\\perp} \\rVert} \\right)\\,,
\\end{align}
```

holding when the centred forecast is zero. The closed form is the exact maximiser of ``\\langle \\boldsymbol{w}, \\hat{\\boldsymbol{x}} \\rangle`` over the ``\\epsilon``-ball intersected with the budget hyperplane, and the projection restores non-negativity afterwards, the same drop-then-project shape as the reversion step. `eps` is a step **length**, not a rate: the forecast fixes the direction alone. At the papers' defaults the step saturates — the projected answer is one-hot on the asset with the largest centred forecast unless the top two are within `2 / eps` of each other in the unit direction — and the ball bites only well below `eps = 1`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForecastTracking(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(; alg = WindowPeak()),
        eps::Real = 100,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ForecastTracking

Keywords correspond to the struct's fields. `rows_needed` forwards to `me`.

## Validation

  - `eps > 0`. A `DomainError` is thrown otherwise.
  - `me` is not, and holds no, [`Online`](@ref) wrapper. An `ArgumentError` is thrown otherwise, by name.

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

# References

  - $(ref_dict[:lai2018ppt])
"""
struct ForecastTracking{T1 <: AbstractExpectedReturnsEstimator, T2 <: Real,
                        T3 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:forecaster])
    """
    me::T1
    """
    The Euclidean length of the step along the centred forecast.
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

The peak price tracking of Lai, Dai, Ren and Huang (2018): a [`ForecastTracking`](@ref) whose forecast is the [`WindowPeak`](@ref) of the last `window` price levels over the last price (PPT).

At the paper's `eps = 100` the answer is, in practice, all wealth on the asset with the highest peak-over-last ratio; the paper's sensitivity study is flat from `eps = 50` up, which is this saturation.

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

The adaptive input and composite trend representation of Lai, Dai, Ren and Huang (2018): a [`ForecastTracking`](@ref) whose forecast is the [`CompositeTrend`](@ref) of the [`MovingAverage`](@ref), the [`ExponentialMovingAverage`](@ref) and the [`WindowPeak`](@ref) over `window` levels (AICTR).

The paper states no smoothing weight for its exponential average; `alpha` defaults to the library's `0.5`. The exponential average is a full-history recursion, so the head holds every row for this rule.

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

The trend-promote price tracking of Dai, Liang, Dai, Huang and Adnan (2022): a [`ForecastTracking`](@ref) whose forecast is a [`TrendSwitch`](@ref) on the [`PairwiseSlopeSum`](@ref) — the [`TruncatedExponentialMovingAverage`](@ref) of a rising asset, the current price of a flat one, the [`WindowPeak`](@ref) of a falling one (TPPT).

The paper's expression for the rising branch is not computable as printed; [`TruncatedExponentialMovingAverage`](@ref) states the reading taken, so a parity test against the paper's numbers is not possible.

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

The kernel-scaled tracking step toward a Price Relative Forecast of Lai, Yang, Wu and Fang (2018): the centred forecast, scaled per asset by its similarity to the centred allocation, added at a fixed rate and projected; the sibling of [`ForecastTracking`](@ref) whose step is not normalised.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\boldsymbol{w}}_{t} &= \\boldsymbol{w}_{t} - \\bar{w}_{t} \\boldsymbol{1}\\,,\\quad
\\tilde{\\boldsymbol{x}}_{t+1} = \\hat{\\boldsymbol{x}}_{t+1} - \\bar{x}_{t+1} \\boldsymbol{1}\\,,\\quad
K_{i} = \\exp\\left( -\\lvert \\tilde{w}_{t, i} - \\tilde{x}_{t+1, i} \\rvert^{1/q} \\right)\\,,\\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}\\left( \\boldsymbol{w}_{t} + \\eta\\, \\boldsymbol{K} \\odot \\tilde{\\boldsymbol{x}}_{t+1} \\right)\\,,
\\end{align}
```

holding when the centred forecast is zero. The diagonal kernel ``\\boldsymbol{K}`` is the paper's similarity between the current allocation and the forecast, largest where an asset's centred weight already matches its centred forecast; the step is the maximiser of ``\\langle \\boldsymbol{w}, \\boldsymbol{K}^{-1} \\tilde{\\boldsymbol{x}} \\rangle`` over an ellipsoid of radius ``\\eta \\lVert \\tilde{\\boldsymbol{x}} \\rVert`` in the ``\\boldsymbol{K}^{-2}`` metric, so the radius grows with the forecast and the normalisation of [`ForecastTracking`](@ref) cancels. `eta` is a rate on the centred forecast, not a step length: at the paper's `eta = 1000` the projected answer is one-hot on the asset with the largest kernel-scaled centred forecast unless two are within `1 / eta` of each other, as the peak-tracking rules' at their defaults, and the step spreads only well below `eta = 1`. The forecast is the expected-returns estimator on `me`, advanced by the fold-or-refit rule of [`forecast_relative`](@ref); [`KernelTrendPatternTracking`](@ref) is the constructor that fills it with the paper's [`KernelTrendPattern`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KernelTrendTracking(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(; alg = KernelTrendPattern()),
        eta::Real = 1000,
        q::Real = 6,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> KernelTrendTracking

Keywords correspond to the struct's fields, and the defaults are the paper's. `rows_needed` forwards to `me`.

## Validation

  - `eta > 0`. A `DomainError` is thrown otherwise.
  - `q > 0`. A `DomainError` is thrown otherwise.
  - `me` is not, and holds no, [`Online`](@ref) wrapper. An `ArgumentError` is thrown otherwise, by name.

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
       │       │          │   iters ┼ Int64: 1000
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
    The rate on the kernel-scaled centred forecast.
    """
    eta::T2
    """
    The shape of the kernel, the root the distance between an asset's centred weight and centred forecast is raised to; larger is a flatter similarity.
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
    KernelTrendPatternTracking(; window::Integer = 5, nu::Real = 0.5, theta::Real = 0.99, q::Real = 6, eta::Real = 1000, proj::EuclideanProjection = EuclideanProjection())

The kernel-based trend pattern tracking of Lai, Yang, Wu and Fang (2018): a [`KernelTrendTracking`](@ref) whose forecast is the [`KernelTrendPattern`](@ref) over `window` levels, its initial state mixed by `nu` and its intermediate state read off an [`ElasticNetPath`](@ref) at `theta` (KTPT).

The paper's sensitivity study is flat over `eta` from 800 to 1300, which is the saturation the step's docstring states; `q` around 6 is where it reports the wealth stable. The statistic folds with a memory, so the head holds no rows for this rule.

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
       │       │          │   iters ┼ Int64: 1000
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
                                    theta::Real = 0.99, q::Real = 6, eta::Real = 1000,
                                    proj::EuclideanProjection = EuclideanProjection())::KernelTrendTracking
    alg = KernelTrendPattern(; window = window, nu = nu,
                             path = ElasticNetPath(; theta = theta))
    return KernelTrendTracking(; me = PriceLevelExpectedReturns(; alg = alg), eta = eta,
                               q = q, proj = proj)
end
"""
$(DocStringExtensions.TYPEDEF)

The transaction-cost optimisation of Li, Wang, Huang and Hoi (2018): one linearised step of `log ⟨w, x̂⟩` from the Price-Adjusted Allocation, soft-thresholded at the cost rate, so small trades are left at zero (TCO).

# Mathematical definition

With ``\\hat{\\boldsymbol{w}}_t`` the Price-Adjusted Allocation the fund holds before it trades,

```math
\\begin{align}
\\boldsymbol{v} &= \\eta \\left( \\frac{\\hat{\\boldsymbol{x}}_{t+1}}{\\langle \\hat{\\boldsymbol{w}}_t, \\hat{\\boldsymbol{x}}_{t+1} \\rangle} - \\bar{v} \\boldsymbol{1} \\right)\\,,\\quad
\\boldsymbol{w}_{t+1} = \\mathrm{Proj}\\left( \\hat{\\boldsymbol{w}}_t + \\operatorname{sign}(\\boldsymbol{v}) \\odot \\max(\\lvert \\boldsymbol{v} \\rvert - \\lambda, 0) \\right)\\,,\\quad
\\lambda = 10\\, \\eta\\, \\gamma\\,,
\\end{align}
```

where ``\\bar{v}`` centres the gradient so the step stays on the budget hyperplane. The step solves ``\\max_{\\boldsymbol{w}} \\log \\langle \\boldsymbol{w}, \\hat{\\boldsymbol{x}} \\rangle - \\lambda \\lVert \\boldsymbol{w} - \\hat{\\boldsymbol{w}}_t \\rVert_1`` with the log linearised at ``\\hat{\\boldsymbol{w}}_t``; the ``L_1`` term is the soft threshold. The paper's first form reads the reciprocal of the last relative, [`LaggedPrice`](@ref) at `lag = 1`, the default; its second form reads the [`MovingAverage`](@ref) of the last five levels.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TransactionCostOptimisation(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(; alg = LaggedPrice()),
        eta::Real = 10,
        gamma::Real = 0.001,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> TransactionCostOptimisation

Keywords correspond to the struct's fields. `rows_needed` forwards to `me`. `gamma` is the proportional cost rate the trader faces, which the paper leaves to the market; the threshold ``\\lambda = 10 \\eta \\gamma`` is derived from it. The paper's text sets the threshold at ``10 \\gamma``; ``10 \\eta \\gamma`` is the value its authors' own implementation uses and its reported results rest on, as Moon (2019) records, and it is the one taken here.

## Validation

  - `eta > 0`. A `DomainError` is thrown otherwise.
  - `gamma >= 0`. A `DomainError` is thrown otherwise.
  - `me` is not, and holds no, [`Online`](@ref) wrapper. An `ArgumentError` is thrown otherwise, by name.

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
    The step size on the linearised log return.
    """
    eta::T2
    """
    The proportional transaction cost rate; the soft threshold is `10 * eta * gamma`.
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

The short-term sparse portfolio optimisation of Lai, Yang, Fang and Wu (2018): a linear objective on the generalised log return of the window peak with an ``L_1`` penalty, solved by an alternating direction method, then the projection of the scaled iterate (SSPO).

# Mathematical definition

With ``\\hat{\\boldsymbol{x}}_{t+1}`` the Price Relative Forecast — the window peak over the last price by default, so ``\\hat{\\boldsymbol{x}} \\geq \\boldsymbol{1}`` — the objective reads ``\\boldsymbol{\\phi}_t = -(1.1 \\log \\hat{\\boldsymbol{x}}_{t+1} + \\boldsymbol{1})`` and solves

```math
\\begin{align}
\\min_{\\boldsymbol{b}} \\; \\langle \\boldsymbol{b}, \\boldsymbol{\\phi}_t \\rangle + \\lambda \\lVert \\boldsymbol{b} \\rVert_1 \\quad \\text{s.t.} \\quad \\boldsymbol{1}^\\intercal \\boldsymbol{b} = 1
\\end{align}
```

by the iteration, seeded at ``\\boldsymbol{b} = \\boldsymbol{g} = \\boldsymbol{w}_t`` and ``\\rho = 0``,

```math
\\begin{align}
\\boldsymbol{b} &\\leftarrow \\left( \\tfrac{\\lambda}{\\gamma} \\boldsymbol{I} + \\eta \\boldsymbol{1} \\boldsymbol{1}^\\intercal \\right)^{-1} \\left( \\tfrac{\\lambda}{\\gamma} \\boldsymbol{g} + (\\eta - \\rho) \\boldsymbol{1} - \\boldsymbol{\\phi}_t \\right)\\,,\\quad
\\boldsymbol{g} \\leftarrow \\operatorname{sign}(\\boldsymbol{b}) \\odot \\max(\\lvert \\boldsymbol{b} \\rvert - \\gamma, 0)\\,,\\quad
\\rho \\leftarrow \\rho + \\eta (\\boldsymbol{1}^\\intercal \\boldsymbol{b} - 1)\\,,
\\end{align}
```

until ``\\lvert \\boldsymbol{1}^\\intercal \\boldsymbol{b} - 1 \\rvert < \\texttt{tol}`` or `iters` steps, and ``\\boldsymbol{w}_{t+1} = \\mathrm{Proj}(\\zeta \\boldsymbol{b})``. The fixed matrix is inverted once in closed form through the Sherman–Morrison identity, so every iteration is ``O(N)``. The augmented Lagrangian is proved to have a saddle point in the paper; the final step is the Euclidean projection of a **scaled** iterate, and the scale ``\\zeta`` is what makes the answer sparse. The budget residual changes sign as the dual variable adapts, so the paper's `tol = 1e-4` is met at a zero crossing after a few hundred to a few thousand iterations, while the iterate is still moving by tenths; the scaled projection of that point and of the point `iters` steps later can land on different assets. The answer at the paper's tolerance is the paper's; a tolerance the crossings never reach runs every one of the `iters` steps.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShortTermSparsePortfolio(;
        me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(; alg = WindowPeak()),
        lambda::Real = 0.5,
        gamma::Real = 0.01,
        eta::Real = 0.005,
        zeta::Real = 500,
        iters::Integer = 10_000,
        tol::Real = 1e-4,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ShortTermSparsePortfolio

Keywords correspond to the struct's fields, and the defaults are the paper's. `rows_needed` forwards to `me`.

## Validation

  - `lambda > 0`, `gamma > 0`, `eta > 0`, `zeta > 0`, `tol > 0`. A `DomainError` is thrown otherwise.
  - `iters >= 1`. A `DomainError` is thrown otherwise.
  - `me` is not, and holds no, [`Online`](@ref) wrapper. An `ArgumentError` is thrown otherwise, by name.
  - At the update, `x̂ > 0` in every asset, so the log is defined. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ShortTermSparsePortfolio()
ShortTermSparsePortfolio
      me ┼ PriceLevelExpectedReturns
         │   alg ┼ WindowPeak
         │       │   window ┴ Int64: 5
  lambda ┼ Float64: 0.5
   gamma ┼ Float64: 0.01
     eta ┼ Float64: 0.005
    zeta ┼ Int64: 500
   iters ┼ Int64: 10000
     tol ┼ Float64: 0.0001
    proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`WindowPeak`](@ref)
  - [`ForecastTracking`](@ref)

# References

  - $(ref_dict[:lai2018sspo])
"""
struct ShortTermSparsePortfolio{T1 <: AbstractExpectedReturnsEstimator, T2 <: Real,
                                T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Integer,
                                T7 <: Real, T8 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:forecaster])
    """
    me::T1
    """
    The weight of the ``L_1`` penalty.
    """
    lambda::T2
    """
    The soft-threshold width, and the ratio `lambda / gamma` is the quadratic coupling of the iteration.
    """
    gamma::T3
    """
    The penalty on the budget equality and the dual step.
    """
    eta::T4
    """
    The scale of the iterate before its projection.
    """
    zeta::T5
    """
    Maximum number of iterations.
    """
    iters::T6
    """
    Convergence tolerance on the budget residual.
    """
    tol::T7
    """
    $(field_dict[:proj])
    """
    proj::T8
    function ShortTermSparsePortfolio(me::AbstractExpectedReturnsEstimator, lambda::Real,
                                      gamma::Real, eta::Real, zeta::Real, iters::Integer,
                                      tol::Real, proj::EuclideanProjection)
        assert_forecaster(me)
        @argcheck(lambda > zero(lambda), DomainError(lambda, "lambda must be positive"))
        @argcheck(gamma > zero(gamma), DomainError(gamma, "gamma must be positive"))
        @argcheck(eta > zero(eta), DomainError(eta, "eta must be positive"))
        @argcheck(zeta > zero(zeta), DomainError(zeta, "zeta must be positive"))
        @argcheck(iters >= 1, DomainError(iters, "iters must be at least 1"))
        @argcheck(tol > zero(tol), DomainError(tol, "tol must be positive"))
        return new{typeof(me), typeof(lambda), typeof(gamma), typeof(eta), typeof(zeta),
                   typeof(iters), typeof(tol), typeof(proj)}(me, lambda, gamma, eta, zeta,
                                                             iters, tol, proj)
    end
end
function ShortTermSparsePortfolio(me::Online, ::Real, ::Real, ::Real, ::Real, ::Integer,
                                  ::Real, ::EuclideanProjection)
    return assert_forecaster(me)
end
function ShortTermSparsePortfolio(;
                                  me::Union{<:AbstractExpectedReturnsEstimator, <:Online} = PriceLevelExpectedReturns(;
                                                                                                                      alg = WindowPeak()),
                                  lambda::Real = 0.5, gamma::Real = 0.01, eta::Real = 0.005,
                                  zeta::Real = 500, iters::Integer = 10_000,
                                  tol::Real = 1e-4,
                                  proj::EuclideanProjection = EuclideanProjection())::ShortTermSparsePortfolio
    return ShortTermSparsePortfolio(me, lambda, gamma, eta, zeta, iters, tol, proj)
end
function port_opt_view(alg::ShortTermSparsePortfolio, i, args...)
    return ShortTermSparsePortfolio(; me = port_opt_view(alg.me, i, args...),
                                    lambda = alg.lambda, gamma = alg.gamma, eta = alg.eta,
                                    zeta = alg.zeta, iters = alg.iters, tol = alg.tol,
                                    proj = alg.proj)
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
    b = collect(w)
    g = copy(b)
    rho = zero(eltype(b)) * alg.eta
    a = alg.lambda / alg.gamma
    N = length(w)
    for _ in 1:(alg.iters)
        rhs = a .* g .+ (alg.eta - rho) .- phi
        # `(a I + η 1 1ᵀ)⁻¹ v = v / a − η (1ᵀ v) / (a (a + η N)) 1` by Sherman–Morrison.
        b = rhs ./ a .- alg.eta * sum(rhs) / (a * (a + alg.eta * N))
        g = sign.(b) .* max.(abs.(b) .- alg.gamma, zero(alg.gamma))
        res = sum(b) - one(eltype(b))
        rho += alg.eta * res
        if abs(res) < alg.tol
            break
        end
    end
    return st, project(alg.proj, set, alg.zeta .* b, price_adjusted_allocation(w, x))
end
export ForecastReversion, MovingAverageReversion, ExponentialMovingAverageReversion,
       RobustMedianReversion, ReweightedPriceRelativeTracking, GaussianWeightingReversion,
       LocalAdaptiveLearning, ForecastTracking, PeakPriceTracking,
       AdaptiveInputCompositeTrend, TrendPromotePriceTracking, KernelTrendTracking,
       KernelTrendPatternTracking, TransactionCostOptimisation, ShortTermSparsePortfolio
public ForecasterState, forecast_relative
