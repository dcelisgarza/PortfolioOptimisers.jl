# https://portfoliooptimizationbook.com/slides/slides-index-tracking.pdf
"""
$(DocStringExtensions.TYPEDEF)

Represents the Risk Tracking Error configuration for benchmark weight tracking.

`RiskTrackingError` specifies that tracking error against a benchmark should be measured as a risk quantity (rather than a norm). It wraps a `WeightsTracking` benchmark, a risk measure `r`, a scalar error tolerance `err`, and a tracking algorithm `alg`.

# Fields

  - `tr`: Benchmark weights tracking specification.
  - `r`: Risk measure used to compute the tracking error.
  - `err`: Scalar error tolerance (non-negative finite number).
  - `alg`: Tracking algorithm (`IndependentVariableTracking` or `DependentVariableTracking`).

# Constructors

    RiskTrackingError(;
        tr::WeightsTracking,
        r::AbstractBaseRiskMeasure = StandardDeviation(),
        err::Number = 0.0,
        alg::VariableTracking = IndependentVariableTracking()
    ) -> RiskTrackingError

Keywords correspond to the struct's fields.

## Validation

  - `err` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Related
  - [`set_tracking_error_constraints!`](@ref)

  - [`TrackingRiskMeasure`](@ref)
  - [`RiskTrackingRiskMeasure`](@ref)
  - [`WeightsTracking`](@ref)
  - [`IndependentVariableTracking`](@ref)
  - [`DependentVariableTracking`](@ref)
"""
@concrete struct RiskTrackingError <: AbstractTracking
    tr
    r
    err
    alg
    function RiskTrackingError(tr::WeightsTracking, r::AbstractBaseRiskMeasure, err::Number,
                               alg::VariableTracking)
        assert_nonempty_nonneg_finite_val(err, :err)
        r = no_bounds_no_risk_expr_risk_measure(r)
        return new{typeof(tr), typeof(r), typeof(err), typeof(alg)}(tr, r, err, alg)
    end
end
function RiskTrackingError(; tr::WeightsTracking,
                           r::AbstractBaseRiskMeasure = StandardDeviation(),
                           err::Number = 0.0,
                           alg::VariableTracking = IndependentVariableTracking())
    return RiskTrackingError(tr, r, err, alg)
end
function tracking_view(::Nothing, args...)
    return nothing
end
function tracking_view(tr::RiskTrackingError, i, X::MatNum)
    return RiskTrackingError(; tr = tracking_view(tr.tr, i),
                             r = risk_measure_view(tr.r, i, X), err = tr.err, alg = tr.alg)
end
function factory(tr::RiskTrackingError, pr::AbstractPriorResult, slv::Any, ucs::Any,
                 w::Option{<:VecNum} = nothing, args...; kwargs...)
    return RiskTrackingError(; tr = factory(tr.tr, w),
                             r = factory(tr.r, pr, slv, ucs, w, args...; kwargs...),
                             err = tr.err, alg = tr.alg)
end
function needs_previous_weights(tr::RiskTrackingError)
    return (needs_previous_weights(tr.tr) || needs_previous_weights(tr.r))
end
function factory(tr::RiskTrackingError, w::VecNum)
    return RiskTrackingError(; tr = factory(tr.tr, w), r = factory(tr.r, w), err = tr.err,
                             alg = tr.alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Tracking Error risk measure.

`TrackingRiskMeasure` penalises portfolio deviation from a benchmark by computing a norm of the difference between portfolio returns and a benchmark return series or benchmark weights. The tracking error can be defined using returns-based or weights-based benchmarks, and the norm is configurable.

# Mathematical Definition

Let ``\\boldsymbol{x}`` be the portfolio returns series, ``\\boldsymbol{b}`` the benchmark returns, and ``N_T`` the number of observations. The ``L^2`` tracking error is:

```math
\\mathrm{TE}(\\boldsymbol{w}) = \\sqrt{\\frac{1}{N_T}\\lVert \\boldsymbol{x} - \\boldsymbol{b} \\rVert_2^2}\\,.
```

Other norms can be selected via the `alg` field.

# Fields

  - `settings`: Risk measure configuration.
  - `tr`: Tracking algorithm specifying the benchmark (weights- or returns-based).
  - `alg`: Norm type for the tracking error (`L2Tracking`, `SquaredL2Tracking`, etc.).

# Constructors

    TrackingRiskMeasure(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        tr::AbstractTrackingAlgorithm,
        alg::NormTracking = L2Tracking()
    ) -> TrackingRiskMeasure

Keywords correspond to the struct's fields.

# Functor

    (r::TrackingRiskMeasure)(w::VecNum, X::MatNum, fees = nothing)

Computes the Tracking Error of a portfolio weight vector `w`.

## Arguments

  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee structure.

# Examples

```jldoctest
julia> TrackingRiskMeasure(; tr = ReturnsTracking(; b = zeros(100)))
TrackingRiskMeasure
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        tr ┼ ReturnsTracking
       alg ┴ L2Tracking
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`TurnoverRiskMeasure`](@ref)
  - [`RiskTrackingRiskMeasure`](@ref)
  - [`AbstractTrackingAlgorithm`](@ref)
  - [`NormTracking`](@ref)
"""
@concrete struct TrackingRiskMeasure <: RiskMeasure
    settings
    tr
    alg
    function TrackingRiskMeasure(settings::RiskMeasureSettings,
                                 tr::AbstractTrackingAlgorithm, alg::NormTracking)
        return new{typeof(settings), typeof(tr), typeof(alg)}(settings, tr, alg)
    end
end
function TrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             tr::AbstractTrackingAlgorithm,
                             alg::NormTracking = L2Tracking())
    return TrackingRiskMeasure(settings, tr, alg)
end
function (r::TrackingRiskMeasure)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    benchmark = tracking_benchmark(r.tr, X)
    return norm_tracking(r.alg, calc_net_returns(w, X, fees), benchmark, size(X, 1))
end
function (r::TrackingRiskMeasure{ReturnsTracking})(X::VecNum)
    benchmark = tracking_benchmark(r.tr, X)
    return norm_tracking(r.alg, X, benchmark, length(X))
end
function (r::TrackingRiskMeasure{WeightsTracking})(::VecNum)
    throw(MethodError(r,
                      "Tracking risk measure using the `WeightsTracking` algorithm cannot be computed for a prediction of portfolio returns because there are no weights."))
end
function risk_measure_view(r::TrackingRiskMeasure, i, args...)
    tr = tracking_view(r.tr, i)
    return TrackingRiskMeasure(; settings = r.settings, tr = tr, alg = r.alg)
end
function needs_previous_weights(r::TrackingRiskMeasure)
    return needs_previous_weights(r.tr)
end
function factory(r::TrackingRiskMeasure, w::VecNum)
    return TrackingRiskMeasure(; settings = r.settings, tr = factory(r.tr, w), alg = r.alg)
end
function factory(r::TrackingRiskMeasure, ::Any, ::Any, ::Any, w::VecNum, args...; kwargs...)
    return factory(r, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Risk Tracking risk measure.

`RiskTrackingRiskMeasure` computes the deviation of portfolio risk from a benchmark portfolio risk, using any base risk measure. Two modes are supported:

- **Independent** (`IndependentVariableTracking`): computes the risk of the weight difference ``\\boldsymbol{w} - \\boldsymbol{w}_b``.
- **Dependent** (`DependentVariableTracking`): computes the absolute difference between the portfolio risk and the benchmark risk.

# Mathematical Definition

**Independent mode:**

```math
\\mathrm{RkTrack}_{\\mathrm{indep}}(\\boldsymbol{w}) = \\rho(\\boldsymbol{w} - \\boldsymbol{w}_b)\\,,
```

**Dependent mode:**

```math
\\mathrm{RkTrack}_{\\mathrm{dep}}(\\boldsymbol{w}) = |\\rho(\\boldsymbol{w}) - \\rho(\\boldsymbol{w}_b)|\\,,
```

where ``\\boldsymbol{w}_b`` are the benchmark weights and ``\\rho`` is the chosen risk measure.

# Fields

  - `settings`: Risk measure configuration.
  - `tr`: Benchmark weights tracking specification.
  - `r`: Risk measure for computing the tracking deviation.
  - `alg`: Tracking mode (`IndependentVariableTracking` or `DependentVariableTracking`).

# Constructors

    RiskTrackingRiskMeasure(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        tr::WeightsTracking,
        r::AbstractBaseRiskMeasure = Variance(),
        alg::VariableTracking = IndependentVariableTracking()
    ) -> RiskTrackingRiskMeasure

Keywords correspond to the struct's fields.

# Functor

    (r::RiskTrackingRiskMeasure)(w::VecNum, X::MatNum, fees = nothing)

Computes the Risk Tracking deviation of a portfolio weight vector `w`.

## Arguments

  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee structure.

# Examples

```jldoctest
julia> RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = [0.5, 0.5]))
RiskTrackingRiskMeasure
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        tr ┼ WeightsTracking
         r ┼ Variance
       alg ┴ IndependentVariableTracking
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`TrackingRiskMeasure`](@ref)
  - [`TurnoverRiskMeasure`](@ref)
  - [`WeightsTracking`](@ref)
  - [`IndependentVariableTracking`](@ref)
  - [`DependentVariableTracking`](@ref)
"""
@concrete struct RiskTrackingRiskMeasure <: RiskMeasure
    settings
    tr
    r
    alg
    function RiskTrackingRiskMeasure(settings::RiskMeasureSettings, tr::WeightsTracking,
                                     r::AbstractBaseRiskMeasure, alg::VariableTracking)
        if isa(alg, DependentVariableTracking) && isa(r, QuadExpressionRiskMeasures)
            @warn("Risk measures that produce JuMP.QuadExpr risk expressions are not guaranteed to work. The variance with SDP constraints works because the risk measure is the trace of a matrix, an affine expression.")
        end
        r = no_bounds_no_risk_expr_risk_measure(r)
        return new{typeof(settings), typeof(tr), typeof(r), typeof(alg)}(settings, tr, r,
                                                                         alg)
    end
end
function RiskTrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 tr::WeightsTracking,
                                 r::AbstractBaseRiskMeasure = Variance(),
                                 alg::VariableTracking = IndependentVariableTracking())
    return RiskTrackingRiskMeasure(settings, tr, r, alg)
end
function (r::RiskTrackingRiskMeasure{<:Any, <:Any, <:AbstractBaseRiskMeasure,
                                     <:IndependentVariableTracking})(w::VecNum, X::MatNum,
                                                                     fees::Option{<:Fees} = nothing)
    wb = r.tr.w
    wd = w - wb
    return expected_risk(r.r, wd, X, fees)
end
function (r::RiskTrackingRiskMeasure{<:Any, <:Any, <:AbstractBaseRiskMeasure,
                                     <:DependentVariableTracking})(w::VecNum, X::MatNum,
                                                                   fees::Option{<:Fees} = nothing)
    wb = r.tr.w
    r1 = expected_risk(r.r, w, X, fees)
    r2 = expected_risk(r.r, wb, X, fees)
    return abs(r1 - r2)
end
function risk_measure_view(r::RiskTrackingRiskMeasure, i, X::MatNum)
    tr = tracking_view(r.tr, i)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = tr,
                                   r = risk_measure_view(r.r, i, X), alg = r.alg)
end
function factory(r::RiskTrackingRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = r.tr,
                                   r = factory(r.r, pr, args...; kwargs...), alg = r.alg)
end
function needs_previous_weights(r::RiskTrackingRiskMeasure)
    return (needs_previous_weights(r.tr) || needs_previous_weights(r.r))
end
function factory(r::RiskTrackingRiskMeasure, w::VecNum)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = factory(r.tr, w),
                                   r = factory(r.r, w), alg = r.alg)
end
const TrRM = Union{<:TrackingRiskMeasure, <:RiskTrackingRiskMeasure}

export TrackingRiskMeasure, RiskTrackingRiskMeasure, RiskTrackingError
