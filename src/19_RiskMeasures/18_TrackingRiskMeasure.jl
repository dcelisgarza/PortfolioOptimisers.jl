# https://portfoliooptimizationbook.com/slides/slides-index-tracking.pdf
"""
$(DocStringExtensions.TYPEDEF)

Represents the Risk Tracking Error configuration for benchmark weight tracking.

`RiskTrackingError` specifies tracking error measurement against a benchmark as a risk quantity (rather than a norm). It wraps a `WeightsTracking` benchmark, a risk measure `r`, a scalar error tolerance `err`, and a tracking algorithm `alg`.

# Fields

$(DocStringExtensions.FIELDS)

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
    "$(field_dict[:tr_spec])"
    tr
    "$(field_dict[:r_risk])"
    r
    "$(field_dict[:err])"
    err
    "$(field_dict[:tralg])"
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
                           alg::VariableTracking = IndependentVariableTracking())::RiskTrackingError
    return RiskTrackingError(tr, r, err, alg)
end
"""
    tracking_view(::Nothing, args...)

Return `nothing` unchanged.

Identity pass-through for optional tracking configuration fields.

# Related

  - [`tracking_view`](@ref)
  - [`RiskTrackingError`](@ref)
"""
function tracking_view(::Nothing, args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`RiskTrackingError`](@ref) `tr` sliced to asset indices `i`.

Slices both the inner tracking benchmark and the risk measure for cluster-based optimisation.

# Related

  - [`RiskTrackingError`](@ref)
  - [`tracking_view`](@ref)
  - [`risk_measure_view`](@ref)
"""
function tracking_view(tr::RiskTrackingError, i, X::MatNum)
    return RiskTrackingError(; tr = tracking_view(tr.tr, i),
                             r = risk_measure_view(tr.r, i, X), err = tr.err, alg = tr.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RiskTrackingError`](@ref) updating the inner benchmark and risk measure from the prior result and solver context.

# Related

  - [`RiskTrackingError`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
"""
function factory(tr::RiskTrackingError, pr::AbstractPriorResult, slv::Any, ucs::Any,
                 w::Option{<:VecNum} = nothing, args...; kwargs...)::RiskTrackingError
    return RiskTrackingError(; tr = factory(tr.tr, w),
                             r = factory(tr.r, pr, slv, ucs, w, args...; kwargs...),
                             err = tr.err, alg = tr.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`RiskTrackingError`](@ref) `tr` requires previous portfolio weights.

Returns `true` if either the inner tracking benchmark or the inner risk measure requires previous weights.

# Related

  - [`RiskTrackingError`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(tr::RiskTrackingError)
    return (needs_previous_weights(tr.tr) || needs_previous_weights(tr.r))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RiskTrackingError`](@ref) updating the inner benchmark and risk measure from new portfolio weights `w`.

# Related

  - [`RiskTrackingError`](@ref)
  - [`factory`](@ref)
"""
function factory(tr::RiskTrackingError, w::VecNum)::RiskTrackingError
    return RiskTrackingError(; tr = factory(tr.tr, w), r = factory(tr.r, w), err = tr.err,
                             alg = tr.alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Tracking Error risk measure.

`TrackingRiskMeasure` penalises portfolio deviation from a benchmark by computing a norm of the difference between portfolio returns and a benchmark return series or benchmark weights. The tracking error is defined using returns-based or weights-based benchmarks, and the norm is configurable.

# Mathematical definition

Let ``\\boldsymbol{x}`` be the portfolio returns series, ``\\boldsymbol{b}`` the benchmark returns, and ``N_T`` the number of observations. The ``L^2`` tracking error is:

```math
\\begin{align}
\\mathrm{TE}(\\boldsymbol{w}) &= \\sqrt{\\frac{1}{N_T}\\lVert \\boldsymbol{x} - \\boldsymbol{b} \\rVert_2^2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}(\\boldsymbol{w})``: ``L^2`` tracking error of the portfolio.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{x}``: Portfolio returns series ``N_T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark returns series ``N_T \\times 1``.
  - ``N_T``: Number of observations.

Other norms can be selected via the `alg` field.

# Fields

$(DocStringExtensions.FIELDS)

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
julia> TrackingRiskMeasure(; tr = ReturnsTracking(; w = [0.1, -0.2, 0.3]))
TrackingRiskMeasure
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        tr ┼ ReturnsTracking
           │   w ┴ Vector{Float64}: [0.1, -0.2, 0.3]
       alg ┼ L2Tracking
           │   ddof ┴ Int64: 1
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
    "$(field_dict[:settings_rm])"
    settings
    "$(field_dict[:tr_spec])"
    tr
    "$(field_dict[:tralg])"
    alg
    function TrackingRiskMeasure(settings::RiskMeasureSettings,
                                 tr::AbstractTrackingAlgorithm, alg::NormTracking)
        return new{typeof(settings), typeof(tr), typeof(alg)}(settings, tr, alg)
    end
end
function TrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             tr::AbstractTrackingAlgorithm,
                             alg::NormTracking = L2Tracking())::TrackingRiskMeasure
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`TrackingRiskMeasure`](@ref) `r` sliced to asset indices `i`.

Slices the inner tracking specification for cluster-based optimisation.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`risk_measure_view`](@ref)
  - [`tracking_view`](@ref)
"""
function risk_measure_view(r::TrackingRiskMeasure, i, args...)
    tr = tracking_view(r.tr, i)
    return TrackingRiskMeasure(; settings = r.settings, tr = tr, alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`TrackingRiskMeasure`](@ref) `r` requires previous portfolio weights.

Delegates to the inner tracking specification.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(r::TrackingRiskMeasure)
    return needs_previous_weights(r.tr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`TrackingRiskMeasure`](@ref) updating the inner tracking specification with new weights `w`.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::TrackingRiskMeasure, w::VecNum)
    return TrackingRiskMeasure(; settings = r.settings, tr = factory(r.tr, w), alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`TrackingRiskMeasure`](@ref) from a full optimisation context, forwarding `w` to `factory(r, w)`.

Ignores prior result, solver, and uncertainty set arguments.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::TrackingRiskMeasure, ::Any, ::Any, ::Any, w::VecNum, args...; kwargs...)
    return factory(r, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Risk Tracking risk measure.

`RiskTrackingRiskMeasure` computes the deviation of portfolio risk from a benchmark portfolio risk, using any base risk measure. Two modes are supported:

  - **Independent** (`IndependentVariableTracking`): computes the risk of the weight difference ``\\boldsymbol{w} - \\boldsymbol{w}_b``.
  - **Dependent** (`DependentVariableTracking`): computes the absolute difference between the portfolio risk and the benchmark risk.

# Mathematical definition

**Independent mode:**

```math
\\begin{align}
\\mathrm{RkTrack}_{\\mathrm{indep}}(\\boldsymbol{w}) &= \\rho(\\boldsymbol{w} - \\boldsymbol{w}_b)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RkTrack}_{\\mathrm{indep}}(\\boldsymbol{w})``: Risk of the weight difference between portfolio and benchmark.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{w}_b``: Benchmark portfolio weights vector ``N \\times 1``.
  - ``\\rho``: Chosen base risk measure.

**Dependent mode:**

```math
\\begin{align}
\\mathrm{RkTrack}_{\\mathrm{dep}}(\\boldsymbol{w}) &= |\\rho(\\boldsymbol{w}) - \\rho(\\boldsymbol{w}_b)|\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RkTrack}_{\\mathrm{dep}}(\\boldsymbol{w})``: Absolute difference between portfolio risk and benchmark risk.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{w}_b``: Benchmark portfolio weights vector ``N \\times 1``.
  - ``\\rho``: Chosen base risk measure.

# Fields

$(DocStringExtensions.FIELDS)

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
           │    fees ┼ nothing
           │       w ┼ Vector{Float64}: [0.5, 0.5]
           │   fixed ┴ Bool: false
         r ┼ Variance
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Int64: 1
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: false
           │      sigma ┼ nothing
           │       chol ┼ nothing
           │         rc ┼ nothing
           │        alg ┴ SquaredSOCRiskExpr()
       alg ┴ IndependentVariableTracking()
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
    "$(field_dict[:settings_rm])"
    settings
    "$(field_dict[:tr_spec])"
    tr
    "$(field_dict[:r_risk])"
    r
    "$(field_dict[:tralg])"
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`RiskTrackingRiskMeasure`](@ref) `r` sliced to asset indices `i`.

Slices both the inner tracking benchmark and the risk measure for cluster-based optimisation.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`risk_measure_view`](@ref)
  - [`tracking_view`](@ref)
"""
function risk_measure_view(r::RiskTrackingRiskMeasure, i, X::MatNum)
    tr = tracking_view(r.tr, i)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = tr,
                                   r = risk_measure_view(r.r, i, X), alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RiskTrackingRiskMeasure`](@ref) updating the inner risk measure from the prior result.

The inner tracking benchmark is preserved; the risk measure is updated via `factory`.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
"""
function factory(r::RiskTrackingRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = r.tr,
                                   r = factory(r.r, pr, args...; kwargs...), alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`RiskTrackingRiskMeasure`](@ref) `r` requires previous portfolio weights.

Returns `true` if either the inner tracking benchmark or the inner risk measure requires previous weights.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(r::RiskTrackingRiskMeasure)
    return (needs_previous_weights(r.tr) || needs_previous_weights(r.r))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RiskTrackingRiskMeasure`](@ref) updating the inner benchmark and risk measure from new portfolio weights `w`.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::RiskTrackingRiskMeasure, w::VecNum)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = factory(r.tr, w),
                                   r = factory(r.r, w), alg = r.alg)
end
"""
    const TrRM = Union{<:TrackingRiskMeasure, <:RiskTrackingRiskMeasure}

Union of tracking risk measures used for dispatch on factory methods and expected risk computations.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`RiskTrackingRiskMeasure`](@ref)
"""
const TrRM = Union{<:TrackingRiskMeasure, <:RiskTrackingRiskMeasure}

export TrackingRiskMeasure, RiskTrackingRiskMeasure, RiskTrackingError
