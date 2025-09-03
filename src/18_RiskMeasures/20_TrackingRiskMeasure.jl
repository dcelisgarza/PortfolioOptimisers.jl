struct RiskTrackingError{T1,T2,T3,T4} <: AbstractTracking
    tracking::T1
    r::T2
    err::T3
    alg::T4
end
function RiskTrackingError(;
    tracking::WeightsTracking,
    r::AbstractBaseRiskMeasure = Variance(),
    err::Real = 0.0,
    alg::VariableTracking = IndependentVariableTracking(),
)
    @argcheck(isfinite(err) && err >= zero(err))
    r = no_bounds_no_risk_expr_risk_measure(r)
    return RiskTrackingError(tracking, r, err, alg)
end
function tracking_view(::Nothing, args...)
    return nothing
end
function tracking_view(tracking::RiskTrackingError, i::AbstractVector, X::AbstractMatrix)
    return RiskTrackingError(;
        tracking = tracking_view(tracking.tracking, i),
        r = risk_measure_view(tracking.r, i, X),
        err = tracking.err,
        alg = tracking.alg,
    )
end
function factory(
    tracking::RiskTrackingError,
    prior::AbstractPriorResult,
    args...;
    kwargs...,
)
    return RiskTrackingError(;
        tracking = tracking.tracking,
        r = factory(tracking.r, prior, args...; kwargs...),
        err = tracking.err,
        alg = tracking.alg,
    )
end
function factory(tracking::RiskTrackingError, w::AbstractVector)
    return RiskTrackingError(;
        tracking = factory(tracking.tracking, w),
        r = tracking.r,
        err = tracking.err,
        alg = tracking.alg,
    )
end
struct TrackingRiskMeasure{T1,T2,T3} <: RiskMeasure
    settings::T1
    tracking::T2
    alg::T3
end
function TrackingRiskMeasure(;
    settings::RiskMeasureSettings = RiskMeasureSettings(),
    tracking::AbstractTrackingAlgorithm,
    alg::NormTracking = SOCTracking(),
)
    return TrackingRiskMeasure(settings, tracking, alg)
end
function (r::TrackingRiskMeasure)(
    w::AbstractVector,
    X::AbstractMatrix,
    fees::Union{Nothing,<:Fees} = nothing,
)
    benchmark = tracking_benchmark(r.tracking, X)
    return norm_tracking(r.alg, calc_net_returns(w, X, fees), benchmark, size(X, 1))
end
function risk_measure_view(r::TrackingRiskMeasure, i::AbstractVector, args...)
    tracking = tracking_view(r.tracking, i)
    return TrackingRiskMeasure(; settings = r.settings, tracking = tracking, alg = r.alg)
end
function factory(r::TrackingRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    return TrackingRiskMeasure(; settings = r.settings, tracking = r.tracking, alg = r.alg)
end
function factory(r::TrackingRiskMeasure, w::AbstractVector)
    return TrackingRiskMeasure(;
        settings = r.settings,
        tracking = factory(r.tracking, w),
        alg = r.alg,
    )
end
struct RiskTrackingRiskMeasure{T1,T2,T3,T4} <: RiskMeasure
    settings::T1
    tracking::T2
    r::T3
    alg::T4
end
function RiskTrackingRiskMeasure(;
    settings::RiskMeasureSettings = RiskMeasureSettings(),
    tracking::WeightsTracking,
    r::AbstractBaseRiskMeasure = Variance(),
    alg::VariableTracking = IndependentVariableTracking(),
)
    r = no_bounds_no_risk_expr_risk_measure(r)
    return RiskTrackingRiskMeasure(settings, tracking, r, alg)
end
function (
    r::RiskTrackingRiskMeasure{
        <:Any,
        <:Any,
        <:AbstractBaseRiskMeasure,
        <:IndependentVariableTracking,
    }
)(
    w::AbstractVector,
    X::AbstractMatrix,
    fees::Union{Nothing,<:Fees} = nothing,
)
    wb = r.tracking.w
    wd = w - wb
    return expected_risk(r.r, wd, X, fees)
end
function (
    r::RiskTrackingRiskMeasure{
        <:Any,
        <:Any,
        <:AbstractBaseRiskMeasure,
        <:DependentVariableTracking,
    }
)(
    w::AbstractVector,
    X::AbstractMatrix,
    fees::Union{Nothing,<:Fees} = nothing,
)
    wb = r.tracking.w
    r1 = expected_risk(r.r, w, X, fees)
    r2 = expected_risk(r.r, wb, X, fees)
    return abs(r1 - r2)
end
function risk_measure_view(r::RiskTrackingRiskMeasure, i::AbstractVector, X::AbstractMatrix)
    tracking = tracking_view(r.tracking, i)
    return RiskTrackingRiskMeasure(;
        settings = r.settings,
        tracking = tracking,
        r = risk_measure_view(r.r, i, X),
        alg = r.alg,
    )
end
function factory(r::RiskTrackingRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    return RiskTrackingRiskMeasure(;
        settings = r.settings,
        tracking = r.tracking,
        r = factory(r.r, prior, args...; kwargs...),
        alg = r.alg,
    )
end
function factory(r::RiskTrackingRiskMeasure, w::AbstractVector)
    return RiskTrackingRiskMeasure(;
        settings = r.settings,
        tracking = factory(r.tracking, w),
        r = r.r,
        alg = r.alg,
    )
end

export SOCTracking,
    NOCTracking,
    IndependentVariableTracking,
    DependentVariableTracking,
    TrackingRiskMeasure,
    RiskTrackingRiskMeasure
