struct RiskTrackingError{T1 <: WeightsTracking, T2 <: AbstractBaseRiskMeasure, T3 <: Real,
                         T4 <: VariableTracking} <: AbstractTracking
    tracking::T1
    r::T2
    err::T3
    formulation::T4
end
function RiskTrackingError(; tracking::WeightsTracking,
                           r::AbstractBaseRiskMeasure = Variance(), err::Real = 0.0,
                           formulation::VariableTracking = IndependentVariableTracking())
    @smart_assert(isfinite(err) && err >= zero(err))
    r = no_bounds_no_risk_expr_risk_measure(r)
    return RiskTrackingError{typeof(tracking), typeof(r), typeof(err), typeof(formulation)}(tracking,
                                                                                            r,
                                                                                            err,
                                                                                            formulation)
end
function tracking_view(::Nothing, args...)
    return nothing
end
function tracking_view(tracking::RiskTrackingError, i::AbstractVector, X::AbstractMatrix)
    return RiskTrackingError(; tracking = tracking_view(tracking.tracking, i),
                             r = risk_measure_view(tracking.r, i, X), err = tracking.err,
                             formulation = tracking.formulation)
end
function factory(tracking::RiskTrackingError, prior::AbstractPriorResult, args...;
                 kwargs...)
    return RiskTrackingError(; tracking = tracking.tracking,
                             r = factory(tracking.r, prior, args...; kwargs...),
                             err = tracking.err, formulation = tracking.formulation)
end
function factory(tracking::RiskTrackingError, w::AbstractVector)
    return RiskTrackingError(; tracking = factory(tracking.tracking, w), r = tracking.r,
                             err = tracking.err, formulation = tracking.formulation)
end
struct TrackingRiskMeasure{T1 <: RiskMeasureSettings, T2 <: AbstractTrackingAlgorithm,
                           T3 <: NormTracking} <: RiskMeasure
    settings::T1
    tracking::T2
    formulation::T3
end
function TrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             tracking::AbstractTrackingAlgorithm,
                             formulation::NormTracking = SOCTracking())
    return TrackingRiskMeasure{typeof(settings), typeof(tracking), typeof(formulation)}(settings,
                                                                                        tracking,
                                                                                        formulation)
end
function (r::TrackingRiskMeasure)(w::AbstractVector, X::AbstractMatrix,
                                  fees::Union{Nothing, <:Fees} = nothing)
    benchmark = tracking_benchmark(r.tracking, X)
    return norm_tracking(r.formulation, calc_net_returns(w, X, fees), benchmark, size(X, 1))
end
function risk_measure_view(r::TrackingRiskMeasure, i::AbstractVector, args...)
    tracking = tracking_view(r.tracking, i)
    return TrackingRiskMeasure(; settings = r.settings, tracking = tracking,
                               formulation = r.formulation)
end
function factory(r::TrackingRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    return TrackingRiskMeasure(; settings = r.settings, tracking = r.tracking,
                               formulation = r.formulation)
end
function factory(r::TrackingRiskMeasure, w::AbstractVector)
    return TrackingRiskMeasure(; settings = r.settings, tracking = factory(r.tracking, w),
                               formulation = r.formulation)
end
struct RiskTrackingRiskMeasure{T1 <: RiskMeasureSettings, T2 <: WeightsTracking,
                               T3 <: AbstractBaseRiskMeasure, T4 <: VariableTracking} <:
       RiskMeasure
    settings::T1
    tracking::T2
    r::T3
    formulation::T4
end
function RiskTrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 tracking::WeightsTracking,
                                 r::AbstractBaseRiskMeasure = Variance(),
                                 formulation::VariableTracking = IndependentVariableTracking())
    r = no_bounds_no_risk_expr_risk_measure(r)
    return RiskTrackingRiskMeasure{typeof(settings), typeof(tracking), typeof(r),
                                   typeof(formulation)}(settings, tracking, r, formulation)
end
function (r::RiskTrackingRiskMeasure{<:Any, <:Any, <:AbstractBaseRiskMeasure,
                                     <:IndependentVariableTracking})(w::AbstractVector,
                                                                     X::AbstractMatrix,
                                                                     fees::Union{Nothing,
                                                                                 <:Fees} = nothing)
    wb = r.tracking.w
    wd = w - wb
    return expected_risk(r.r, wd, X, fees)
end
function (r::RiskTrackingRiskMeasure{<:Any, <:Any, <:AbstractBaseRiskMeasure,
                                     <:DependentVariableTracking})(w::AbstractVector,
                                                                   X::AbstractMatrix,
                                                                   fees::Union{Nothing,
                                                                               <:Fees} = nothing)
    wb = r.tracking.w
    r1 = expected_risk(r.r, w, X, fees)
    r2 = expected_risk(r.r, wb, X, fees)
    return abs(r1 - r2)
end
function risk_measure_view(r::RiskTrackingRiskMeasure, i::AbstractVector, X::AbstractMatrix)
    tracking = tracking_view(r.tracking, i)
    return RiskTrackingRiskMeasure(; settings = r.settings, tracking = tracking,
                                   r = risk_measure_view(r.r, i, X),
                                   formulation = r.formulation)
end
function factory(r::RiskTrackingRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    return RiskTrackingRiskMeasure(; settings = r.settings, tracking = r.tracking,
                                   r = factory(r.r, prior, args...; kwargs...),
                                   formulation = r.formulation)
end
function factory(r::RiskTrackingRiskMeasure, w::AbstractVector)
    return RiskTrackingRiskMeasure(; settings = r.settings,
                                   tracking = factory(r.tracking, w), r = r.r,
                                   formulation = r.formulation)
end

export SOCTracking, NOCTracking, IndependentVariableTracking, DependentVariableTracking,
       TrackingRiskMeasure, RiskTrackingRiskMeasure
