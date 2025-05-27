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
#=
struct VolTrackingRiskMeasure{T1 <: RiskMeasureSettings, T2 <: WeightsTracking,
                              T3 <: Union{Nothing, <:AbstractMatrix},
                              T4 <: Union{<:QuadSqrtRiskExpr, <:SOCRiskExpr}} <: RiskMeasure
    settings::T1
    tracking::T2
    sigma::T3
    formulation::T4
end
function VolTrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                tracking::WeightsTracking,
                                sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                                formulation::Union{<:QuadSqrtRiskExpr, <:SOCRiskExpr} = SqrtRiskExpr())
    return VolTrackingRiskMeasure{typeof(settings), typeof(tracking), typeof(sigma),
                                  typeof(formulation)}(settings, tracking, sigma,
                                                       formulation)
end
function (r::VolTrackingRiskMeasure{<:Any, <:Any, <:Any, <:SqrtRiskExpr})(w::AbstractVector)
    wb = r.tracking.w
    wd = w - wb
    sigma = r.sigma
    return sqrt(dot(wd, sigma, wd))
end
function (r::VolTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                    <:Union{<:QuadRiskExpr, <:SOCRiskExpr}})(w::AbstractVector)
    wb = r.tracking.w
    wd = w - wb
    sigma = r.sigma
    return dot(wd, sigma, wd)
end
function factory(r::VolTrackingRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    sigma = nothing_scalar_array_factory(r.sigma, prior.sigma)
    return VolTrackingRiskMeasure(; settings = r.settings, tracking = r.tracking,
                                  sigma = sigma, formulation = r.formulation)
end
function risk_measure_view(r::VolTrackingRiskMeasure, i::AbstractVector, args...)
    tracking = tracking_view(r.tracking, i)
    sigma = nothing_scalar_array_view(r.sigma, i)
    return VolTrackingRiskMeasure(; settings = r.settings, tracking = tracking,
                                  sigma = sigma, formulation = r.formulation)
end
=#
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
    return norm(r1 - r2)
end
function factory(r::RiskTrackingRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    return RiskTrackingRiskMeasure(; settings = r.settings, tracking = r.tracking,
                                   r = factory(r.r, prior, args...; kwargs...),
                                   formulation = r.formulation)
end
function risk_measure_view(r::RiskTrackingRiskMeasure, i::AbstractVector, X::AbstractMatrix)
    tracking = tracking_view(r.tracking, i)
    return RiskTrackingRiskMeasure(; settings = r.settings, tracking = tracking,
                                   r = risk_measure_view(r.r, i, X),
                                   formulation = r.formulation)
end

export SOCTracking, NOCTracking, IndependentVariableTracking, DependentVariableTracking,
       TrackingRiskMeasure, VolTrackingRiskMeasure, RiskTrackingRiskMeasure
