abstract type TrackingFormulation <: AbstractAlgorithm end
struct SOCTracking <: TrackingFormulation end
struct NOCTracking <: TrackingFormulation end
struct TrackingRiskMeasure{T1 <: RiskMeasureSettings, T2 <: AbstractTracking,
                           T3 <: TrackingFormulation} <: RiskMeasure
    settings::T1
    tracking::T2
    formulation::T3
end
function TrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             tracking::AbstractTracking,
                             formulation::TrackingFormulation = SOCTracking())
    return TrackingRiskMeasure{typeof(settings), typeof(tracking), typeof(formulation)}(settings,
                                                                                        tracking,
                                                                                        formulation)
end
function (r::TrackingRiskMeasure{<:Any, <:Any, <:SOCTracking})(w::AbstractVector,
                                                               X::AbstractMatrix,
                                                               fees::Union{Nothing, <:Fees} = nothing)
    benchmark = tracking_benchmark(r.tracking, X)
    return norm(calc_net_returns(w, X, fees) - benchmark) / sqrt(size(X, 1) - 1)
end
function (r::TrackingRiskMeasure{<:Any, <:Any, <:NOCTracking})(w::AbstractVector,
                                                               X::AbstractMatrix,
                                                               fees::Union{Nothing, <:Fees} = nothing)
    benchmark = tracking_benchmark(r.tracking, X)
    return norm(calc_net_returns(w, X, fees) - benchmark, 1) / size(X, 1)
end
function risk_measure_view(r::TrackingRiskMeasure, i::AbstractVector, args...)
    tracking = tracking_view(r.tracking, i)
    return TrackingRiskMeasure(; settings = r.settings, tracking = tracking,
                               formulation = r.formulation)
end
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

export TrackingRiskMeasure
