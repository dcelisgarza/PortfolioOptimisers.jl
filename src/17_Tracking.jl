abstract type AbstractTracking <: AbstractEstimator end
abstract type AbstractTrackingAlgorithm <: AbstractAlgorithm end
abstract type TrackingFormulation <: AbstractAlgorithm end
abstract type NormTracking <: TrackingFormulation end
abstract type VariableTracking <: TrackingFormulation end
struct SOCTracking{T1 <: Integer} <: NormTracking
    ddof::T1
end
function SOCTracking(; ddof::Integer = 1)
    @smart_assert(ddof > 0)
    return SOCTracking{typeof(ddof)}(ddof)
end
struct NOCTracking <: NormTracking end
function norm_tracking(f::SOCTracking, a, b, N = nothing)
    factor = isnothing(N) ? 1 : sqrt(N - f.ddof)
    return norm(a - b, 2) / factor
end
function norm_tracking(::NOCTracking, a, b, N = nothing)
    factor = isnothing(N) ? 1 : N
    return norm(a - b, 1) / factor
end
struct IndependentVariableTracking <: VariableTracking end
struct DependentVariableTracking <: VariableTracking end
function tracking_view(::Nothing, ::Any)
    return nothing
end
struct WeightsTracking{T1 <: Union{Nothing, <:Fees}, T2 <: AbstractVector{<:Real}} <:
       AbstractTrackingAlgorithm
    fees::T1
    w::T2
end
function factory(tracking::WeightsTracking, w::AbstractVector)
    return WeightsTracking(; fees = factory(tracking.fees, tracking.w), w = w)
end
function WeightsTracking(; fees::Union{Nothing, <:Fees} = nothing,
                         w::AbstractVector{<:Real})
    @smart_assert(!isempty(w))
    return WeightsTracking{typeof(fees), typeof(w)}(fees, w)
end
function tracking_view(tracking::WeightsTracking, i::AbstractVector)
    fees = fees_view(tracking.fees, i)
    w = view(tracking.w, i)
    return WeightsTracking(; w = w, fees = fees)
end
function tracking_benchmark(tracking::WeightsTracking, X::AbstractMatrix{<:Real})
    return calc_net_returns(tracking.w, X, tracking.fees)
end
struct ReturnsTracking{T1 <: AbstractVector{<:Real}} <: AbstractTrackingAlgorithm
    w::T1
end
function ReturnsTracking(; w::AbstractVector{<:Real})
    @smart_assert(!isempty(w))
    return ReturnsTracking{typeof(w)}(w)
end
function tracking_view(tracking::ReturnsTracking, ::Any)
    return tracking
end
function tracking_benchmark(tracking::ReturnsTracking, ::Any)
    return tracking.w
end
function factory(tracking::ReturnsTracking, ::Any)
    return tracking
end
struct TrackingError{T1 <: AbstractTrackingAlgorithm, T2 <: Real, T3 <: NormTracking} <:
       AbstractTracking
    tracking::T1
    err::T2
    formulation::T3
end
function TrackingError(; tracking::AbstractTrackingAlgorithm, err::Real = 0.0,
                       formulation::NormTracking = SOCTracking())
    @smart_assert(isfinite(err) && err >= zero(err))
    return TrackingError{typeof(tracking), typeof(err), typeof(formulation)}(tracking, err,
                                                                             formulation)
end
function tracking_view(tracking::TrackingError, i::AbstractVector, args...)
    return TrackingError(; tracking = tracking_view(tracking.tracking, i),
                         err = tracking.err, formulation = tracking.formulation)
end
function tracking_view(tracking::AbstractVector{<:AbstractTracking}, args...)
    return [tracking_view(t, args...) for t ∈ tracking]
end
function factory(tracking::TrackingError, w::AbstractVector)
    return TrackingError(; tracking = factory(tracking.tracking, w), err = tracking.err,
                         formulation = tracking.formulation)
end

export WeightsTracking, ReturnsTracking, TrackingError
