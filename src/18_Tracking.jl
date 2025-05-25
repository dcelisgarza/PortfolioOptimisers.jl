abstract type AbstractTracking <: AbstractEstimator end
abstract type AbstractTrackingAlgorithm <: AbstractAlgorithm end
function tracking_view(::Nothing, ::Any)
    return nothing
end
struct WeightsTracking{T1 <: Union{Nothing, <:Fees}, T2 <: AbstractVector{<:Real}} <:
       AbstractTrackingAlgorithm
    fees::T1
    w::T2
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
struct TrackingError{T1 <: AbstractTrackingAlgorithm, T2 <: Real} <: AbstractTracking
    tracking::T1
    err::T2
end
function TrackingError(; tracking::AbstractTrackingAlgorithm, err::Real = 0.0)
    @smart_assert(isfinite(err) && err >= zero(err))
    return TrackingError{typeof(tracking), typeof(err)}(tracking, err)
end
function tracking_view(tracking::TrackingError, i::AbstractVector)
    tracking = tracking_view(tracking.tracking, i)
    return TrackingError(; tracking = tracking, err = tracking.err)
end
struct VolTrackingError{T1 <: WeightsTracking, T2 <: Real,
                        T3 <: Union{Nothing, <:AbstractMatrix}} <: AbstractTracking
    tracking::T1
    err::T2
    sigma::T3
end
function VolTrackingError(; tracking::WeightsTracking, err::Real = 0.0,
                          sigma::Union{Nothing, <:AbstractMatrix} = nothing)
    @smart_assert(isfinite(err) && err >= zero(err))
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
    end
    return VolTrackingError{typeof(tracking), typeof(err), typeof(sigma)}(tracking, err,
                                                                          sigma)
end
function tracking_view(tracking::VolTrackingError, i::AbstractVector)
    tracking = tracking_view(tracking.tracking, i)
    sigma = nothing_scalar_array_view(tracking.sigma, i)
    return VolTrackingError(; tracking = tracking, err = tracking.err, sigma = sigma)
end

export WeightsTracking, ReturnsTracking, TrackingError, VolTrackingError
