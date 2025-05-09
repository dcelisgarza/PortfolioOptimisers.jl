abstract type AbstractTracking <: AbstractEstimator end
function tracking_view(::Nothing, ::Any)
    return nothing
end
struct WeightsTracking{T1 <: AbstractVector{<:Real}, T2 <: Union{Nothing, <:Fees}} <:
       AbstractTracking
    w::T1
    fees::T2
end
function WeightsTracking(; w::AbstractVector{<:Real},
                         fees::Union{Nothing, <:Fees} = nothing)
    @smart_assert(!isempty(w))
    return WeightsTracking{typeof(w), typeof(fees)}(w, fees)
end
function tracking_view(tracking::WeightsTracking, i::AbstractVector)
    w = view(tracking.w, i)
    fees = fees_view(tracking.fees, i)
    return WeightsTracking(; w = w, fees = fees)
end
function tracking_benchmark(tracking::WeightsTracking, X::AbstractMatrix{<:Real})
    return calc_net_returns(tracking.w, X, tracking.fees)
end
struct ReturnsTracking{T1 <: AbstractVector{<:Real}} <: AbstractTracking
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
struct TrackingError{T1 <: Real, T2 <: AbstractTracking}
    err::T1
    tracking::T2
end
function TrackingError(; err::Real = 0.0, tracking::AbstractTracking)
    @smart_assert(isfinite(err) && err >= zero(err))
    return TrackingError{typeof(err), typeof(tracking)}(err, tracking)
end
function tracking_view(tracking::TrackingError, i::AbstractVector)
    tracking = tracking_view(tracking.tracking, i)
    return TrackingError(; err = tracking.err, tracking = tracking)
end

export WeightsTracking, ReturnsTracking, TrackingError
