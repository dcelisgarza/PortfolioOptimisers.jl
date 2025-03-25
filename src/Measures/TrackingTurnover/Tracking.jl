abstract type AbstractTracking end
struct WeightsTracking{T1 <: Fees, T2 <: AbstractVector{<:Real}} <: AbstractTracking
    fees::T1
    w::T2
end
function WeightsTracking(; fees::Fees = Fees(), w::AbstractVector{<:Real})
    @smart_assert(!isempty(w))
    return WeightsTracking{typeof(fees), typeof(w)}(fees, w)
end
function tracking_benchmark(X::AbstractMatrix{<:Real}, tracking::WeightsTracking)
    return calc_net_asset_returns(X, tracking.w, tracking.fees)
end
struct ReturnsTracking{T1 <: AbstractVector{<:Real}} <: AbstractTracking
    w::T1
end
function ReturnsTracking(; w::AbstractVector{<:Real})
    @smart_assert(!isempty(w))
    return ReturnsTracking{typeof(w)}(w)
end
function tracking_benchmark(::Any, tracking::ReturnsTracking)
    return tracking.w
end
abstract type AbstractTrackingError end
struct NoTrackingError <: AbstractTrackingError end
struct TrackingError{T1 <: Real, T2 <: AbstractTracking} <: AbstractTrackingError
    err::T1
    tracking::T2
end
function TrackingError(; err::Real = 0.0, tracking::AbstractTracking)
    @smart_assert(isfinite(err) && err >= zero(err))
    return TrackingError{typeof(err), typeof(tracking)}(err, tracking)
end

export WeightsTracking, ReturnsTracking, TrackingError
