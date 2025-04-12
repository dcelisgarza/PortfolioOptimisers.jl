abstract type AbstractTracking <: AbstractEstimator end
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
struct TrackingError{T1 <: Real, T2 <: AbstractTracking}
    err::T1
    tracking::T2
end
function TrackingError(; err::Real = 0.0, tracking::AbstractTracking)
    @smart_assert(isfinite(err) && err >= zero(err))
    return TrackingError{typeof(err), typeof(tracking)}(err, tracking)
end

export WeightsTracking, ReturnsTracking, TrackingError
