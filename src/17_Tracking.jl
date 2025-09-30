abstract type AbstractTracking <: AbstractResult end
abstract type AbstractTrackingAlgorithm <: AbstractAlgorithm end
abstract type TrackingFormulation <: AbstractAlgorithm end
abstract type NormTracking <: TrackingFormulation end
abstract type VariableTracking <: TrackingFormulation end
struct SOCTracking{T1} <: NormTracking
    ddof::T1
    function SOCTracking(ddof::Integer)
        @argcheck(ddof > 0, DomainError("`ddof` must be greater than 0:\nddof => $ddof"))
        return new{typeof(ddof)}(ddof)
    end
end
function SOCTracking(; ddof::Integer = 1)
    return SOCTracking(ddof)
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
struct WeightsTracking{T1, T2} <: AbstractTrackingAlgorithm
    fees::T1
    w::T2
    function WeightsTracking(fees::Union{Nothing, <:Fees}, w::AbstractVector{<:Real})
        @argcheck(!isempty(w), IsEmptyError(non_empty_msg("`w`") * "."))
        return new{typeof(fees), typeof(w)}(fees, w)
    end
end
function WeightsTracking(; fees::Union{Nothing, <:Fees} = nothing,
                         w::AbstractVector{<:Real})
    return WeightsTracking(fees, w)
end
function factory(tracking::WeightsTracking, w::AbstractVector)
    return WeightsTracking(; fees = factory(tracking.fees, tracking.w), w = w)
end
function tracking_view(tracking::WeightsTracking, i::AbstractVector)
    fees = fees_view(tracking.fees, i)
    w = view(tracking.w, i)
    return WeightsTracking(; fees = fees, w = w)
end
function tracking_benchmark(tracking::WeightsTracking, X::AbstractMatrix{<:Real})
    return calc_net_returns(tracking.w, X, tracking.fees)
end
struct ReturnsTracking{T1} <: AbstractTrackingAlgorithm
    w::T1
    function ReturnsTracking(w::AbstractVector{<:Real})
        @argcheck(!isempty(w), IsEmptyError(non_empty_msg("`w`") * "."))
        return new{typeof(w)}(w)
    end
end
function ReturnsTracking(; w::AbstractVector{<:Real})
    return ReturnsTracking(w)
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
struct TrackingError{T1, T2, T3} <: AbstractTracking
    tracking::T1
    err::T2
    alg::T3
    function TrackingError(tracking::AbstractTrackingAlgorithm, err::Real,
                           alg::NormTracking)
        @argcheck(isfinite(err) && err >= zero(err),
                  DomainError("`err` must be finite and non-negative:\nerr => $err"))
        return new{typeof(tracking), typeof(err), typeof(alg)}(tracking, err, alg)
    end
end
function TrackingError(; tracking::AbstractTrackingAlgorithm, err::Real = 0.0,
                       alg::NormTracking = SOCTracking())
    return TrackingError(tracking, err, alg)
end
function tracking_view(tracking::TrackingError, i::AbstractVector, args...)
    return TrackingError(; tracking = tracking_view(tracking.tracking, i),
                         err = tracking.err, alg = tracking.alg)
end
function tracking_view(tracking::AbstractVector{<:AbstractTracking}, args...)
    return [tracking_view(t, args...) for t in tracking]
end
function factory(tracking::TrackingError, w::AbstractVector)
    return TrackingError(; tracking = factory(tracking.tracking, w), err = tracking.err,
                         alg = tracking.alg)
end

export WeightsTracking, ReturnsTracking, TrackingError
