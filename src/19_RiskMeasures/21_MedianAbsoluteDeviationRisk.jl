abstract type MedianCenteringFunction <: AbstractAlgorithm end
struct MedianCentering <: MedianCenteringFunction end
struct MeanCentering <: MedianCenteringFunction end
struct MedianAbsoluteDeviation{T1, T2, T3, T4} <: HierarchicalRiskMeasure
    settings::T1
    w::T2
    mu::T3
    flag::T4
    function MedianAbsoluteDeviation(settings::HierarchicalRiskMeasureSettings,
                                     w::Union{Nothing, <:AbstractWeights},
                                     mu::Union{<:Real, <:AbstractVector{<:Real},
                                               <:VecScalar, <:MedianCenteringFunction},
                                     flag::Bool = true)
        if isa(mu, AbstractVector)
            @argcheck(!isempty(mu) && all(isfinite, mu))
        elseif isa(mu, Real)
            @argcheck(isfinite(mu))
        end
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(flag)}(settings, w, mu,
                                                                          flag)
    end
end
function MedianAbsoluteDeviation(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                 w::Union{Nothing, <:AbstractWeights} = nothing,
                                 mu::Union{<:Real, <:AbstractVector{<:Real}, <:VecScalar,
                                           <:MedianCenteringFunction} = MedianCentering(),
                                 flag::Bool = true)
    return MedianAbsoluteDeviation(settings, w, mu, flag)
end
function factory(r::MedianAbsoluteDeviation, prior::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    return MedianAbsoluteDeviation(; settings = r.settings, w = w, mu = r.mu, flag = r.flag)
end
function nothing_scalar_array_view(x::MedianCenteringFunction, ::Any)
    return x
end
function risk_measure_view(r::MedianAbsoluteDeviation, i::AbstractVector, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return MedianAbsoluteDeviation(; settings = r.settings, w = r.w, mu = mu, flag = r.flag)
end
function calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MeanCentering,
                                                      <:Any}, ::Any, x::AbstractVector)
    return mean(x)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:AbstractWeights,
                                                       <:MeanCentering, <:Any}, ::Any,
                            x::AbstractVector)
    return mean(x, r.w)
end
function calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MedianCentering,
                                                      <:Any}, ::Any, x::AbstractVector)
    return median(x)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:AbstractWeights,
                                                       <:MedianCentering, <:Any}, ::Any,
                            x::AbstractVector)
    return median(x, r.w)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:AbstractVector,
                                                       <:Any}, w::AbstractVector, ::Any)
    return dot(w, r.mu)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:Real, <:Any}, ::Any,
                            ::Any)
    return r.mu
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:VecScalar, <:Any},
                            w::AbstractVector, ::Any)
    return dot(w, r.mu.v) + r.mu.s
end
function calc_deviations_vec(r::MedianAbsoluteDeviation, w::AbstractVector,
                             X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function (r::MedianAbsoluteDeviation)(w::AbstractVector, X::AbstractMatrix,
                                      fees::Union{Nothing, <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return mad(val; center = zero(eltype(X)), normalize = r.flag)
end

export MedianAbsoluteDeviation, MedianCentering, MeanCentering
