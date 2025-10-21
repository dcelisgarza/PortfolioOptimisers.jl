abstract type MedianCenteringFunction <: AbstractAlgorithm end
struct MedianCentering <: MedianCenteringFunction end
struct MeanCentering <: MedianCenteringFunction end
struct MedianAbsoluteDeviation{T1, T2, T3, T4} <: HierarchicalRiskMeasure
    settings::T1
    w::T2
    center::T3
    flag::T4
    function MedianAbsoluteDeviation(settings::HierarchicalRiskMeasureSettings,
                                     w::Union{Nothing, <:AbstractWeights},
                                     center::Union{<:Real, <:AbstractVector{<:Real},
                                                   <:MedianCenteringFunction},
                                     flag::Bool = true)
        if isa(center, AbstractVector)
            @argcheck(!isempty(center) && all(isfinite, center))
        elseif isa(center, Real)
            @argcheck(isfinite(center))
        end
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(center), typeof(flag)}(settings, w,
                                                                              center, flag)
    end
end
function MedianAbsoluteDeviation(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                 w::Union{Nothing, <:AbstractWeights} = nothing,
                                 center::Union{<:Real, <:AbstractVector{<:Real},
                                               <:MedianCenteringFunction} = MedianCentering(),
                                 flag::Bool = true)
    return MedianAbsoluteDeviation(settings, w, center, flag)
end
function factory(r::MedianAbsoluteDeviation, prior::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    return MedianAbsoluteDeviation(; settings = r.settings, w = w, center = r.center,
                                   flag = r.flag)
end
function nothing_scalar_array_view(x::MedianCenteringFunction, ::Any)
    return x
end
function risk_measure_view(r::MedianAbsoluteDeviation, i::AbstractVector, args...)
    center = nothing_scalar_array_view(r.center, i)
    return MedianAbsoluteDeviation(; settings = r.settings, w = r.w, center = center,
                                   flag = r.flag)
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
    return dot(w, r.center)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:Real, <:Any}, ::Any,
                            ::Any)
    return r.center
end
function calc_moment_val(r::MedianAbsoluteDeviation, w::AbstractVector, X::AbstractMatrix,
                         fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function (r::MedianAbsoluteDeviation)(w::AbstractVector, X::AbstractMatrix,
                                      fees::Union{Nothing, <:Fees} = nothing)
    return mad(calc_moment_val(r, w, X, fees); center = zero(eltype(X)), normalize = r.flag)
end

export MedianAbsoluteDeviation, MedianCentering, MeanCentering
