struct MeanReturn{T1 <: Union{Nothing, <:AbstractWeights}} <: NoOptimisationRiskMeasure
    w::T1
end
function MeanReturn(; w::Union{Nothing, <:AbstractWeights} = nothing)
    return MeanReturn{typeof(w)}(w)
end
function (r::MeanReturn)(x::AbstractVector)
    return isnothing(r.w) ? mean(x) : mean(x, r.w)
end
function risk_measure_factory(r::MeanReturn, prior::EntropyPoolingPriorResult, args...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.w)
    return MeanReturn(; w = w)
end
function risk_measure_factory(r::MeanReturn,
                              prior::HighOrderPriorResult{<:EntropyPoolingPriorResult,
                                                          <:Any, <:Any, <:Any, <:Any},
                              args...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.pr.w)
    return MeanReturn(; w = w)
end
function risk_measure_view(r::MeanReturn, args...; kwargs...)
    return risk_measure_factory(r, args...; kwargs...)
end
struct ThirdCentralMoment{T1 <: Union{Nothing, <:AbstractWeights},
                          T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
       AbstractMomentNoOptimisationRiskMeasure
    w::T1
    mu::T2
end
function ThirdCentralMoment(; w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return ThirdCentralMoment{typeof(w), typeof(mu)}(w, mu)
end
struct Skewness{T1 <: AbstractVarianceEstimator, T2 <: Union{Nothing, <:AbstractWeights},
                T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
       AbstractMomentNoOptimisationRiskMeasure
    ve::T1
    w::T2
    mu::T3
end
function Skewness(; ve::AbstractVarianceEstimator = SimpleVariance(),
                  w::Union{Nothing, <:AbstractWeights} = nothing,
                  mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return Skewness{typeof(ve), typeof(w), typeof(mu)}(ve, w, mu)
end
function calc_moment_target(::Union{<:ThirdCentralMoment{Nothing, Nothing},
                                    <:Skewness{<:Any, Nothing, Nothing}}, ::Any,
                            x::AbstractVector)
    return mean(x)
end
function calc_moment_target(r::Union{<:ThirdCentralMoment{<:AbstractWeights, Nothing},
                                     <:Skewness{<:Any, <:AbstractWeights, Nothing}}, ::Any,
                            x::AbstractVector)
    return mean(x, r.w)
end
function calc_moment_target(r::Union{<:ThirdCentralMoment{<:Any, <:AbstractVector},
                                     <:Skewness{<:Any, <:Any, <:AbstractVector}},
                            w::AbstractVector, ::Any)
    return dot(w, r.mu)
end
function calc_moment_target(r::Union{<:ThirdCentralMoment{<:Any, <:Real},
                                     <:Skewness{<:Any, <:Any, <:Real}}, ::Any, ::Any)
    return r.mu
end
function risk_measure_factory(r::ThirdCentralMoment, prior::AbstractPriorResult, args...;
                              kwargs...)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    return ThirdCentralMoment(; w = r.w, mu = mu)
end
function risk_measure_factory(r::ThirdCentralMoment, prior::EntropyPoolingPriorResult,
                              args...; kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.w)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    return ThirdCentralMoment(; w = w, mu = mu)
end
function risk_measure_factory(r::ThirdCentralMoment,
                              prior::HighOrderPriorResult{<:EntropyPoolingPriorResult,
                                                          <:Any, <:Any, <:Any, <:Any},
                              args...; kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.pr.w)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    return ThirdCentralMoment(; w = w, mu = mu)
end
function risk_measure_view(r::ThirdCentralMoment, i::AbstractVector, args...; kwargs...)
    mu = nothing_scalar_array_view(r.mu, i)
    return ThirdCentralMoment(; w = r.w, mu = mu)
end
function risk_measure_view(r::ThirdCentralMoment, i::AbstractVector,
                           prior::AbstractPriorResult, args...; kwargs...)
    mu = risk_measure_nothing_scalar_array_view(r.mu, prior.mu, i)
    return ThirdCentralMoment(; w = r.w, mu = mu)
end
function risk_measure_view(r::ThirdCentralMoment, i::AbstractVector,
                           prior::EntropyPoolingPriorResult, args...; kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.w)
    mu = risk_measure_nothing_scalar_array_view(r.mu, prior.mu, i)
    return ThirdCentralMoment(; w = w, mu = mu)
end
function risk_measure_view(r::ThirdCentralMoment, i::AbstractVector,
                           prior::HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any,
                                                       <:Any, <:Any, <:Any}, args...;
                           kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.pr.w)
    mu = risk_measure_nothing_scalar_array_view(r.mu, prior.mu, i)
    return ThirdCentralMoment(; w = w, mu = mu)
end
function (r::ThirdCentralMoment)(w::AbstractVector, X::AbstractMatrix,
                                 fees::Union{Nothing, <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return sum(val .^ 3) / size(X, 1)
end
function risk_measure_factory(r::Skewness, prior::AbstractPriorResult, args...; kwargs...)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    return Skewness(; ve = r.ve, w = r.w, mu = mu)
end
function risk_measure_factory(r::Skewness, prior::EntropyPoolingPriorResult, args...;
                              kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.w)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    return Skewness(; ve = factory(r.ve, w), w = w, mu = mu)
end
function risk_measure_factory(r::Skewness,
                              prior::HighOrderPriorResult{<:EntropyPoolingPriorResult,
                                                          <:Any, <:Any, <:Any, <:Any},
                              args...; kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.pr.w)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    return Skewness(; ve = factory(r.ve, w), w = w, mu = mu)
end
function risk_measure_view(r::Skewness, i::AbstractVector, args...; kwargs...)
    mu = nothing_scalar_array_view(r.mu, i)
    return Skewness(; ve = r.ve, w = r.w, mu = mu)
end
function risk_measure_view(r::Skewness, i::AbstractVector, prior::AbstractPriorResult,
                           args...; kwargs...)
    mu = risk_measure_nothing_scalar_array_view(r.mu, prior.mu, i)
    return Skewness(; ve = r.ve, w = r.w, mu = mu)
end
function risk_measure_view(r::Skewness, i::AbstractVector, prior::EntropyPoolingPriorResult,
                           args...; kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.w)
    mu = risk_measure_nothing_scalar_array_view(r.mu, prior.mu, i)
    return Skewness(; ve = factory(r.ve, w), w = w, mu = mu)
end
function risk_measure_view(r::Skewness, i::AbstractVector,
                           prior::HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any,
                                                       <:Any, <:Any, <:Any}, args...;
                           kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.pr.w)
    mu = risk_measure_nothing_scalar_array_view(r.mu, prior.mu, i)
    return Skewness(; ve = factory(r.ve, w), w = w, mu = mu)
end
function (r::Skewness)(w::AbstractVector, X::AbstractMatrix,
                       fees::Union{Nothing, <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    sigma = StatsBase.std(r.ve, val; mean = zero(eltype(val)))
    return sum(val .^ 3) / size(X, 1) / sigma^3
end

export MeanReturn, ThirdCentralMoment, Skewness
