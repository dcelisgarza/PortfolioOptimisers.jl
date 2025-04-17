struct MeanReturn{T1 <: Union{Nothing, <:AbstractWeights}} <: NoOptimisationRiskMeasure
    w::T1
end
function MeanReturn(; w::AbstractWeights = nothing)
    return MeanReturn{typeof(w)}(w)
end
function (r::MeanReturn)(x::AbstractVector)
    return isnothing(r.w) ? mean(x) : mean(x, r.w)
end
function risk_measure_factory(r::MeanReturn, args...)
    return r(; w = r.w)
end
function risk_measure_factory(r::MeanReturn, prior::EntropyPoolingPriorResult, args...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.w)
    return r(; w = w)
end
struct Skewness{T1 <: AbstractVarianceEstimator, T2 <: Union{Nothing, <:AbstractWeights},
                T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
       AbstractMomentNoOptimisationRiskMeasure
    ve::T1
    w::T2
    mu::T3
end
function Skewness(; ve::AbstractVarianceEstimator = SimpleVariance(),
                  target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                  w::Union{Nothing, <:AbstractWeights} = nothing,
                  mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return Skewness{typeof(ve), typeof(w), typeof(mu)}(ve, w, mu)
end
function (r::Skewness)(w::AbstractVector, X::AbstractMatrix,
                       fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(x, w, r)
    val = x .- target
    sigma = std(r.ve, x)
    return sum(val .^ 3) / length(x) / sigma^3
end
struct ThirdCentralMoment{T1 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                          T2 <: Union{Nothing, <:AbstractWeights},
                          T3 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       AbstractMomentNoOptimisationRiskMeasure
    target::T1
    w::T2
    mu::T3
end
function ThirdCentralMoment(;
                            target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                            w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return ThirdCentralMoment{typeof(target), typeof(w), typeof(mu)}(target, w, mu)
end
function (r::ThirdCentralMoment)(w::AbstractVector, X::AbstractMatrix,
                                 fees::Union{Nothing, <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return sum(val .^ 3) / size(X, 1)
end

export MeanReturn, ThirdCentralMoment, Skewness
