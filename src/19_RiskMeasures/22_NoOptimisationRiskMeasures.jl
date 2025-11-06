struct MeanReturn{T1} <: NoOptimisationRiskMeasure
    w::T1
    function MeanReturn(w::Union{Nothing, <:AbstractWeights})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return MeanReturn(w)
    end
end
function MeanReturn(; w::Union{Nothing, <:AbstractWeights} = nothing)
    return MeanReturn(w)
end
function (r::MeanReturn)(x::AbstractVector)
    return isnothing(r.w) ? mean(x) : mean(x, r.w)
end
function factory(r::MeanReturn, prior::AbstractPriorResult, args...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    return MeanReturn(; w = w)
end
function risk_measure_view(r::MeanReturn, ::Any, args...)
    return r
end
struct ThirdCentralMoment{T1, T2} <: NoOptimisationRiskMeasure
    w::T1
    mu::T2
    function ThirdCentralMoment(w::Union{Nothing, <:AbstractWeights},
                                mu::Union{Nothing, <:Real, <:AbstractVector{<:Real},
                                          <:VecScalar})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        if isa(mu, AbstractVector)
            @argcheck(!isempty(mu))
        end
        return new{typeof(w), typeof(mu)}(w, mu)
    end
end
function ThirdCentralMoment(; w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:Real, <:AbstractVector{<:Real},
                                      <:VecScalar} = nothing)
    return ThirdCentralMoment(w, mu)
end
struct Skewness{T1, T2, T3} <: NoOptimisationRiskMeasure
    ve::T1
    w::T2
    mu::T3
    function Skewness(ve::AbstractVarianceEstimator, w::Union{Nothing, <:AbstractWeights},
                      mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:VecScalar})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        if isa(mu, AbstractVector)
            @argcheck(!isempty(mu))
        end
        return new{typeof(ve), typeof(w), typeof(mu)}(ve, w, mu)
    end
end
function Skewness(; ve::AbstractVarianceEstimator = SimpleVariance(),
                  w::Union{Nothing, <:AbstractWeights} = nothing,
                  mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:VecScalar} = nothing)
    return Skewness(ve, w, mu)
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
function calc_moment_target(r::Union{<:ThirdCentralMoment{<:Any, <:VecScalar},
                                     <:Skewness{<:Any, <:Any, <:VecScalar}},
                            w::AbstractVector, ::Any)
    return dot(w, r.mu.v) + r.mu.s
end
function calc_moment_target(r::Union{<:ThirdCentralMoment{<:Any, <:Real},
                                     <:Skewness{<:Any, <:Any, <:Real}}, ::Any, ::Any)
    return r.mu
end
function calc_deviations_vec(r::Union{<:ThirdCentralMoment, <:Skewness}, w::AbstractVector,
                             X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function factory(r::ThirdCentralMoment, prior::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    mu = nothing_scalar_array_factory(r.mu, prior.mu)
    return ThirdCentralMoment(; w = w, mu = mu)
end
function risk_measure_view(r::ThirdCentralMoment, i::AbstractVector, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return ThirdCentralMoment(; w = r.w, mu = mu)
end
function (r::ThirdCentralMoment)(w::AbstractVector, X::AbstractMatrix,
                                 fees::Union{Nothing, <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    val .= val .^ 3
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function factory(r::Skewness, prior::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    mu = nothing_scalar_array_factory(r.mu, prior.mu)
    return Skewness(; ve = factory(r.ve, w), w = w, mu = mu)
end
function risk_measure_view(r::Skewness, i::AbstractVector, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return Skewness(; ve = r.ve, w = r.w, mu = mu)
end
function (r::Skewness)(w::AbstractVector, X::AbstractMatrix,
                       fees::Union{Nothing, <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    sigma = Statistics.std(r.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^3
end

export MeanReturn, ThirdCentralMoment, Skewness
