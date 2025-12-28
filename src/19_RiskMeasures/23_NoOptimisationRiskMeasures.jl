struct MeanReturn{T1} <: NoOptimisationRiskMeasure
    w::T1
    function MeanReturn(w::Option{<:AbstractWeights})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return MeanReturn(w)
    end
end
function MeanReturn(; w::Option{<:AbstractWeights} = nothing)
    return MeanReturn(w)
end
function (r::MeanReturn)(x::VecNum)
    return isnothing(r.w) ? mean(x) : mean(x, r.w)
end
function factory(r::MeanReturn, pr::AbstractPriorResult, args...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    return MeanReturn(; w = w)
end
function risk_measure_view(r::MeanReturn, ::Any, args...)
    return r
end
struct ThirdCentralMoment{T1, T2} <: NoOptimisationRiskMeasure
    w::T1
    mu::T2
    function ThirdCentralMoment(w::Option{<:AbstractWeights},
                                mu::Option{<:Num_VecNum_VecScalar})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        if isa(mu, VecNum)
            @argcheck(!isempty(mu))
        end
        return new{typeof(w), typeof(mu)}(w, mu)
    end
end
function ThirdCentralMoment(; w::Option{<:AbstractWeights} = nothing,
                            mu::Option{<:Num_VecNum_VecScalar} = nothing)
    return ThirdCentralMoment(w, mu)
end
struct Skewness{T1, T2, T3} <: NoOptimisationRiskMeasure
    ve::T1
    w::T2
    mu::T3
    function Skewness(ve::AbstractVarianceEstimator, w::Option{<:AbstractWeights},
                      mu::Option{<:Num_VecNum_VecScalar})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        if isa(mu, VecNum)
            @argcheck(!isempty(mu))
        end
        return new{typeof(ve), typeof(w), typeof(mu)}(ve, w, mu)
    end
end
function Skewness(; ve::AbstractVarianceEstimator = SimpleVariance(),
                  w::Option{<:AbstractWeights} = nothing,
                  mu::Option{<:Num_VecNum_VecScalar} = nothing)
    return Skewness(ve, w, mu)
end
const TCM_Sk{T1, T2} = Union{<:ThirdCentralMoment{T1, T2}, <:Skewness{<:Any, T1, T2}}
function calc_moment_target(::TCM_Sk{Nothing, Nothing}, ::Any, x::VecNum)
    return mean(x)
end
function calc_moment_target(r::TCM_Sk{<:AbstractWeights, Nothing}, ::Any, x::VecNum)
    return mean(x, r.w)
end
function calc_moment_target(r::TCM_Sk{<:Any, <:VecNum}, w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu)
end
function calc_moment_target(r::TCM_Sk{<:Any, <:VecScalar}, w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu.v) + r.mu.s
end
function calc_moment_target(r::TCM_Sk{<:Any, <:Number}, ::Any, ::Any)
    return r.mu
end
function calc_deviations_vec(r::TCM_Sk, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x)
    return x .- tgt
end
function factory(r::ThirdCentralMoment, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    return ThirdCentralMoment(; w = w, mu = mu)
end
function risk_measure_view(r::ThirdCentralMoment, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return ThirdCentralMoment(; w = r.w, mu = mu)
end
function (r::ThirdCentralMoment)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    val .= val .^ 3
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function factory(r::Skewness, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    return Skewness(; ve = factory(r.ve, w), w = w, mu = mu)
end
function risk_measure_view(r::Skewness, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return Skewness(; ve = r.ve, w = r.w, mu = mu)
end
function (r::Skewness)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    sigma = Statistics.std(r.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^3
end

export MeanReturn, ThirdCentralMoment, Skewness
