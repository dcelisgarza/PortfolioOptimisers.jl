struct SimpleVariance{T1 <: AbstractExpectedReturnsEstimator, T2 <: Bool,
                      T3 <: Union{Nothing, <:AbstractWeights}} <: AbstractVarianceEstimator
    me::T1
    corrected::T2
    w::T3
end
function SimpleVariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        corrected::Bool = true,
                        w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleVariance{typeof(me), typeof(corrected), typeof(w)}(me, corrected, w)
end
function StatsBase.std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    mu = isnothing(mean) ? StatsBase.mean(ve.me, X; dims = dims) : mean
    return if isnothing(ve.w)
        std(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        std(X, ve.w, dims; corrected = ve.corrected, mean = mu)
    end
end
function StatsBase.var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    mu = isnothing(mean) ? StatsBase.mean(ve.me, X; dims = dims) : mean
    return if isnothing(ve.w)
        var(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        var(X, ve.w, dims; corrected = ve.corrected, mean = mu)
    end
end
function w_moment_factory(ve::SimpleVariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleVariance(; me = ve.me, corrected = ve.corrected, w = w)
end

export SimpleVariance
