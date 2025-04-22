struct SimpleVariance{T1 <: Union{Nothing, <:AbstractExpectedReturnsEstimator}, T2 <: Bool,
                      T3 <: Union{Nothing, <:AbstractWeights}} <: AbstractVarianceEstimator
    me::T1
    corrected::T2
    w::T3
end
function SimpleVariance(;
                        me::Union{Nothing, <:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
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
function StatsBase.std(ve::SimpleVariance, X::AbstractVector; mean = nothing)
    return if isnothing(ve.w)
        std(X; corrected = ve.corrected, mean = mean)
    else
        std(X, ve.w; corrected = ve.corrected, mean = mean)
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
function StatsBase.var(ve::SimpleVariance, X::AbstractVector; mean = nothing)
    return if isnothing(ve.w)
        var(X; corrected = ve.corrected, mean = mean)
    else
        var(X, ve.w; corrected = ve.corrected, mean = mean)
    end
end
function factory(ve::SimpleVariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleVariance(; me = factory(ve.me, w), corrected = ve.corrected,
                          w = isnothing(w) ? ve.w : w)
end

export SimpleVariance
