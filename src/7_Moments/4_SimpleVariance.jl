struct SimpleVariance{T1 <: Union{Nothing, <:AbstractExpectedReturnsEstimator},
                      T2 <: Union{Nothing, <:AbstractWeights}, T3 <: Bool} <:
       AbstractVarianceEstimator
    me::T1
    w::T2
    corrected::T3
end
function SimpleVariance(;
                        me::Union{Nothing, <:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
                        w::Union{Nothing, <:AbstractWeights} = nothing,
                        corrected::Bool = true)
    return SimpleVariance{typeof(me), typeof(w), typeof(corrected)}(me, w, corrected)
end
function StatsBase.std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing,
                       kwargs...)
    mu = isnothing(mean) ? StatsBase.mean(ve.me, X; dims = dims, kwargs...) : mean
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
function StatsBase.var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing,
                       kwargs...)
    mu = isnothing(mean) ? StatsBase.mean(ve.me, X; dims = dims, kwargs...) : mean
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
    return SimpleVariance(; me = factory(ve.me, w), w = isnothing(w) ? ve.w : w,
                          corrected = ve.corrected)
end

export SimpleVariance
