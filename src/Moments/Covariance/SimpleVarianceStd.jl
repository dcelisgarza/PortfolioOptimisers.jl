struct SimpleVariance{T1 <: Bool} <: POVarianceEstimator
    corrected::T1
end
function SimpleVariance(; corrected::Bool = true)
    return SimpleVariance{typeof(corrected)}(corrected)
end
function StatsBase.std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return std(X; dims = dims, corrected = ve.corrected, mean = mean)
end
function StatsBase.std(ve::SimpleVariance, X::AbstractMatrix, w::AbstractWeights;
                       dims::Int = 1, mean = nothing)
    return std(X, w, dims; corrected = ve.corrected, mean = mean)
end
function StatsBase.var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return var(X; dims = dims, corrected = ve.corrected, mean = mean)
end
function StatsBase.var(ve::SimpleVariance, X::AbstractMatrix, w::AbstractWeights;
                       dims::Int = 1, mean = nothing)
    return var(X, w, dims; corrected = ve.corrected, mean = mean)
end

export SimpleVariance
