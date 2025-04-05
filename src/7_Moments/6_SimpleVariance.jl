struct SimpleVariance{T1 <: Bool, T2 <: Union{Nothing, <:AbstractWeights}} <:
       AbstractVarianceEstimator
    corrected::T1
    w::T2
end
function SimpleVariance(; corrected::Bool = true,
                        w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleVariance{typeof(corrected), typeof(w)}(corrected, w)
end
function StatsBase.std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return if isnothing(ve.w)
        std(X; dims = dims, corrected = ve.corrected, mean = mean)
    else
        std(X, ve.w, dims; corrected = ve.corrected, mean = mean)
    end
end
function StatsBase.var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return if isnothing(ve.w)
        var(X; dims = dims, corrected = ve.corrected, mean = mean)
    else
        var(X, ve.w, dims; corrected = ve.corrected, mean = mean)
    end
end
function w_moment_factory(ve::SimpleVariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleVariance(; corrected = ve.corrected, w = w)
end

export SimpleVariance
