abstract type AbstractPosdefEstimator <: AbstractEstimator end
struct PosDefEstimator{T1} <: AbstractPosdefEstimator
    alg::T1
end
function PosDefEstimator(; alg = NearestCorrelationMatrix.Newton())
    return PosDefEstimator{typeof(alg)}(alg)
end
function posdef!(::Nothing, args...)
    return nothing
end
function posdef(::Nothing, args...)
    return nothing
end
function posdef!(method::PosDefEstimator{<:NearestCorrelationMatrix.NCMAlgorithm},
                 X::AbstractMatrix)
    if isposdef(X)
        return nothing
    end
    s = diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    NearestCorrelationMatrix.nearest_cor!(X, method.alg)
    if !isposdef(X)
        @warn("Matrix could not be made positive definite.")
    end
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end
function posdef(method::PosDefEstimator{<:NearestCorrelationMatrix.NCMAlgorithm},
                X::AbstractMatrix)
    X = copy(X)
    posdef!(method, X)
    return X
end

export PosDefEstimator, posdef, posdef!
