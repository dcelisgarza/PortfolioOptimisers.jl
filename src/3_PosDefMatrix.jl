abstract type PosDefMatrixEstimator <: AbstractEstimator end
struct NearestPosDef{T1 <: NearestCorrelationMatrix.NCMAlgorithm} <: PosDefMatrixEstimator
    alg::T1
end
function NearestPosDef(;
                       alg::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton())
    return NearestPosDef{typeof(alg)}(alg)
end
function fit!(method::NearestPosDef, X::AbstractMatrix)
    if isposdef(X)
        return nothing
    end
    s = diag(X)
    iscov = any(.!isone.(s))
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
function fit(method::NearestPosDef, X::AbstractMatrix)
    X = copy(X)
    fit!(method, X)
    return X
end

export NearestPosDef
