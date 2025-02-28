struct FNPDM_NearestCorrelationMatrix{T1 <: NearestCorrelationMatrix.NCMAlgorithm} <:
       FixNonPositiveDefiniteMatrix
    alg::T1
end
function FNPDM_NearestCorrelationMatrix(;
                                        alg::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton())
    return FNPDM_NearestCorrelationMatrix{typeof(alg)}(alg)
end
function fix_non_positive_definite_matrix!(method::FNPDM_NearestCorrelationMatrix,
                                           X::AbstractMatrix)
    if isposdef(X)
        return nothing
    end

    s = diag(X)
    iscov = any(.!isone.(s))
    _X = if iscov
        s .= sqrt.(s)
        cov2cor(X, s)
    else
        X
    end

    NearestCorrelationMatrix.nearest_cor!(_X, method.alg)

    if !isposdef(_X)
        @warn("Matrix could not be made positive definite.")
        return nothing
    end

    if iscov
        StatsBase.cor2cov!(_X, s)
    end

    X .= _X

    return nothing
end

export FNPDM_NearestCorrelationMatrix
