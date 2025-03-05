struct Gerber2Covariance{T1 <: StatsBase.CovarianceEstimator,
                         T2 <: FixNonPositiveDefiniteMatrix, T3 <: Real} <:
       BaseGerberCovariance
    ve::T1
    fnpdm::T2
    threshold::T3
end
function Gerber2Covariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                           fnpdm::FixNonPositiveDefiniteMatrix = FNPDM_NearestCorrelationMatrix(),
                           threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return Gerber2Covariance{typeof(ve), typeof(fnpdm), typeof(threshold)}(ve, fnpdm,
                                                                           threshold)
end
function _gerber2(ce::Gerber2Covariance, X::AbstractMatrix, std_vec)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    std_vec = std_vec * ce.threshold
    U .= X .>= std_vec
    D .= X .<= -std_vec
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    H = transpose(UmD) * (UmD)
    h = sqrt.(diag(H))
    rho = H ./ (h * transpose(h))
    fix_non_positive_definite_matrix!(ce.fnpdm, rho)
    return rho
end
function StatsBase.cor(ce::Gerber2Covariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return _gerber2(ce, X, std_vec)
end
function StatsBase.cov(ce::Gerber2Covariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return _gerber2(ce, X, std_vec) .* (std_vec ⊗ std_vec)
end

export Gerber2Covariance
