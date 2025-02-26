struct Gerber0Covariance{T1 <: StatsBase.CovarianceEstimator,
                         T2 <: FixNonPositiveDefiniteMatrix, T3 <: Real} <:
       BaseGerberCovariance
    ve::T1
    fix_non_pos_def::T2
    threshold::T3
end
function Gerber0Covariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                           fix_non_pos_def::FixNonPositiveDefiniteMatrix = FNPDM_NearestCorrelationMatrix(),
                           threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return Gerber0Covariance{typeof(ve), typeof(fix_non_pos_def), typeof(threshold)}(ve,
                                                                                     fix_non_pos_def,
                                                                                     threshold)
end
function _gerber0(ce::Gerber0Covariance, X::AbstractMatrix, std_vec)
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
    UpD = U + D
    rho = (transpose(UmD) * UmD) ./ (transpose(UpD) * UpD)
    fix_non_positive_definite_matrix!(ce.fix_non_pos_def, rho)
    return rho
end
function StatsBase.cor(ce::Gerber0Covariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return _gerber0(ce, X, std_vec)
end
function StatsBase.cov(ce::Gerber0Covariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return _gerber0(ce, X, std_vec) .* (std_vec ⊗ std_vec)
end

export Gerber0Covariance
