struct Gerber0Covariance{T1 <: StatsBase.CovarianceEstimator,
                         T2 <: FixNonPositiveDefiniteMatrix, T3 <: Real} <:
       BaseGerberCovariance
    ve::T1
    fnpdm::T2
    threshold::T3
end
function Gerber0Covariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                           fnpdm::Union{Nothing, <:FixNonPositiveDefiniteMatrix} = FNPDM_NearestCorrelationMatrix(),
                           threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return Gerber0Covariance{typeof(ve), typeof(fnpdm), typeof(threshold)}(ve, fnpdm,
                                                                           threshold)
end
function moment_factory_w(ce::Gerber0Covariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return Gerber0Covariance(; ve = moment_factory_w(ce.ve, w), fnpdm = ce.fnpdm,
                             threshold = ce.threshold)
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
    fix_non_positive_definite_matrix!(ce.fnpdm, rho)
    return rho
end
function StatsBase.cor(ce::Gerber0Covariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return _gerber0(ce, X, std_vec)
end
function StatsBase.cov(ce::Gerber0Covariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return _gerber0(ce, X, std_vec) .* (std_vec ⊗ std_vec)
end

export Gerber0Covariance
