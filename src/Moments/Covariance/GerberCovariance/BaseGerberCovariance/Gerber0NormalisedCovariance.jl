struct Gerber0NormalisedCovariance{T1 <: ExpectedReturnsEstimator,
                                   T2 <: StatsBase.CovarianceEstimator,
                                   T3 <: FixNonPositiveDefiniteMatrix, T4 <: Real} <:
       BaseGerberCovariance
    me::T1
    ve::T2
    fnpd::T3
    threshold::T4
end
function Gerber0NormalisedCovariance(;
                                     me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                     ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                                     fnpd::FixNonPositiveDefiniteMatrix = FNPD_NearestCorrelationMatrix(),
                                     threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return Gerber0NormalisedCovariance{typeof(me), typeof(ve), typeof(fnpd),
                                       typeof(threshold)}(me, ve, fnpd, threshold)
end
function _gerber0normalised(ce::Gerber0NormalisedCovariance, X::AbstractMatrix)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    U .= X .>= ce.threshold
    D .= X .<= -ce.threshold
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    UpD = U + D
    rho = (transpose(UmD) * UmD) ./ (transpose(UpD) * UpD)
    fix_non_positive_definite_matrix!(ce.fnpd, rho)
    return rho
end
function StatsBase.cor(ce::Gerber0NormalisedCovariance, X::AbstractMatrix; dims::Int = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = mean(ce.me, X; dims = 1)
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ./ std_vec
    return _gerber0normalised(ce, X)
end
function StatsBase.cov(ce::Gerber0NormalisedCovariance, X::AbstractMatrix; dims::Int = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = mean(ce.me, X; dims = 1)
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ./ std_vec
    return _gerber0normalised(ce, X) .* (std_vec ⊗ std_vec)
end

export Gerber0NormalisedCovariance
