struct Gerber0NormalisedCovariance{T1 <: ExpectedReturnsEstimator,
                                   T2 <: StatsBase.CovarianceEstimator,
                                   T3 <: FixNonPositiveDefiniteMatrix, T4 <: Real} <:
       BaseGerberCovariance
    me::T1
    ve::T2
    fix_non_pos_def::T3
    threshold::T4
end
function Gerber0NormalisedCovariance(;
                                     me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                     ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                                     fix_non_pos_def::FixNonPositiveDefiniteMatrix = FNPDM_NearestCorrelationMatrix(),
                                     threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))

    return Gerber0NormalisedCovariance{typeof(me), typeof(ve), typeof(fix_non_pos_def),
                                       typeof(threshold)}(me, ve, fix_non_pos_def,
                                                          threshold)
end
function StatsBase.cor(ce::Gerber0NormalisedCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    T, N = size(X)

    mean_vec = mean(ce.me, X; dims = 1)
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)

    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))

    X = (X .- mean_vec) ./ std_vec

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
    fix_non_positive_definite_matrix!(ce.fix_non_pos_def, rho)

    return rho
end
function StatsBase.cov(ce::Gerber0NormalisedCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    T, N = size(X)

    mean_vec = mean(X; dims = 1)
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)

    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))

    X = (X .- mean_vec) ./ std_vec

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
    fix_non_positive_definite_matrix!(ce.fix_non_pos_def, rho)

    return rho .* (std_vec ⊗ std_vec)
end

export Gerber0NormalisedCovariance
