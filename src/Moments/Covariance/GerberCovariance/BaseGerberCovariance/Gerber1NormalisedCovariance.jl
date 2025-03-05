struct Gerber1NormalisedCovariance{T1 <: ExpectedReturnsEstimator,
                                   T2 <: StatsBase.CovarianceEstimator,
                                   T3 <: FixNonPositiveDefiniteMatrix, T4 <: Real} <:
       BaseGerberCovariance
    me::T1
    ve::T2
    fnpdm::T3
    threshold::T4
end
function Gerber1NormalisedCovariance(;
                                     me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                     ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                                     fnpdm::FixNonPositiveDefiniteMatrix = FNPDM_NearestCorrelationMatrix(),
                                     threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return Gerber1NormalisedCovariance{typeof(me), typeof(ve), typeof(fnpdm),
                                       typeof(threshold)}(me, ve, fnpdm, threshold)
end
function _gerber1normalised(ce::Gerber1NormalisedCovariance, X::AbstractMatrix)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    N = Matrix{Bool}(undef, T, N)
    U .= X .>= ce.threshold
    D .= X .<= -ce.threshold
    N .= .!U .& .!D
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    rho = transpose(UmD) * (UmD) ./ (T .- transpose(N) * N)
    fix_non_positive_definite_matrix!(ce.fnpdm, rho)
    return rho
end
function StatsBase.cor(ce::Gerber1NormalisedCovariance, X::AbstractMatrix; dims::Int = 1,
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
    return _gerber1normalised(ce, X)
end
function StatsBase.cov(ce::Gerber1NormalisedCovariance, X::AbstractMatrix; dims::Int = 1,
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
    return _gerber1normalised(ce, X) .* (std_vec ⊗ std_vec)
end

export Gerber1NormalisedCovariance
