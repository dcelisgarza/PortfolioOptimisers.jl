struct Gerber2NormalisedCovariance{T1 <: ExpectedReturnsEstimator,
                                   T2 <: StatsBase.CovarianceEstimator,
                                   T3 <: FixNonPositiveDefiniteMatrix, T4 <: Real} <:
       BaseGerberCovariance
    me::T1
    ve::T2
    fnpdm::T3
    threshold::T4
end
function Gerber2NormalisedCovariance(;
                                     me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                     ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                                     fnpdm::FixNonPositiveDefiniteMatrix = FNPDM_NearestCorrelationMatrix(),
                                     threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return Gerber2NormalisedCovariance{typeof(me), typeof(ve), typeof(fnpdm),
                                       typeof(threshold)}(me, ve, fnpdm, threshold)
end
function moment_factory_w(ce::Gerber2NormalisedCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return Gerber2NormalisedCovariance(; me = moment_factory_w(ce.me, w),
                                       ve = moment_factory_w(ce.ve, w), fnpdm = ce.fnpdm,
                                       threshold = ce.threshold)
end
function _gerber2normalised(ce::Gerber2NormalisedCovariance, X::AbstractMatrix)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    U .= X .>= ce.threshold
    D .= X .<= -ce.threshold
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
function StatsBase.cor(ce::Gerber2NormalisedCovariance, X::AbstractMatrix; dims::Int = 1,
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
    return _gerber2normalised(ce, X)
end
function StatsBase.cov(ce::Gerber2NormalisedCovariance, X::AbstractMatrix; dims::Int = 1,
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
    return _gerber2normalised(ce, X) .* (std_vec ⊗ std_vec)
end

export Gerber2NormalisedCovariance
