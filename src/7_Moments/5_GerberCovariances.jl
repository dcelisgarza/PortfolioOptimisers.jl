abstract type BaseGerberCovariance <: AbstractCovarianceEstimator end
abstract type GerberCovarianceAlgorithm <: AbstractCovarianceAlgorithm end
struct Gerber0 <: GerberCovarianceAlgorithm end
struct Gerber1 <: GerberCovarianceAlgorithm end
struct Gerber2 <: GerberCovarianceAlgorithm end
struct GerberCovariance{T1 <: GerberCovarianceAlgorithm,
                        T2 <: StatsBase.CovarianceEstimator, T3 <: PosDefEstimator,
                        T4 <: Real} <: BaseGerberCovariance
    alg::T1
    ve::T2
    pdm::T3
    threshold::T4
end
function GerberCovariance(; alg::GerberCovarianceAlgorithm = Gerber1(),
                          ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                          pdm::Union{Nothing, <:PosDefEstimator} = PosDefEstimator(),
                          threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return GerberCovariance{typeof(alg), typeof(ve), typeof(pdm), typeof(threshold)}(alg,
                                                                                     ve,
                                                                                     pdm,
                                                                                     threshold)
end
function gerber(ce::GerberCovariance{<:Gerber0, <:Any, <:Any, <:Any}, X::AbstractMatrix,
                std_vec::AbstractArray)
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
    fit_estimator!(ce.pdm, rho)
    return rho
end
function gerber(ce::GerberCovariance{<:Gerber1, <:Any, <:Any, <:Any}, X::AbstractMatrix,
                std_vec::AbstractArray)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    N = Matrix{Bool}(undef, T, N)
    std_vec = std_vec * ce.threshold
    U .= X .>= std_vec
    D .= X .<= -std_vec
    N .= .!U .& .!D
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    rho = transpose(UmD) * (UmD) ./ (T .- transpose(N) * N)
    fit_estimator!(ce.pdm, rho)
    return rho
end
function gerber(ce::GerberCovariance{<:Gerber2, <:Any, <:Any, <:Any}, X::AbstractMatrix,
                std_vec::AbstractArray)
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
    fit_estimator!(ce.pdm, rho)
    return rho
end
function StatsBase.cor(ce::GerberCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return gerber(ce, X, std_vec)
end
function StatsBase.cov(ce::GerberCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return gerber(ce, X, std_vec) .* (std_vec ⊗ std_vec)
end
function factory(ce::GerberCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return GerberCovariance(; alg = ce.alg, ve = factory(ce.ve, w), pdm = ce.pdm,
                            threshold = ce.threshold)
end
struct NormalisedGerberCovariance{T1 <: GerberCovarianceAlgorithm,
                                  T2 <: AbstractExpectedReturnsEstimator,
                                  T3 <: StatsBase.CovarianceEstimator,
                                  T4 <: PosDefEstimator, T5 <: Real} <: BaseGerberCovariance
    alg::T1
    me::T2
    ve::T3
    pdm::T4
    threshold::T5
end
function NormalisedGerberCovariance(; alg::GerberCovarianceAlgorithm = Gerber1(),
                                    me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                                    pdm::Union{Nothing, <:PosDefEstimator} = PosDefEstimator(),
                                    threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return NormalisedGerberCovariance{typeof(alg), typeof(me), typeof(ve), typeof(pdm),
                                      typeof(threshold)}(alg, me, ve, pdm, threshold)
end
function factory(ce::NormalisedGerberCovariance,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return NormalisedGerberCovariance(; alg = ce.alg, me = factory(ce.me, w),
                                      ve = factory(ce.ve, w), pdm = ce.pdm,
                                      threshold = ce.threshold)
end
function gerber(ce::NormalisedGerberCovariance{<:Gerber0, <:Any, <:Any, <:Any, <:Any},
                X::AbstractMatrix)
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
    fit_estimator!(ce.pdm, rho)
    return rho
end
function gerber(ce::NormalisedGerberCovariance{<:Gerber1, <:Any, <:Any, <:Any, <:Any},
                X::AbstractMatrix)
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
    fit_estimator!(ce.pdm, rho)
    return rho
end
function gerber(ce::NormalisedGerberCovariance{<:Gerber2, <:Any, <:Any, <:Any, <:Any},
                X::AbstractMatrix)
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
    fit_estimator!(ce.pdm, rho)
    return rho
end
function StatsBase.cor(ce::NormalisedGerberCovariance, X::AbstractMatrix; dims::Int = 1,
                       mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = 1) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ./ std_vec
    return gerber(ce, X)
end
function StatsBase.cov(ce::NormalisedGerberCovariance, X::AbstractMatrix; dims::Int = 1,
                       mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = 1) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ./ std_vec
    return gerber(ce, X) .* (std_vec ⊗ std_vec)
end

export GerberCovariance, NormalisedGerberCovariance, Gerber0, Gerber1, Gerber2
