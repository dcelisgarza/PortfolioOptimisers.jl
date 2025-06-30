abstract type BaseGerberCovariance <: AbstractCovarianceEstimator end
abstract type GerberCovarianceAlgorithm <: AbstractMomentAlgorithm end
abstract type UnNormalisedGerberCovarianceAlgorithm <: GerberCovarianceAlgorithm end
abstract type NormalisedGerberCovarianceAlgorithm <: GerberCovarianceAlgorithm end
struct Gerber0 <: UnNormalisedGerberCovarianceAlgorithm end
struct Gerber1 <: UnNormalisedGerberCovarianceAlgorithm end
struct Gerber2 <: UnNormalisedGerberCovarianceAlgorithm end
struct NormalisedGerber0{T1 <: AbstractExpectedReturnsEstimator} <:
       NormalisedGerberCovarianceAlgorithm
    me::T1
end
function NormalisedGerber0(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
    return NormalisedGerber0{typeof(me)}(me)
end
struct NormalisedGerber1{T1 <: AbstractExpectedReturnsEstimator} <:
       NormalisedGerberCovarianceAlgorithm
    me::T1
end
function NormalisedGerber1(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
    return NormalisedGerber1{typeof(me)}(me)
end
struct NormalisedGerber2{T1 <: AbstractExpectedReturnsEstimator} <:
       NormalisedGerberCovarianceAlgorithm
    me::T1
end
function NormalisedGerber2(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
    return NormalisedGerber2{typeof(me)}(me)
end
for alg in (Gerber0, Gerber1, Gerber2)
    eval(quote
             function factory(alg::$(alg), ::Any)
                 return alg
             end
         end)
end
for alg in (NormalisedGerber0, NormalisedGerber1, NormalisedGerber2)
    eval(quote
             function factory(alg::$(alg), w::Union{Nothing, <:AbstractWeights})
                 return $(alg)(; me = factory(alg.me, w))
             end
         end)
end
struct GerberCovariance{T1 <: StatsBase.CovarianceEstimator, T2 <: PosDefEstimator,
                        T3 <: Real, T4 <: GerberCovarianceAlgorithm} <: BaseGerberCovariance
    ve::T1
    pdm::T2
    threshold::T3
    alg::T4
end
function GerberCovariance(; alg::GerberCovarianceAlgorithm = Gerber1(),
                          ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                          pdm::Union{Nothing, <:PosDefEstimator} = PosDefEstimator(),
                          threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return GerberCovariance{typeof(ve), typeof(pdm), typeof(threshold), typeof(alg)}(ve,
                                                                                     pdm,
                                                                                     threshold,
                                                                                     alg)
end
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix,
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
    rho = (transpose(UmD) * UmD) ⊘ (transpose(UpD) * UpD)
    posdef!(ce.pdm, rho)
    return rho
end
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0},
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
    rho = (transpose(UmD) * UmD) ⊘ (transpose(UpD) * UpD)
    posdef!(ce.pdm, rho)
    return rho
end
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix,
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
    rho = transpose(UmD) * (UmD) ⊘ (T .- transpose(N) * N)
    posdef!(ce.pdm, rho)
    return rho
end
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1},
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
    rho = transpose(UmD) * (UmD) ⊘ (T .- transpose(N) * N)
    posdef!(ce.pdm, rho)
    return rho
end
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix,
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
    rho = H ⊘ (h * transpose(h))
    posdef!(ce.pdm, rho)
    return rho
end
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2},
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
    rho = H ⊘ (h * transpose(h))
    posdef!(ce.pdm, rho)
    return rho
end
function StatsBase.cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                            <:UnNormalisedGerberCovarianceAlgorithm},
                       X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return gerber(ce, X, std_vec)
end
function StatsBase.cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                            <:UnNormalisedGerberCovarianceAlgorithm},
                       X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return gerber(ce, X, std_vec) ⊙ (std_vec ⊗ std_vec)
end
function StatsBase.cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                            <:NormalisedGerberCovarianceAlgorithm},
                       X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? StatsBase.mean(ce.alg.me, X; dims = 1, kwargs...) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec, kwargs...)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ⊘ std_vec
    return gerber(ce, X)
end
function StatsBase.cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                            <:NormalisedGerberCovarianceAlgorithm},
                       X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? StatsBase.mean(ce.alg.me, X; dims = 1, kwargs...) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec, kwargs...)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ⊘ std_vec
    return gerber(ce, X) ⊙ (std_vec ⊗ std_vec)
end
function factory(ce::GerberCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return GerberCovariance(; alg = factory(ce.alg, w), ve = factory(ce.ve, w),
                            pdm = ce.pdm, threshold = ce.threshold)
end

export GerberCovariance, Gerber0, Gerber1, Gerber2, NormalisedGerber0, NormalisedGerber1,
       NormalisedGerber2
