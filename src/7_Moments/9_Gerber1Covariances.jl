struct Gerber1Covariance{T1 <: AbstractBaseGerberCovariance} <: AbstractBaseGerberCovariance
    ce::T1
end
function Gerber1Covariance(; ce::BaseGerberCovariance = BaseGerberCovariance())
    return Gerber1Covariance{typeof(ce)}(ce)
end
function w_moment_factory(ce::Gerber1Covariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return Gerber1Covariance(; ce = w_moment_factory(ce.ce, w))
end
function _gerber1(ce::Gerber1Covariance, X::AbstractMatrix, std_vec)
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
    fit!(ce.pdm, rho)
    return rho
end
function StatsBase.cor(ce::Gerber1Covariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return _gerber1(ce, X, std_vec)
end
function StatsBase.cov(ce::Gerber1Covariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return _gerber1(ce, X, std_vec) .* (std_vec ⊗ std_vec)
end
struct Gerber1NormalisedCovariance{T1 <: BaseNormalisedCovariance} <:
       AbstractBaseNormalisedGerberCovariance
    ce::T1
end
function Gerber1NormalisedCovariance(;
                                     ce::BaseNormalisedCovariance = BaseNormalisedCovariance())
    return Gerber1NormalisedCovariance{typeof(ce)}(ce)
end
function w_moment_factory(ce::Gerber1NormalisedCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return Gerber1NormalisedCovariance(; ce = w_moment_factory(ce.ce, w))
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
    fit!(ce.pdm, rho)
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

export Gerber1Covariance, Gerber1NormalisedCovariance
