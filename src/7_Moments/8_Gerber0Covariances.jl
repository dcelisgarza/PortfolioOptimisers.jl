struct Gerber0Covariance{T1 <: AbstractBaseGerberCovariance} <: AbstractBaseGerberCovariance
    ce::T1
end
function Gerber0Covariance(; ce::BaseGerberCovariance = BaseGerberCovariance())
    return Gerber0Covariance{typeof(ce)}(ce)
end
function w_moment_factory(ce::Gerber0Covariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return Gerber0Covariance(; ce = w_moment_factory(ce.ce, w))
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
    fit!(ce.pdm, rho)
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
struct Gerber0NormalisedCovariance{T1 <: BaseNormalisedCovariance} <:
       AbstractBaseNormalisedGerberCovariance
    ce::T1
end
function Gerber0NormalisedCovariance(;
                                     ce::BaseNormalisedCovariance = BaseNormalisedCovariance())
    return Gerber0NormalisedCovariance{typeof(ce)}(ce)
end
function w_moment_factory(ce::Gerber0NormalisedCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return Gerber0NormalisedCovariance(; ce = w_moment_factory(ce.ce, w))
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
    fit!(ce.pdm, rho)
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

export Gerber0Covariance, Gerber0NormalisedCovariance