struct Gerber2Covariance{T1 <: AbstractBaseGerberCovariance} <: AbstractBaseGerberCovariance
    ce::T1
end
function Gerber2Covariance(; ce::BaseGerberCovariance = BaseGerberCovariance())
    return Gerber2Covariance{typeof(ce)}(ce)
end
function w_moment_factory(ce::Gerber2Covariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return Gerber2Covariance(; ce = w_moment_factory(ce.ce, w))
end
function _gerber2(ce::Gerber2Covariance, X::AbstractMatrix, std_vec)
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
    fit!(ce.pdm, rho)
    return rho
end
function StatsBase.cor(ce::Gerber2Covariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return _gerber2(ce, X, std_vec)
end
function StatsBase.cov(ce::Gerber2Covariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return _gerber2(ce, X, std_vec) .* (std_vec ⊗ std_vec)
end
struct Gerber2NormalisedCovariance{T1 <: BaseNormalisedCovariance} <:
       AbstractBaseNormalisedGerberCovariance
    ce::T1
end
function Gerber2NormalisedCovariance(;
                                     ce::BaseNormalisedCovariance = BaseNormalisedCovariance())
    return Gerber2NormalisedCovariance{typeof(ce)}(ce)
end
function w_moment_factory(ce::Gerber2NormalisedCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return Gerber2NormalisedCovariance(; ce = w_moment_factory(ce.ce, w))
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
    fit!(ce.pdm, rho)
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

export Gerber2Covariance, Gerber2NormalisedCovariance
