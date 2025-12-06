struct DenoiseCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
    ce::T1
    denoise::T2
    pdm::T3
    function DenoiseCovariance(ce::AbstractCovarianceEstimator,
                               denoise::Denoise = Denoise(),
                               pdm::Option{<:Posdef} = Posdef())
        return new{typeof(ce), typeof(denoise), typeof(pdm)}(ce, denoise, pdm)
    end
end
function DenoiseCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                           denoise::Denoise = Denoise(), pdm::Option{<:Posdef} = Posdef())
    return DenoiseCovariance(ce, denoise, pdm)
end
function factory(ce::DenoiseCovariance, w::Option{<:AbstractWeights} = nothing)
    return DenoiseCovariance(; ce = factory(ce.ce, w), denoise = ce.denoise, pdm = ce.pdm)
end
function Statistics.cov(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    T, N = size(X)
    posdef!(ce.pdm, sigma)
    denoise!(ce.denoise, sigma, T / N, ce.pdm)
    return sigma
end
function Statistics.cor(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    T, N = size(X)
    posdef!(ce.pdm, rho)
    denoise!(ce.denoise, rho, T / N, ce.pdm)
    return rho
end

export DenoiseCovariance