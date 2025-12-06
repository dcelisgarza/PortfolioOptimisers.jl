struct DetoneCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
    ce::T1
    detone::T2
    pdm::T3
    function DetoneCovariance(ce::AbstractCovarianceEstimator, detone::Detone = Detone(),
                              pdm::Option{<:Posdef} = Posdef())
        return new{typeof(ce), typeof(detone), typeof(pdm)}(ce, detone, pdm)
    end
end
function DetoneCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                          detone::Detone = Detone(), pdm::Option{<:Posdef} = Posdef())
    return DetoneCovariance(ce, detone, pdm)
end
function factory(ce::DetoneCovariance, w::Option{<:AbstractWeights} = nothing)
    return DetoneCovariance(; ce = factory(ce.ce, w), detone = ce.detone, pdm = ce.pdm)
end
function Statistics.cov(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    posdef!(ce.pdm, sigma)
    detone!(ce.detone, sigma, ce.pdm)
    return sigma
end
function Statistics.cor(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    posdef!(ce.pdm, rho)
    detone!(ce.detone, rho, ce.pdm)
    return rho
end

export DetoneCovariance