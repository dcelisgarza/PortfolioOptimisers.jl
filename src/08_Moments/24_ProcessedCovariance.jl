struct ProcessedCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
    ce::T1
    alg::T2
    pdm::T3
    function ProcessedCovariance(ce::AbstractCovarianceEstimator,
                                 alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                                 pdm::Option{<:Posdef} = Posdef())
        return new{typeof(ce), typeof(alg), typeof(pdm)}(ce, alg, pdm)
    end
end
function ProcessedCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                             alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                             pdm::Option{<:Posdef} = Posdef())
    return ProcessedCovariance(ce, alg, pdm)
end
function factory(ce::ProcessedCovariance, w::Option{<:AbstractWeights} = nothing)
    return ProcessedCovariance(; ce = factory(ce.ce, w), alg = ce.alg, pdm = ce.pdm)
end
function Statistics.cov(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    posdef!(ce.pdm, sigma)
    matrix_processing_algorithm!(ce.alg, ce.pdm, sigma, X; kwargs...)
    return sigma
end
function Statistics.cor(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    posdef!(ce.pdm, rho)
    matrix_processing_algorithm!(ce.alg, ce.pdm, rho, X; kwargs...)
    return rho
end

export ProcessedCovariance