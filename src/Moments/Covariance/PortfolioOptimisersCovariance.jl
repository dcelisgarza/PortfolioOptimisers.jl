struct PortfolioOptimisersCovariance{T1 <: StatsBase.CovarianceEstimator,
                                     T2 <: MatrixProcessing} <:
       PortfolioOptimisersCovarianceEstimator
    ce::T1
    processing::T2
end
function PortfolioOptimisersCovariance(;
                                       ce::StatsBase.CovarianceEstimator = FullCovariance(),
                                       processing::MatrixProcessing = DefaultMatrixProcessing())
    return PortfolioOptimisersCovariance{typeof(ce), typeof(processing)}(ce, processing)
end
function StatsBase.cov(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    T, N = size(X)
    sigma = cov(ce.ce, X)
    mtx_process!(ce.processing, sigma, T, N)
    return sigma
end
function StatsBase.cor(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    T, N = size(X)
    rho = cor(ce.ce, X)
    mtx_process!(ce.processing, rho, T, N)
    return rho
end

export PortfolioOptimisersCovariance
