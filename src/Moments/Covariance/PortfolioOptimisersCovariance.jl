struct PortfolioOptimisersCovariance{T1 <: PortfolioOptimisersCovarianceEstimator,
                                     T2 <: MatrixProcessing} <:
       PortfolioOptimisersCovarianceEstimator
    ce::T1
    mp::T2
end
function PortfolioOptimisersCovariance(;
                                       ce::PortfolioOptimisersCovarianceEstimator = FullCovariance(),
                                       mp::MatrixProcessing = DefaultMatrixProcessing())
    return PortfolioOptimisersCovariance{typeof(ce), typeof(mp)}(ce, mp)
end
function StatsBase.cov(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    T, N = size(X)
    sigma = cov(ce.ce, X)
    mtx_process!(ce.mp, sigma, T, N)
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
    mtx_process!(ce.mp, rho, T, N)
    return rho
end

export PortfolioOptimisersCovariance
