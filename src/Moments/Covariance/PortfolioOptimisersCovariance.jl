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
function moment_factory_w(ce::PortfolioOptimisersCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return PortfolioOptimisersCovariance(; ce = moment_factory_w(ce.ce, w), mp = ce.mp)
end
function StatsBase.cov(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = cov(ce.ce, X; kwargs...)
    mtx_process!(ce.mp, sigma, X)
    return sigma
end
function StatsBase.cor(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = cor(ce.ce, X; kwargs...)
    mtx_process!(ce.mp, rho, X)
    return rho
end

export PortfolioOptimisersCovariance
