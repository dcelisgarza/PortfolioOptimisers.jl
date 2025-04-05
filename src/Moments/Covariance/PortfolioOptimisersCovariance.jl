struct PortfolioOptimisersCovariance{T1 <: PortfolioOptimisersCovarianceEstimator,
                                     T2 <: AbstractMatrixProcessingEstimator} <:
       PortfolioOptimisersCovarianceEstimator
    ce::T1
    mp::T2
end
function PortfolioOptimisersCovariance(;
                                       ce::PortfolioOptimisersCovarianceEstimator = FullCovariance(),
                                       mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())
    return PortfolioOptimisersCovariance{typeof(ce), typeof(mp)}(ce, mp)
end
function w_moment_factory(ce::PortfolioOptimisersCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return PortfolioOptimisersCovariance(; ce = w_moment_factory(ce.ce, w), mp = ce.mp)
end
function StatsBase.cov(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = cov(ce.ce, X; kwargs...)
    fit!(ce.mp, sigma, X)
    return sigma
end
function StatsBase.cor(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = cor(ce.ce, X; kwargs...)
    fit!(ce.mp, rho, X)
    return rho
end

export PortfolioOptimisersCovariance
