"""
"""
struct StandardDeviationExpectedReturns{T1} <: AbstractExpectedReturnsEstimator
    ce::T1
    function StandardDeviationExpectedReturns(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function StandardDeviationExpectedReturns(;
                                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance())
    return StandardDeviationExpectedReturns(ce)
end
function factory(ce::StandardDeviationExpectedReturns,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)
    return StandardDeviationExpectedReturns(; ce = factory(ce.ce, w))
end
function Statistics.mean(me::StandardDeviationExpectedReturns, X::AbstractMatrix{<:Real};
                         dims::Int = 1, kwargs...)
    mu = sqrt.(LinearAlgebra.diag(Statistics.cov(me.ce, X; dims = dims, kwargs...)))
    if isone(dims)
        mu = reshape(mu, 1, :)
    end
    return mu
end

export StandardDeviationExpectedReturns
