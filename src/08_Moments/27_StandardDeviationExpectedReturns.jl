@concrete struct StandardDeviationExpectedReturns <: AbstractExpectedReturnsEstimator
    "$(field_dict[:ce])"
    ce
    function StandardDeviationExpectedReturns(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function StandardDeviationExpectedReturns(;
                                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance())
    return StandardDeviationExpectedReturns(ce)
end
function factory(ce::StandardDeviationExpectedReturns, w::ObsWeights)
    return StandardDeviationExpectedReturns(; ce = factory(ce.ce, w))
end
function Statistics.mean(me::StandardDeviationExpectedReturns, X::AbstractMatrix{<:Real};
                         dims::Int = 1, kwargs...)
    return Statistics.std(me.ce, X; dims = dims, kwargs...)
end

export StandardDeviationExpectedReturns
