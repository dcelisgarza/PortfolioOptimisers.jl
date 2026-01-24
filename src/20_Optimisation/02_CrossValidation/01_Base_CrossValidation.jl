abstract type CrossValidationEstimator <: AbstractEstimator end
abstract type CrossValidationResult <: AbstractResult end
abstract type CrossValidationAlgorithm <: AbstractAlgorithm end
function predict(res::NonFiniteAllocationOptimisationResult, pr::AbstractPriorResult,
                 args...; kwargs...)
    fees = hasproperty(res, :opt) && hasproperty(res.opt, :fees) ? res.opt.fees : nothing
    return calc_net_returns(res.w, pr.X, fees)
end
