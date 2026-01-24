abstract type CrossValidationEstimator <: AbstractEstimator end
abstract type CrossValidationResult <: AbstractResult end
abstract type CrossValidationAlgorithm <: AbstractAlgorithm end
function predict(res::NonFiniteAllocationOptimisationResult, pr::AbstractPriorResult,
                 fees::Option{<:Fees} = nothing)
    fees = hasproperty(res, :fees) && !isnothing(res.fees) ? res.fees : fees
    return calc_net_returns(res.w, pr.X, fees)
end
