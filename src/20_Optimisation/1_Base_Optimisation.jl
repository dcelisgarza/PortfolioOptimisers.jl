abstract type AbstractOptimisationEstimator <: AbstractEstimator end
abstract type BaseOptimisationEstimator <: AbstractOptimisationEstimator end
abstract type OptimisationEstimator <: AbstractOptimisationEstimator end
abstract type OptimisationAlgorithm <: AbstractAlgorithm end
abstract type OptimisationResult <: AbstractResult end
abstract type OptimisationReturnCode <: AbstractResult end
struct OptimisationSuccess{T1} <: OptimisationReturnCode
    res::T1
end
function OptimisationSuccess(; res = nothing)
    return OptimisationSuccess{typeof(res)}(res)
end
struct OptimisationFailure{T1} <: OptimisationReturnCode
    res::T1
end
function OptimisationFailure(; res = nothing)
    return OptimisationFailure{typeof(res)}(res)
end
abstract type OptimisationModelResult <: AbstractResult end
Base.length(opt::OptimisationEstimator) = 1
Base.iterate(S::OptimisationEstimator, state = 1) = state > 1 ? nothing : (S, state + 1)
function opt_view(opt::OptimisationEstimator, args...)
    return opt
end
function optimise! end

export optimise!, OptimisationSuccess, OptimisationFailure
