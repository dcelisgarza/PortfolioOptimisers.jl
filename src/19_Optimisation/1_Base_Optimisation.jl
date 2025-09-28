abstract type AbstractOptimisationEstimator <: AbstractEstimator end
abstract type BaseOptimisationEstimator <: AbstractOptimisationEstimator end
abstract type OptimisationEstimator <: AbstractOptimisationEstimator end
abstract type OptimisationAlgorithm <: AbstractAlgorithm end
abstract type OptimisationResult <: AbstractResult end
abstract type OptimisationReturnCode <: AbstractResult end
abstract type CrossValidationEstimator <: AbstractEstimator end
abstract type CrossValidationResult <: AbstractResult end
abstract type CrossValidationAlgorithm <: AbstractAlgorithm end
struct OptimisationSuccess{T1} <: OptimisationReturnCode
    res::T1
end
function OptimisationSuccess(; res = nothing)
    return OptimisationSuccess(res)
end
struct OptimisationFailure{T1} <: OptimisationReturnCode
    res::T1
end
function OptimisationFailure(; res = nothing)
    return OptimisationFailure(res)
end
struct SingletonOptimisation{T1} <: OptimisationResult
    retcode::T1
end
function SingletonOptimisation(; retcode::OptimisationReturnCode)
    return SingletonOptimisation(retcode)
end
abstract type OptimisationModelResult <: AbstractResult end
function opt_view(opt::AbstractOptimisationEstimator, args...)
    return opt
end
function opt_view(opt::AbstractVector{<:AbstractOptimisationEstimator}, args...)
    return [opt_view(opti, args...) for opti in opt]
end
function optimise! end
function optimise!(or::OptimisationResult, args...)
    return or
end
function assert_internal_optimiser(::OptimisationResult)
    return nothing
end
function assert_external_optimiser(::OptimisationResult)
    return nothing
end
function predict_outer_estimator_returns(opt::OptimisationEstimator, rd::ReturnsResult,
                                         pr::AbstractPriorResult, wi::AbstractMatrix,
                                         resi::AbstractVector{<:OptimisationResult};
                                         kwargs...)
    iv = isnothing(rd.iv) ? nothing : rd.iv * wi
    ivpa = (isnothing(rd.ivpa) || isa(rd.ivpa, Real)) ? rd.ivpa : transpose(wi) * rd.ivpa
    return pr.X * wi, rd.F, rd.ts, iv, ivpa
end

export optimise!, OptimisationSuccess, OptimisationFailure
