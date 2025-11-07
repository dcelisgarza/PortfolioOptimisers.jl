abstract type AbstractOptimisationEstimator <: AbstractEstimator end
abstract type BaseOptimisationEstimator <: AbstractOptimisationEstimator end
abstract type OptimisationEstimator <: AbstractOptimisationEstimator end
abstract type OptimisationAlgorithm <: AbstractAlgorithm end
abstract type OptimisationResult <: AbstractResult end
abstract type OptimisationReturnCode <: AbstractResult end
abstract type OptimisationModelResult <: AbstractResult end
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
    function SingletonOptimisation(retcode::OptimisationReturnCode)
        return new{typeof(retcode)}(retcode)
    end
end
function SingletonOptimisation(; retcode::OptimisationReturnCode)
    return SingletonOptimisation(retcode)
end
function opt_view(opt::AbstractOptimisationEstimator, args...)
    return opt
end
function opt_view(opt::AbstractVector{<:AbstractOptimisationEstimator}, args...)
    return [opt_view(opti, args...) for opti in opt]
end
function optimise end
function optimise(or::OptimisationResult, args...)
    return or
end
function opt_attempt_factory end
function optimise(opt::OptimisationEstimator, args...; kwargs...)
    fb = Tuple{OptimisationEstimator, OptimisationResult}[]
    current_opt = opt
    res = nothing
    while true
        res = _optimise(current_opt, args...; kwargs...)
        if isa(res.retcode, OptimisationSuccess) || isnothing(current_opt.fb)
            break
        else
            push!(fb, (current_opt, res))
            current_opt = current_opt.fb
            @warn("Using fallback method. Please ignore previous optimisation failure warnings.")
        end
    end
    return isempty(fb) ? res : opt_attempt_factory(res, fb)
end
function assert_internal_optimiser(::OptimisationResult)
    return nothing
end
function assert_external_optimiser(::OptimisationResult)
    return nothing
end
function predict_outer_estimator_returns(opt::OptimisationEstimator, rd::ReturnsResult,
                                         pr::AbstractPriorResult, wi::NumMat,
                                         resi::AbstractVector{<:OptimisationResult};
                                         kwargs...)
    iv = isnothing(rd.iv) ? nothing : rd.iv * wi
    ivpa = (isnothing(rd.ivpa) || isa(rd.ivpa, Number)) ? rd.ivpa : transpose(wi) * rd.ivpa
    return pr.X * wi, rd.F, rd.ts, iv, ivpa
end

export optimise, OptimisationSuccess, OptimisationFailure
