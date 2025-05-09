abstract type OptimisationEstimator <: AbstractEstimator end
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
function opt_view(opt::OptimisationEstimator, ::Any)
    return opt
end
function opt_view(opt::AbstractVector{<:OptimisationEstimator}, i::AbstractVector)
    return opt_view.(opt, Ref(i))
end
Base.length(opt::OptimisationEstimator) = 1
Base.iterate(S::OptimisationEstimator, state = 1) = state > 1 ? nothing : (S, state + 1)
function optimise! end

export optimise!, OptimisationSuccess, OptimisationFailure
