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
struct Frontier{T1 <: Integer, T2 <: Real, T3 <: Bool} <: AbstractAlgorithm
    N::T1
    factor::T2
    flag::T3
end
function Frontier(; N::Integer = 20, factor::Real = 1, flag::Bool = false)
    @smart_assert(N > zero(N))
    @smart_assert(factor > zero(factor))
    return Frontier{typeof(N), typeof(factor), typeof(flag)}(N, factor, flag)
end
abstract type OptimisationModelResult <: AbstractResult end
Base.length(opt::OptimisationEstimator) = 1
Base.iterate(S::OptimisationEstimator, state = 1) = state > 1 ? nothing : (S, state + 1)
function opt_view(opt::OptimisationEstimator, args...)
    return opt
end
function optimise! end
function efficient_frontier! end

export optimise!, efficient_frontier!, OptimisationSuccess, OptimisationFailure, Frontier
