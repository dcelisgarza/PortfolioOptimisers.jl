struct NaiveOptimisationResult{T1 <: AbstractVector} <: OptimisationResult
    w::T1
end
function NaiveOptimisationResult(; w::AbstractVector)
    @smart_assert(!isempty(w))
    return NaiveOptimisationResult{typeof(w)}(w)
end
struct InverseVolatility{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult}} <:
       OptimisationEstimator
    pe::T1
end
function InverseVolatility(;
                           pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator())
    return InverseVolatility{typeof(pe)}(pe)
end
function opt_view(opt::InverseVolatility, i::AbstractVector, args...)
    pe = prior_view(opt.pe, i)
    return InverseVolatility(; pe = pe)
end
function optimise!(iv::InverseVolatility, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, kwargs...)
    pr = prior(iv.pe, rd.X, rd.F; dims = dims)
    w = inv.(sqrt.(diag(pr.sigma)))
    return NaiveOptimisationResult(; w = w / sum(w))
end
struct EqualWeighted <: OptimisationEstimator end
function optimise!(ew::EqualWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @smart_assert(!isnothing(rd.X))
    @smart_assert(dims ∈ (1, 2))
    dims = dims == 1 ? 2 : 1
    N = size(rd.X, dims)
    return NaiveOptimisationResult(; w = range(; start = inv(N), stop = inv(N), length = N))
end
struct RandomWeights{T1 <: Union{Nothing, <:AbstractRNG}} <: OptimisationEstimator
    rng::T1
end
function RandomWeights(; rng::Union{Nothing, <:AbstractRNG} = nothing)
    return RandomWeights{typeof(rng)}(rng)
end
function optimise!(rw::RandomWeights, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @smart_assert(!isnothing(rd.X))
    @smart_assert(dims ∈ (1, 2))
    dims = dims == 1 ? 2 : 1
    N = size(rd.X, dims)
    w = isnothing(rw.rng) ? rand(Dirichlet(N, 1)) : rand(rw.rng, Dirichlet(N, 1))
    return NaiveOptimisationResult(; w = w)
end

export NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeights