struct NaiveOptimisationResult{T1 <: Type, T2 <: Union{Nothing, <:AbstractPriorResult},
                               T3 <: AbstractVector} <: OptimisationResult
    oe::T1
    pr::T2
    w::T3
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
    pr = prior(iv.pe, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, dims = dims)
    w = inv.(sqrt.(diag(pr.sigma)))
    return NaiveOptimisationResult(typeof(iv), pr, w / sum(w))
end
struct EqualWeighted <: OptimisationEstimator end
function optimise!(ew::EqualWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @smart_assert(!isnothing(rd.X))
    @smart_assert(dims ∈ (1, 2))
    dims = dims == 1 ? 2 : 1
    N = size(rd.X, dims)
    return NaiveOptimisationResult(typeof(ew), nothing,
                                   range(; start = inv(N), stop = inv(N), length = N))
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
    return NaiveOptimisationResult(typeof(rw), nothing, w)
end

export NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeights
