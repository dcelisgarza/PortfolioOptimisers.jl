abstract type NaiveOptimisationEstimator <: OptimisationEstimator end
function assert_internal_optimiser(::NaiveOptimisationEstimator)
    return nothing
end
function assert_external_optimiser(::NaiveOptimisationEstimator)
    return nothing
end
struct NaiveOptimisation{T1,T2,T3,T4} <: OptimisationResult
    oe::T1
    pr::T2
    w::T3
    retcode::T4
end
struct InverseVolatility{T1} <: NaiveOptimisationEstimator
    pe::T1
end
function InverseVolatility(;
    pe::Union{<:AbstractPriorEstimator,<:AbstractPriorResult} = EmpiricalPrior(),
)
    return InverseVolatility(pe)
end
function opt_view(opt::InverseVolatility, i::AbstractVector, args...)
    pe = prior_view(opt.pe, i)
    return InverseVolatility(; pe = pe)
end
function assert_external_optimiser(opt::InverseVolatility)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
function optimise!(
    iv::InverseVolatility,
    rd::ReturnsResult = ReturnsResult();
    dims::Int = 1,
    kwargs...,
)
    pr = prior(iv.pe, rd; dims = dims)
    w = inv.(sqrt.(diag(pr.sigma)))
    return NaiveOptimisation(typeof(iv), pr, w / sum(w), OptimisationSuccess(nothing))
end
struct EqualWeighted <: NaiveOptimisationEstimator end
function optimise!(ew::EqualWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X))
    @argcheck(dims in (1, 2))
    dims = dims == 1 ? 2 : 1
    N = size(rd.X, dims)
    return NaiveOptimisation(
        typeof(ew),
        nothing,
        range(; start = inv(N), stop = inv(N), length = N),
        OptimisationSuccess(nothing),
    )
end
struct RandomWeights{T1} <: NaiveOptimisationEstimator
    rng::T1
end
function RandomWeights(; rng::Union{Nothing,<:AbstractRNG} = nothing)
    return RandomWeights(rng)
end
function optimise!(rw::RandomWeights, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X))
    @argcheck(dims in (1, 2))
    dims = dims == 1 ? 2 : 1
    N = size(rd.X, dims)
    w = isnothing(rw.rng) ? rand(Dirichlet(N, 1)) : rand(rw.rng, Dirichlet(N, 1))
    return NaiveOptimisation(typeof(rw), nothing, w, OptimisationSuccess(nothing))
end

export NaiveOptimisation, InverseVolatility, EqualWeighted, RandomWeights
