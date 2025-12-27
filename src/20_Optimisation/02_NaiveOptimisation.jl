abstract type NaiveOptimisationEstimator <: OptimisationEstimator end
function assert_internal_optimiser(::NaiveOptimisationEstimator)
    return nothing
end
function assert_external_optimiser(::NaiveOptimisationEstimator)
    return nothing
end
struct NaiveOptimisation{T1, T2, T3, T4, T5} <: OptimisationResult
    oe::T1
    pr::T2
    w::T3
    retcode::T4
    fb::T5
end
function factory(res::NaiveOptimisation, fb)
    return NaiveOptimisation(res.oe, res.pr, res.w, res.retcode, fb)
end
struct InverseVolatility{T1, T2} <: NaiveOptimisationEstimator
    pe::T1
    fb::T2
    function InverseVolatility(pe::PrE_Pr, fb::Option{<:OptimisationEstimator} = nothing)
        return new{typeof(pe), typeof(fb)}(pe, fb)
    end
end
function InverseVolatility(; pe::PrE_Pr = EmpiricalPrior(),
                           fb::Option{<:OptimisationEstimator} = nothing)
    return InverseVolatility(pe, fb)
end
function opt_view(opt::InverseVolatility, i, args...)
    pe = prior_view(opt.pe, i)
    return InverseVolatility(; pe = pe, fb = opt.fb)
end
function assert_external_optimiser(opt::InverseVolatility)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
function _optimise(iv::InverseVolatility, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, kwargs...)
    pr = prior(iv.pe, rd; dims = dims)
    w = inv.(sqrt.(diag(pr.sigma)))
    return NaiveOptimisation(typeof(iv), pr, w / sum(w), OptimisationSuccess(nothing),
                             nothing)
end
function optimise(iv::InverseVolatility{<:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(iv, rd; dims = dims, kwargs...)
end
struct EqualWeighted <: NaiveOptimisationEstimator end
function optimise(ew::EqualWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X))
    @argcheck(dims in (1, 2))
    dims = dims == 1 ? 2 : 1
    N = size(rd.X, dims)
    iN = inv(N)
    return NaiveOptimisation(typeof(ew), nothing, range(iN, iN; length = N),
                             OptimisationSuccess(nothing), nothing)
end
struct RandomWeighted{T1} <: NaiveOptimisationEstimator
    rng::T1
    function RandomWeighted(rng::Option{<:AbstractRNG})
        return new{typeof(rng)}(rng)
    end
end
function RandomWeighted(; rng::Option{<:AbstractRNG} = nothing)
    return RandomWeighted(rng)
end
function optimise(rw::RandomWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X))
    @argcheck(dims in (1, 2))
    dims = dims == 1 ? 2 : 1
    N = size(rd.X, dims)
    w = if isnothing(rw.rng)
        rand(Distributions.Dirichlet(N, 1))
    else
        rand(rw.rng, Distributions.Dirichlet(N, 1))
    end
    return NaiveOptimisation(typeof(rw), nothing, w, OptimisationSuccess(nothing), nothing)
end

export NaiveOptimisation, InverseVolatility, EqualWeighted, RandomWeighted
