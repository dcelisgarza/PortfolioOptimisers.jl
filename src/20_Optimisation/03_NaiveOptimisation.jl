abstract type NaiveOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
function needs_previous_weights(opt::NaiveOptimisationEstimator)
    return needs_previous_weights(opt.fb)
end
function assert_internal_optimiser(::NaiveOptimisationEstimator)
    return nothing
end
function assert_external_optimiser(::NaiveOptimisationEstimator)
    return nothing
end
struct NaiveOptimisationResult{T1, T2, T3, T4, T5, T6} <:
       NonFiniteAllocationOptimisationResult
    oe::T1
    pr::T2
    wb::T4
    retcode::T5
    w::T3
    fb::T6
end
function factory(res::NaiveOptimisationResult, fb::Option{<:OptE_Opt})
    return NaiveOptimisationResult(res.oe, res.pr, res.wb, res.retcode, res.w, fb)
end
struct InverseVolatility{T1, T2, T3, T4, T5, T6, T7} <: NaiveOptimisationEstimator
    pe::T1
    wb::T2
    sets::T3
    wf::T4
    fb::T5
    rtr::T6
    strict::T7
    function InverseVolatility(pe::PrE_Pr, wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                               wf::WeightFinaliser, fb::Option{<:OptE_Opt}, rtr::Bool,
                               strict::Bool)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pe), typeof(wb), typeof(sets), typeof(wf), typeof(fb),
                   typeof(rtr), typeof(strict)}(pe, wb, sets, wf, fb, rtr, strict)
    end
end
function InverseVolatility(; pe::PrE_Pr = EmpiricalPrior(),
                           wb::Option{<:WbE_Wb} = WeightBounds(),
                           sets::Option{<:AssetSets} = nothing,
                           wf::WeightFinaliser = IterativeWeightFinaliser(),
                           fb::Option{<:OptE_Opt} = nothing, rtr::Bool = false,
                           strict::Bool = false)
    return InverseVolatility(pe, wb, sets, wf, fb, rtr, strict)
end
function factory(opt::InverseVolatility, w::AbstractVector)
    return InverseVolatility(; pe = opt.pe, wb = opt.wb, sets = opt.sets, wf = opt.wf,
                             fb = factory(opt.fb, w), rtr = opt.rtr, strict = opt.strict)
end
function opt_view(opt::InverseVolatility, i, args...)
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    sets = nothing_asset_sets_view(opt.sets, i)
    return InverseVolatility(; pe = pe, wb = wb, sets = sets, wf = opt.wf, fb = opt.fb,
                             rtr = opt.rtr, strict = opt.strict)
end
function assert_external_optimiser(opt::InverseVolatility)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
function _optimise(iv::InverseVolatility, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    rd = returns_result_picker(rd, iv.rtr)
    pr = prior(iv.pe, rd; dims = dims)
    w = inv.(sqrt.(LinearAlgebra.diag(pr.sigma)))
    w /= sum(w)
    wb = weight_bounds_constraints(iv.wb, iv.sets;
                                   N = size(pr.X, ifelse(isone(dims), 2, 1)),
                                   strict = iv.strict, datatype = eltype(pr.X))
    retcode, w = finalise_weight_bounds(iv.wf, wb, w)
    return NaiveOptimisationResult(typeof(iv), pr, wb, retcode, w, nothing)
end
function optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(iv, rd; dims = dims, kwargs...)
end
struct EqualWeighted{T1, T2, T3, T4, T5} <: NaiveOptimisationEstimator
    wb::T1
    sets::T2
    wf::T3
    fb::T4
    strict::T5
    function EqualWeighted(wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                           wf::WeightFinaliser, fb::Option{<:OptE_Opt}, strict::Bool)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(wb), typeof(sets), typeof(wf), typeof(fb), typeof(strict)}(wb,
                                                                                     sets,
                                                                                     wf, fb,
                                                                                     strict)
    end
end
function EqualWeighted(; wb::Option{<:WbE_Wb} = WeightBounds(),
                       sets::Option{<:AssetSets} = nothing,
                       wf::WeightFinaliser = IterativeWeightFinaliser(),
                       fb::Option{<:OptE_Opt} = nothing, strict::Bool = false)
    return EqualWeighted(wb, sets, wf, fb, strict)
end
function factory(opt::EqualWeighted, w::AbstractVector)
    return EqualWeighted(; wb = opt.wb, sets = opt.sets, wf = opt.wf,
                         fb = factory(opt.fb, w), strict = opt.strict)
end
function opt_view(opt::EqualWeighted, i, args...)
    wb = weight_bounds_view(opt.wb, i)
    sets = nothing_asset_sets_view(opt.sets, i)
    return EqualWeighted(; wb = wb, sets = sets, wf = opt.wf, fb = opt.fb,
                         strict = opt.strict)
end
function _optimise(ew::EqualWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X))
    @argcheck(dims in (1, 2))
    dims = ifelse(isone(dims), 2, 1)
    N = size(rd.X, dims)
    w = fill(inv(N), N)
    wb = weight_bounds_constraints(ew.wb, ew.sets; N = N, strict = ew.strict,
                                   datatype = eltype(rd.X))
    retcode, w = finalise_weight_bounds(ew.wf, wb, w)
    return NaiveOptimisationResult(typeof(ew), nothing, wb, retcode, w, nothing)
end
function optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(ew, rd; dims = dims, kwargs...)
end
struct RandomWeighted{T1, T2, T3, T4, T5, T6, T7} <: NaiveOptimisationEstimator
    rng::T1
    seed::T2
    wb::T3
    sets::T4
    wf::T5
    fb::T6
    strict::T7
    function RandomWeighted(rng::Random.AbstractRNG, seed::Option{<:Integer},
                            wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                            wf::WeightFinaliser, fb::Option{<:OptE_Opt}, strict::Bool)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(rng), typeof(seed), typeof(wb), typeof(sets), typeof(wf),
                   typeof(fb), typeof(strict)}(rng, seed, wb, sets, wf, fb, strict)
    end
end
function RandomWeighted(; rng::Random.AbstractRNG = Random.default_rng(),
                        seed::Option{<:Integer} = nothing, wb::Option{<:WbE_Wb} = nothing,
                        sets::Option{<:AssetSets} = nothing,
                        wf::WeightFinaliser = IterativeWeightFinaliser(),
                        fb::Option{<:OptE_Opt} = nothing, strict::Bool = false)
    return RandomWeighted(rng, seed, wb, sets, wf, fb, strict)
end
function factory(opt::RandomWeighted, w::AbstractVector)
    return RandomWeighted(; rng = opt.rng, seed = opt.seed, wb = opt.wb, sets = opt.sets,
                          wf = opt.wf, fb = factory(opt.fb, w), strict = opt.strict)
end
function opt_view(opt::RandomWeighted, i, args...)
    wb = weight_bounds_view(opt.wb, i)
    sets = nothing_asset_sets_view(opt.sets, i)
    return RandomWeighted(; rng = opt.rng, seed = opt.seed, wb = wb, sets = sets,
                          wf = opt.wf, fb = opt.fb, strict = opt.strict)
end
function _optimise(rw::RandomWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X))
    @argcheck(dims in (1, 2))
    dims = ifelse(isone(dims), 2, 1)
    N = size(rd.X, dims)
    dist = Distributions.Dirichlet(size(rd.X, dims), 1)
    if !isnothing(rw.seed)
        Random.seed!(rw.rng, rw.seed)
    end
    w = rand(rw.rng, dist)
    wb = weight_bounds_constraints(rw.wb, rw.sets; N = N, strict = rw.strict,
                                   datatype = eltype(rd.X))
    retcode, w = finalise_weight_bounds(rw.wf, wb, w)
    return NaiveOptimisationResult(typeof(rw), nothing, wb, retcode, w, nothing)
end
function optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(rw, rd; dims = dims, kwargs...)
end

export NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeighted
