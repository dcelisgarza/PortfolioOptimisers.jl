abstract type NaiveOptimisationEstimator <: OptimisationEstimator end
function assert_internal_optimiser(::NaiveOptimisationEstimator)
    return nothing
end
function assert_external_optimiser(::NaiveOptimisationEstimator)
    return nothing
end
"""
"""
struct NaiveOptimisationResult{T1, T2, T3, T4, T5, T6} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T4
    retcode::T5
    w::T3
    fb::T6
end
function factory(res::NaiveOptimisationResult, fb)
    return NaiveOptimisationResult(res.oe, res.pr, res.wb, res.retcode, res.w, fb)
end
"""
"""
struct InverseVolatility{T1, T2, T3, T4, T5, T6} <: NaiveOptimisationEstimator
    pe::T1
    wb::T2
    sets::T3
    wf::T4
    strict::T5
    fb::T6
    function InverseVolatility(pe::PrE_Pr, wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                               wf::WeightFinaliser, strict::Bool,
                               fb::Option{<:OptimisationEstimator})
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pe), typeof(wb), typeof(sets), typeof(wf), typeof(strict),
                   typeof(fb)}(pe, wb, sets, wf, strict, fb)
    end
end
function InverseVolatility(; pe::PrE_Pr = EmpiricalPrior(),
                           wb::Option{<:WbE_Wb} = WeightBounds(),
                           sets::Option{<:AssetSets} = nothing,
                           wf::WeightFinaliser = IterativeWeightFinaliser(),
                           strict::Bool = false,
                           fb::Option{<:OptimisationEstimator} = nothing)
    return InverseVolatility(pe, wb, sets, wf, strict, fb)
end
function opt_view(opt::InverseVolatility, i, args...)
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    sets = nothing_asset_sets_view(opt.sets, i)
    return InverseVolatility(; pe = pe, wb = wb, sets = sets, wf = opt.wf,
                             strict = opt.strict, fb = opt.fb)
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
    pr = prior(iv.pe, rd; dims = dims)
    w = inv.(sqrt.(LinearAlgebra.diag(pr.sigma)))
    w /= sum(w)
    wb = weight_bounds_constraints(iv.wb, iv.sets;
                                   N = size(pr.X, ifelse(isone(dims), 2, 1)),
                                   strict = iv.strict, datatype = eltype(pr.X))
    retcode, w = finalise_weight_bounds(iv.wf, wb, w)
    return NaiveOptimisationResult(typeof(iv), pr, wb, retcode, w, nothing)
end
function optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(iv, rd; dims = dims, kwargs...)
end
"""
"""
struct EqualWeighted{T1, T2, T3, T4, T5} <: NaiveOptimisationEstimator
    wb::T1
    sets::T2
    wf::T3
    strict::T4
    fb::T5
    function EqualWeighted(wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                           wf::WeightFinaliser, strict::Bool,
                           fb::Option{<:OptimisationEstimator})
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(wb), typeof(sets), typeof(wf), typeof(strict), typeof(fb)}(wb,
                                                                                     sets,
                                                                                     wf,
                                                                                     strict,
                                                                                     fb)
    end
end
function EqualWeighted(; wb::Option{<:WbE_Wb} = WeightBounds(),
                       sets::Option{<:AssetSets} = nothing,
                       wf::WeightFinaliser = IterativeWeightFinaliser(),
                       strict::Bool = false, fb::Option{<:OptimisationEstimator} = nothing)
    return EqualWeighted(wb, sets, wf, strict, fb)
end
function opt_view(opt::EqualWeighted, i, args...)
    wb = weight_bounds_view(opt.wb, i)
    sets = nothing_asset_sets_view(opt.sets, i)
    return EqualWeighted(; wb = wb, sets = sets, wf = opt.wf, strict = opt.strict,
                         fb = opt.fb)
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
function optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(ew, rd; dims = dims, kwargs...)
end
"""
"""
struct RandomWeighted{T1, T2, T3, T4, T5, T6, T7} <: NaiveOptimisationEstimator
    rng::T1
    seed::T2
    wb::T3
    sets::T4
    wf::T5
    strict::T6
    fb::T7
    function RandomWeighted(rng::Random.AbstractRNG, seed::Option{<:Integer},
                            wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                            wf::WeightFinaliser, strict::Bool,
                            fb::Option{<:OptimisationEstimator})
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(rng), typeof(seed), typeof(wb), typeof(sets), typeof(wf),
                   typeof(strict), typeof(fb)}(rng, seed, wb, sets, wf, strict, fb)
    end
end
function RandomWeighted(; rng::Random.AbstractRNG = Random.default_rng(),
                        seed::Option{<:Integer} = nothing, wb::Option{<:WbE_Wb} = nothing,
                        sets::Option{<:AssetSets} = nothing,
                        wf::WeightFinaliser = IterativeWeightFinaliser(),
                        strict::Bool = false, fb::Option{<:OptimisationEstimator} = nothing)
    return RandomWeighted(rng, seed, wb, sets, wf, strict, fb)
end
function opt_view(opt::RandomWeighted, i, args...)
    wb = weight_bounds_view(opt.wb, i)
    sets = nothing_asset_sets_view(opt.sets, i)
    return RandomWeighted(; rng = opt.rng, seed = opt.seed, wb = wb, sets = sets,
                          wf = opt.wf, strict = opt.strict, fb = opt.fb)
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
function optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(rw, rd; dims = dims, kwargs...)
end

export NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeighted
