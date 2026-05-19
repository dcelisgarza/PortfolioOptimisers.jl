"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for naive (heuristic) portfolio optimisation estimators.

Naive optimisers compute portfolio weights directly from statistical properties of asset returns (e.g., volatility or equal weights) without solving an optimisation problem.

# Related Types

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`InverseVolatility`](@ref)
  - [`EqualWeighted`](@ref)
  - [`RandomWeighted`](@ref)
"""
abstract type NaiveOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
function needs_previous_weights(opt::NaiveOptimisationEstimator)
    return needs_previous_weights(opt.fb)
end
function assert_internal_optimiser(::NaiveOptimisationEstimator)::Nothing
    return nothing
end
function assert_external_optimiser(::NaiveOptimisationEstimator)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for naive portfolio optimisation estimators.

# Fields

  - `oe`: Type of the optimisation estimator that produced this result.
  - `pr`: Prior result used in optimisation (or `nothing`).
  - `wb`: Weight bounds applied.
  - `retcode`: Optimisation return code (`OptimisationSuccess` or `OptimisationFailure`).
  - `w`: Optimal portfolio weights vector.
  - `fb`: Fallback result (if a fallback optimiser was used).

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`InverseVolatility`](@ref)
  - [`EqualWeighted`](@ref)
  - [`RandomWeighted`](@ref)
"""
@concrete struct NaiveOptimisationResult <: NonFiniteAllocationOptimisationResult
    oe
    pr
    wb
    retcode
    w
    fb
end
function factory(res::NaiveOptimisationResult, fb::Option{<:OptE_Opt})
    return NaiveOptimisationResult(res.oe, res.pr, res.wb, res.retcode, res.w, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Inverse Volatility portfolio optimiser.

`InverseVolatility` allocates portfolio weights inversely proportional to each asset's volatility (standard deviation). Optionally, `sq = true` uses variance instead.

# Mathematical Definition

```math
w_i = \\frac{1 / \\sigma_i}{\\sum_{j=1}^N 1 / \\sigma_j}\\,,
```

where ``\\sigma_i`` is the standard deviation (or variance when `sq = true`) of asset ``i``.

# Fields

  - `pe`: Prior estimator or prior result for computing asset covariance.
  - `wb`: Weight bounds estimator or bounds.
  - `sets`: Asset sets (required when `wb` is a `WeightBoundsEstimator`).
  - `wf`: Weight finaliser for enforcing bounds.
  - `fb`: Fallback optimiser (used if this optimiser fails).
  - `sq`: If `true`, weights are inversely proportional to variance rather than volatility.
  - `brt`: If `true`, uses bootstrap returns instead of the original returns.
  - `strict`: If `true`, strictly enforces weight bounds.

# Constructors

    InverseVolatility(;
        pe::PrE_Pr = EmpiricalPrior(),
        wb::Option{<:WbE_Wb} = WeightBounds(),
        sets::Option{<:AssetSets} = nothing,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        fb::Option{<:OptE_Opt} = nothing,
        sq::Bool = false,
        brt::Bool = false,
        strict::Bool = false
    ) -> InverseVolatility

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> InverseVolatility()
InverseVolatility
      pe ┼ EmpiricalPrior
         │        ce ┼ PortfolioOptimisersCovariance
         │           │   ce ┼ Covariance
         │           │      │    me ┼ SimpleExpectedReturns
         │           │      │       │   w ┴ nothing
         │           │      │    ce ┼ GeneralCovariance
         │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │           │      │       │    w ┴ nothing
         │           │      │   alg ┴ Full()
         │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
         │           │      │     pdm ┼ Posdef
         │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │           │      │      dn ┼ nothing
         │           │      │      dt ┼ nothing
         │           │      │     alg ┼ nothing
         │           │      │   order ┴ DenoiseDetoneAlg()
         │        me ┼ SimpleExpectedReturns
         │           │   w ┴ nothing
         │   horizon ┴ nothing
      wb ┼ WeightBounds
         │   lb ┼ Float64: 0.0
         │   ub ┴ Float64: 1.0
    sets ┼ nothing
      wf ┼ IterativeWeightFinaliser
         │   iter ┴ Int64: 100
      fb ┼ nothing
      sq ┼ Bool: false
     brt ┼ Bool: false
  strict ┴ Bool: false
```

# Related

  - [`NaiveOptimisationEstimator`](@ref)
  - [`EqualWeighted`](@ref)
  - [`RandomWeighted`](@ref)
"""
@concrete struct InverseVolatility <: NaiveOptimisationEstimator
    pe
    wb
    sets
    wf
    fb
    sq
    brt
    strict
    function InverseVolatility(pe::PrE_Pr, wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                               wf::WeightFinaliser, fb::Option{<:OptE_Opt}, sq::Bool,
                               brt::Bool, strict::Bool)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pe), typeof(wb), typeof(sets), typeof(wf), typeof(fb), typeof(sq),
                   typeof(brt), typeof(strict)}(pe, wb, sets, wf, fb, sq, brt, strict)
    end
end
function InverseVolatility(; pe::PrE_Pr = EmpiricalPrior(),
                           wb::Option{<:WbE_Wb} = WeightBounds(),
                           sets::Option{<:AssetSets} = nothing,
                           wf::WeightFinaliser = IterativeWeightFinaliser(),
                           fb::Option{<:OptE_Opt} = nothing, sq::Bool = false,
                           brt::Bool = false, strict::Bool = false)
    return InverseVolatility(pe, wb, sets, wf, fb, sq, brt, strict)
end
function factory(opt::InverseVolatility, w::AbstractVector)
    return InverseVolatility(; pe = opt.pe, wb = opt.wb, sets = opt.sets, wf = opt.wf,
                             fb = factory(opt.fb, w), sq = opt.sq, brt = opt.brt,
                             strict = opt.strict)
end
function opt_view(opt::InverseVolatility, i, args...)
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    sets = asset_sets_view(opt.sets, i)
    return InverseVolatility(; pe = pe, wb = wb, sets = sets, wf = opt.wf, fb = opt.fb,
                             sq = opt.sq, brt = opt.brt, strict = opt.strict)
end
function assert_external_optimiser(opt::InverseVolatility)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
function _optimise(iv::InverseVolatility, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    rd = returns_result_picker(rd, iv.brt)
    pr = prior(iv.pe, rd; dims = dims)
    X = pr.X
    w = LinearAlgebra.diag(pr.sigma)
    w = inv.(!iv.sq ? sqrt.(w) : w)
    w /= sum(w)
    wb = weight_bounds_constraints(iv.wb, iv.sets; N = size(X, ifelse(isone(dims), 2, 1)),
                                   strict = iv.strict, datatype = eltype(X))
    retcode, w = finalise_weight_bounds(iv.wf, wb, w)
    return NaiveOptimisationResult(typeof(iv), pr, wb, retcode, w, nothing)
end
"""
    optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...) -> NaiveOptimisationResult

# Arguments

  - `iv`: The inverse volatility optimiser to use.
  - $(arg_dict[:rd]) If `isa(iv.pe, AbstractPriorResult)`, `rd` is not necessary.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(iv, rd; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Equal-weighted portfolio optimiser.

`EqualWeighted` allocates equal weight to all ``N`` assets in the portfolio:

```math
w_i = \\frac{1}{N} \\quad \\forall i\\,.
```

# Fields

  - `wb`: Weight bounds estimator or bounds.
  - `sets`: Asset sets (required when `wb` is a `WeightBoundsEstimator`).
  - `wf`: Weight finaliser for enforcing bounds.
  - `fb`: Fallback optimiser.
  - `strict`: If `true`, strictly enforces weight bounds.

# Constructors

    EqualWeighted(;
        wb::Option{<:WbE_Wb} = WeightBounds(),
        sets::Option{<:AssetSets} = nothing,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        fb::Option{<:OptE_Opt} = nothing,
        strict::Bool = false
    ) -> EqualWeighted

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> EqualWeighted()
EqualWeighted
      wb ┼ WeightBounds
         │   lb ┼ Float64: 0.0
         │   ub ┴ Float64: 1.0
    sets ┼ nothing
      wf ┼ IterativeWeightFinaliser
         │   iter ┴ Int64: 100
      fb ┼ nothing
  strict ┴ Bool: false
```

# Related

  - [`NaiveOptimisationEstimator`](@ref)
  - [`InverseVolatility`](@ref)
  - [`RandomWeighted`](@ref)
"""
@concrete struct EqualWeighted <: NaiveOptimisationEstimator
    wb
    sets
    wf
    fb
    strict
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
    sets = asset_sets_view(opt.sets, i)
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
"""
    optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; dims::Int = 1, kwargs...) -> NaiveOptimisationResult

# Arguments

  - `ew`: The equal-weighted optimiser to use.
  - $(arg_dict[:rd]) Used to know how many assets there are.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  dims::Int = 1, kwargs...)
    return _optimise(ew, rd; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Random-weighted portfolio optimiser.

`RandomWeighted` draws portfolio weights at random from a Dirichlet distribution with concentration parameter `alpha`. This can be used for simulation, benchmarking, or stress-testing.

# Mathematical Definition

```math
\\boldsymbol{w} \\sim \\mathrm{Dirichlet}(\\boldsymbol{\\alpha})\\,,
```

where ``\\boldsymbol{\\alpha}`` is a scalar or vector concentration parameter. Larger values of ``\\alpha`` concentrate the distribution near equal weights.

# Fields

  - `alpha`: Dirichlet concentration parameter (scalar or vector, all positive).
  - `rng`: Random number generator.
  - `seed`: Optional seed for the RNG.
  - `wb`: Weight bounds estimator or bounds.
  - `sets`: Asset sets.
  - `wf`: Weight finaliser for enforcing bounds.
  - `fb`: Fallback optimiser.
  - `strict`: If `true`, strictly enforces weight bounds.

# Constructors

    RandomWeighted(;
        alpha::Num_VecNum = 1,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        wb::Option{<:WbE_Wb} = nothing,
        sets::Option{<:AssetSets} = nothing,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        fb::Option{<:OptE_Opt} = nothing,
        strict::Bool = false
    ) -> RandomWeighted

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is provided: all elements positive and finite.

# Examples

```jldoctest
julia> RandomWeighted()
RandomWeighted
   alpha ┼ Int64: 1
     rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
    seed ┼ nothing
      wb ┼ nothing
    sets ┼ nothing
      wf ┼ IterativeWeightFinaliser
         │   iter ┴ Int64: 100
      fb ┼ nothing
  strict ┴ Bool: false
```

# Related

  - [`NaiveOptimisationEstimator`](@ref)
  - [`InverseVolatility`](@ref)
  - [`EqualWeighted`](@ref)
"""
@concrete struct RandomWeighted <: NaiveOptimisationEstimator
    alpha
    rng
    seed
    wb
    sets
    wf
    fb
    strict
    function RandomWeighted(alpha::Num_VecNum, rng::Random.AbstractRNG,
                            seed::Option{<:Integer}, wb::Option{<:WbE_Wb},
                            sets::Option{<:AssetSets}, wf::WeightFinaliser,
                            fb::Option{<:OptE_Opt}, strict::Bool)
        if !isnothing(alpha)
            assert_nonempty_gt0_finite_val(alpha, :alpha)
        end
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(alpha), typeof(rng), typeof(seed), typeof(wb), typeof(sets),
                   typeof(wf), typeof(fb), typeof(strict)}(alpha, rng, seed, wb, sets, wf,
                                                           fb, strict)
    end
end
function RandomWeighted(; alpha::Num_VecNum = 1,
                        rng::Random.AbstractRNG = Random.default_rng(),
                        seed::Option{<:Integer} = nothing, wb::Option{<:WbE_Wb} = nothing,
                        sets::Option{<:AssetSets} = nothing,
                        wf::WeightFinaliser = IterativeWeightFinaliser(),
                        fb::Option{<:OptE_Opt} = nothing, strict::Bool = false)
    return RandomWeighted(alpha, rng, seed, wb, sets, wf, fb, strict)
end
function factory(opt::RandomWeighted, w::AbstractVector)
    return RandomWeighted(; alpha = opt.alpha, rng = opt.rng, seed = opt.seed, wb = opt.wb,
                          sets = opt.sets, wf = opt.wf, fb = factory(opt.fb, w),
                          strict = opt.strict)
end
function opt_view(opt::RandomWeighted, i, args...)
    wb = weight_bounds_view(opt.wb, i)
    sets = asset_sets_view(opt.sets, i)
    alpha = nothing_scalar_array_view(opt.alpha, i)
    return RandomWeighted(; alpha = alpha, rng = opt.rng, seed = opt.seed, wb = wb,
                          sets = sets, wf = opt.wf, fb = opt.fb, strict = opt.strict)
end
function _optimise(rw::RandomWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X))
    @argcheck(dims in (1, 2))
    dims = ifelse(isone(dims), 2, 1)
    N = size(rd.X, dims)
    if isa(rw.alpha, VecNum)
        @argcheck(length(rw.alpha) == N)
    end
    dist = if isa(rw.alpha, Number)
        Distributions.Dirichlet(N, rw.alpha)
    else
        Distributions.Dirichlet(rw.alpha)
    end
    if !isnothing(rw.seed)
        Random.seed!(rw.rng, rw.seed)
    end
    w = rand(rw.rng, dist)
    wb = weight_bounds_constraints(rw.wb, rw.sets; N = N, strict = rw.strict,
                                   datatype = eltype(rd.X))
    retcode, w = finalise_weight_bounds(rw.wf, wb, w)
    return NaiveOptimisationResult(typeof(rw), nothing, wb, retcode, w, nothing)
end
"""
    optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; dims::Int = 1, kwargs...) -> NaiveOptimisationResult

# Arguments

  - `rw`: The random-weighted optimiser to use.
  - $(arg_dict[:rd]) Used to know how many assets there are.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)
    return _optimise(rw, rd; dims = dims, kwargs...)
end

export NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeighted
