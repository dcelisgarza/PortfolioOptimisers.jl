"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for naive (heuristic) portfolio optimisation estimators.

Naive optimisers compute portfolio weights directly from statistical properties of asset returns (e.g., volatility or equal weights) without solving an optimisation problem.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`InverseVolatility`](@ref)
  - [`EqualWeighted`](@ref)
  - [`RandomWeighted`](@ref)
"""
abstract type NaiveOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the naive optimiser's fallback estimator requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
"""
function needs_previous_weights(opt::NaiveOptimisationEstimator)
    return needs_previous_weights(opt.fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert internal validity for a naive optimisation estimator. No-op default.

# Related

  - [`assert_internal_optimiser`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
"""
function assert_internal_optimiser(::NaiveOptimisationEstimator)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert external validity for a naive optimisation estimator. No-op default.

# Related

  - [`assert_external_optimiser`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
"""
function assert_external_optimiser(::NaiveOptimisationEstimator)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for naive portfolio optimisation estimators.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NaiveOptimisationResult(oe, pr, wb, retcode, w, fb) -> NaiveOptimisationResult

Positional arguments correspond to the struct's fields.

# Examples

```jldoctest
julia> NaiveOptimisationResult(InverseVolatility, nothing, nothing, OptimisationSuccess(),
                               [0.5, 0.5], nothing)
NaiveOptimisationResult
       oe ┼ UnionAll: InverseVolatility
       pr ┼ nothing
       wb ┼ nothing
  retcode ┼ OptimisationSuccess
          │   res ┴ nothing
        w ┼ Vector{Float64}: [0.5, 0.5]
       fb ┴ nothing
```

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`InverseVolatility`](@ref)
  - [`EqualWeighted`](@ref)
  - [`RandomWeighted`](@ref)
"""
@concrete struct NaiveOptimisationResult <: NonJuMPOptimisationResult
    """
    $(field_dict[:oe])
    """
    oe
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    $(field_dict[:pw])
    """
    w
    """
    $(field_dict[:fb])
    """
    fb
end
"""
$(DocStringExtensions.TYPEDEF)

Inverse Volatility portfolio optimiser.

`InverseVolatility` allocates portfolio weights inversely proportional to each asset's volatility (standard deviation). Optionally, `sq = true` uses variance instead.

# Mathematical definition

```math
\\begin{align}
w_i &= \\frac{1 / \\sigma_i}{\\sum_{j=1}^N 1 / \\sigma_j}\\,,\\,.
\\end{align}
```

Where:

  - ``w_i``: Portfolio weight of asset ``i``.
  - ``\\sigma_i``: Standard deviation of asset ``i`` (variance when `sq = true`).
  - ``N``: Number of assets.

# Fields

$(DocStringExtensions.FIELDS)

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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

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
         │           │   mp ┼ MatrixProcessing
         │           │      │     pdm ┼ Posdef
         │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │           │      │      dn ┼ nothing
         │           │      │      dt ┼ nothing
         │           │      │     alg ┼ nothing
         │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct InverseVolatility <: NaiveOptimisationEstimator
    """
    $(field_dict[:pe])
    """
    @vprop pe
    """
    $(field_dict[:wb])
    """
    @vprop wb
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:fb])
    """
    @fprop fb
    """
    $(field_dict[:sq])
    """
    sq
    """
    $(field_dict[:brt])
    """
    brt
    """
    $(field_dict[:strict_opt])
    """
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
                           brt::Bool = false, strict::Bool = false)::InverseVolatility
    return InverseVolatility(pe, wb, sets, wf, fb, sq, brt, strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that [`InverseVolatility`](@ref) is valid for external use.

Requires that `opt.pe` is not an `AbstractPriorResult`.

# Related

  - [`InverseVolatility`](@ref)
  - [`assert_external_optimiser`](@ref)
"""
function assert_external_optimiser(opt::InverseVolatility)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the inverse volatility portfolio optimisation.

Internal dispatch called by [`optimise`](@ref). Computes covariance via the prior estimator, assigns weights inversely proportional to volatility (or variance when `iv.sq = true`), then applies weight bounds.

# Related

  - [`InverseVolatility`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
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

Run the inverse volatility portfolio optimisation.

# Arguments

  - `iv`: The inverse volatility optimiser to use.
  - $(arg_dict[:rd]) If `isa(iv.pe, AbstractPriorResult)`, `rd` is not necessary.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  kwargs...)::NaiveOptimisationResult
    return _optimise(iv, rd; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Equal-weighted portfolio optimiser.

`EqualWeighted` allocates equal weight to all ``N`` assets in the portfolio.

# Mathematical definition

```math
\\begin{align}
w_i &= \\frac{1}{N} \\quad \\forall i\\,.
\\end{align}
```

Where:

  - ``w_i``: Portfolio weight of asset ``i``.
  - ``N``: Number of assets.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EqualWeighted(;
        wb::Option{<:WbE_Wb} = WeightBounds(),
        sets::Option{<:AssetSets} = nothing,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        fb::Option{<:OptE_Opt} = nothing,
        strict::Bool = false
    ) -> EqualWeighted

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct EqualWeighted <: NaiveOptimisationEstimator
    """
    $(field_dict[:wb])
    """
    @vprop wb
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:fb])
    """
    @fprop fb
    """
    $(field_dict[:strict_opt])
    """
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
                       fb::Option{<:OptE_Opt} = nothing,
                       strict::Bool = false)::EqualWeighted
    return EqualWeighted(wb, sets, wf, fb, strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the equal-weighted portfolio optimisation.

Internal dispatch called by [`optimise`](@ref). Assigns equal weights to all assets, then applies weight bounds.

# Related

  - [`EqualWeighted`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
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

Run the equal-weighted portfolio optimisation.

# Arguments

  - `ew`: The equal-weighted optimiser to use.
  - $(arg_dict[:rd]) Used to know how many assets there are.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  dims::Int = 1, kwargs...)::NaiveOptimisationResult
    return _optimise(ew, rd; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Random-weighted portfolio optimiser.

`RandomWeighted` draws portfolio weights at random from a Dirichlet distribution with concentration parameter `alpha`. This can be used for simulation, benchmarking, or stress-testing.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w} \\sim \\mathrm{Dirichlet}(\\boldsymbol{\\alpha})\\,,\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: Portfolio weight vector.
  - ``\\boldsymbol{\\alpha}``: Scalar or vector concentration parameter. Larger values concentrate the distribution near equal weights.

# Fields

$(DocStringExtensions.FIELDS)

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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `alpha`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct RandomWeighted <: NaiveOptimisationEstimator
    """
    $(field_dict[:alpha_dirichlet])
    """
    @vprop alpha
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
    """
    $(field_dict[:wb])
    """
    @vprop wb
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:fb])
    """
    @fprop fb
    """
    $(field_dict[:strict_opt])
    """
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
                        fb::Option{<:OptE_Opt} = nothing,
                        strict::Bool = false)::RandomWeighted
    return RandomWeighted(alpha, rng, seed, wb, sets, wf, fb, strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the random-weighted portfolio optimisation.

Internal dispatch called by [`optimise`](@ref). Draws weights from a Dirichlet distribution parameterised by `rw.alpha`, then applies weight bounds.

# Related

  - [`RandomWeighted`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
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

Run the random-weighted portfolio optimisation.

# Arguments

  - `rw`: The random-weighted optimiser to use.
  - $(arg_dict[:rd]) Used to know how many assets there are.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)::NaiveOptimisationResult
    return _optimise(rw, rd; dims = dims, kwargs...)
end

export NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeighted
