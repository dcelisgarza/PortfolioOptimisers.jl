"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for naive (heuristic) portfolio optimisation estimators.

Naive optimisers compute portfolio weights directly from statistical properties of asset returns (e.g., volatility or equal weights) without solving an optimisation problem.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`InverseVolatility`](@ref)
  - [`EqualWeighted`](@ref)
  - [`RandomWeighted`](@ref)
  - [`PreviousWeights`](@ref)
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
    return any(f -> needs_previous_weights(getfield(opt, f)), time_dependent_fields(opt)) ||
           needs_previous_weights(opt.fb)
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
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if the naive optimiser configuration carries time-dependent constraints.

# Related

  - [`NaiveOptimisationEstimator`](@ref)
  - [`TimeDependent`](@ref)
  - [`is_time_dependent`](@ref)
"""
function is_time_dependent(opt::NaiveOptimisationEstimator)
    return !isempty(time_dependent_fields(opt)) || is_time_dependent(opt.fb)
end
function assert_time_dependent_fold_count(opt::NaiveOptimisationEstimator, n::Integer,
                                          all_binds::Bool = true)::Nothing
    assert_time_dependent_fields_fold_count(opt, n, all_binds)
    assert_time_dependent_fold_count(opt.fb, n, all_binds)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the time-dependent constraints of a naive optimiser for the fold described by `ctx`.

Rebuilds the optimiser through its validated keyword constructor with each [`TimeDependent`](@ref)-valued field replaced by its resolved per-fold value, recursing into the fallback, so the result is an ordinary static optimiser.

# Related

  - [`NaiveOptimisationEstimator`](@ref)
  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
"""
function update_time_dependent_estimator(opt::NaiveOptimisationEstimator,
                                         ctx::TimeDependentContext, all_binds::Bool = true)
    if !is_time_dependent(opt)
        return opt
    end
    opt = update_time_dependent_fields(opt, ctx, all_binds)
    return rebuild_estimator(opt,
                             (;
                              fb = update_time_dependent_estimator(opt.fb, ctx, all_binds)))
end
function time_dependent_field_defaults(::NaiveOptimisationEstimator)::NamedTuple
    return naive_optimiser_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the naive-optimiser fields that may hold a [`TimeDependent`](@ref).

Shared by the constructors' test-substitution passes and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted; [`RandomWeighted`](@ref) overrides the trait because its `wb` default is `nothing`, and [`InverseVolatility`](@ref) extends it with its prior estimator.

# Related

  - [`NaiveOptimisationEstimator`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function naive_optimiser_td_defaults()::NamedTuple
    return (; wb = WeightBounds(), wf = IterativeWeightFinaliser())
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace time-dependent constraints with their static defaults, both on the naive optimiser's own fields and by recursing into the fallback.
"""
function reset_time_dependent_estimator(opt::NaiveOptimisationEstimator)
    if !is_time_dependent(opt)
        return opt
    end
    opt = reset_time_dependent_fields(opt)
    return rebuild_estimator(opt, (; fb = reset_time_dependent_estimator(opt.fb)))
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for naive portfolio optimisation estimators.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NaiveOptimisationResult(;
        pr::Option{<:Pr_RR},
        wb::Option{<:WeightBounds}, retcode::OptimisationReturnCode, w::Option{<:VecNum},
        imsk::Option{<:BitVector} = nothing, fb::Option{<:OptE_Opt_FbChain}
    ) -> NaiveOptimisationResult

Keywords correspond to the struct's fields. The keyword constructor expands `w` onto the full asset universe through [`expand_investable_weights`](@ref), which is the one door [`_optimise`](@ref) exits through. The positional constructor never expands.

# Examples

```jldoctest
julia> NaiveOptimisationResult(; pr = nothing, wb = nothing, retcode = OptimisationSuccess(),
                               w = [0.5, 0.5], fb = nothing)
NaiveOptimisationResult
       pr ┼ nothing
       wb ┼ nothing
  retcode ┼ OptimisationSuccess
          │   res ┴ nothing
        w ┼ Vector{Float64}: [0.5, 0.5]
     imsk ┼ nothing
       fb ┴ nothing
```

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`InverseVolatility`](@ref)
  - [`EqualWeighted`](@ref)
  - [`RandomWeighted`](@ref)
  - [`expand_investable_weights`](@ref)
"""
@concrete struct NaiveOptimisationResult <: NonJuMPOptimisationResult
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
    $(field_dict[:imsk])
    """
    imsk
    """
    $(field_dict[:fb_res])
    """
    fb
    function NaiveOptimisationResult(pr::Option{<:Pr_RR}, wb::Option{<:WeightBounds},
                                     retcode::OptimisationReturnCode, w::Option{<:VecNum},
                                     imsk::Option{<:BitVector},
                                     fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(pr), typeof(wb), typeof(retcode), typeof(w), typeof(imsk),
                   typeof(fb)}(pr, wb, retcode, w, imsk, fb)
    end
end
function NaiveOptimisationResult(; pr::Option{<:Pr_RR}, wb::Option{<:WeightBounds},
                                 retcode::OptimisationReturnCode, w::Option{<:VecNum},
                                 imsk::Option{<:BitVector} = nothing,
                                 fb::Option{<:OptE_Opt_FbChain})::NaiveOptimisationResult
    return NaiveOptimisationResult(pr, wb, retcode, expand_investable_weights(imsk, w),
                                   imsk, fb)
end
# The naive family carries the mask on the result itself, so the fold reads it directly.
function result_investable_mask(res::NaiveOptimisationResult)
    return res.imsk
end
"""
$(DocStringExtensions.TYPEDEF)

Allocates each asset a weight inversely proportional to its volatility, or to its variance when `sq = true`.

The volatilities come from the diagonal of the covariance matrix the prior estimator `pe` returns, so every choice inside `pe` reaches the result. This is the naive risk parity allocation that [`HierarchicalRiskParity`](@ref) applies inside a cluster.

# Mathematical definition

```math
\\begin{align}
w_i &= \\frac{\\sigma_i^{-1}}{\\sum_{j=1}^N \\sigma_j^{-1}} \\quad \\textrm{when } \\texttt{sq = false}\\,,\\\\
w_i &= \\frac{\\sigma_i^{-2}}{\\sum_{j=1}^N \\sigma_j^{-2}} \\quad \\textrm{when } \\texttt{sq = true}\\,.
\\end{align}
```

Where:

  - ``w_i``: Portfolio weight of asset ``i`` before the weight bounds are applied.
  - ``\\sigma_i^2``: Variance of asset ``i``, the ``i``-th diagonal entry of the prior covariance matrix.
  - ``\\sigma_i``: Standard deviation of asset ``i``.
  - ``N``: Number of assets.

The weight finaliser `wf` then imposes the resolved weight bounds on ``\\boldsymbol{w}``, so the returned weights equal the expression above only when no bound binds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    InverseVolatility(;
        pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        sets::TD_Option{<:UniverseSets} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        sq::Bool = false,
        brt::Bool = false,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> InverseVolatility

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the prior estimator, weight bounds, asset sets, weight finaliser and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default. `sq`, `brt` and `strict` are execution control and stay static.

## Validation

  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.
  - `fb` schedules: `bind !== :nearest`.

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
         │           ce ┼ PortfolioOptimisersCovariance
         │              │   ce ┼ Covariance
         │              │      │    me ┼ SimpleExpectedReturns
         │              │      │       │   w ┴ nothing
         │              │      │    ce ┼ GeneralCovariance
         │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │              │      │       │    w ┴ nothing
         │              │      │   alg ┼ FullMoment()
         │              │      │     w ┴ nothing
         │              │   mp ┼ MatrixProcessing
         │              │      │     pdm ┼ Posdef
         │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │              │      │      dn ┼ nothing
         │              │      │      dt ┼ nothing
         │              │      │     alg ┼ nothing
         │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │           me ┼ SimpleExpectedReturns
         │              │   w ┴ nothing
         │      horizon ┼ nothing
         │   fill_limit ┴ nothing
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

  - [`optimise`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
  - [`EqualWeighted`](@ref)
  - [`RandomWeighted`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 12.1.3, footnote 5.
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
    """
    $(field_dict[:cache_opt])
    """
    @fprop @vprop cache
    function InverseVolatility(pe::Onl{<:TD{<:PrE_Pr}}, wb::TD_Option{<:WbE_Wb},
                               sets::TD_Option{<:UniverseSets}, wf::TD{<:WeightFinaliser},
                               fb::TDO_Option{<:OptE_Opt}, sq::Bool, brt::Bool,
                               strict::Bool, cache::Option{<:ReturnsBufferState})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :InverseVolatility)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb is a WeightBoundsEstimator"))
        end
        assert_time_dependent_substitution(InverseVolatility,
                                           (; pe, wb, sets, wf, fb, sq, brt, strict),
                                           merge(naive_optimiser_td_defaults(),
                                                 (; pe = EmpiricalPrior())))
        return new{typeof(pe), typeof(wb), typeof(sets), typeof(wf), typeof(fb), typeof(sq),
                   typeof(brt), typeof(strict), typeof(cache)}(pe, wb, sets, wf, fb, sq,
                                                               brt, strict, cache)
    end
end
function InverseVolatility(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
                           wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                           sets::TD_Option{<:UniverseSets} = nothing,
                           wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                           fb::TDO_Option{<:OptE_Opt} = nothing, sq::Bool = false,
                           brt::Bool = false, strict::Bool = false,
                           cache::Option{<:ReturnsBufferState} = nothing)::InverseVolatility
    return InverseVolatility(pe, wb, sets, wf, fb, sq, brt, strict, cache)
end
function non_investable_universe(iv::InverseVolatility, ni::VecStr)::InverseVolatility
    return rebuild_estimator(iv, (; sets = non_investable_sets(iv.sets, ni)))
end
function time_dependent_field_defaults(::InverseVolatility)::NamedTuple
    return merge(naive_optimiser_td_defaults(), (; pe = EmpiricalPrior()))
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
    @argcheck(!isa(opt.pe, AbstractPriorResult),
              ArgumentError("opt.pe must not be an AbstractPriorResult for external use, got $(typeof(opt.pe))"))
    assert_internal_optimiser(opt)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the inverse volatility portfolio optimisation.

Internal dispatch called by [`optimise`](@ref). Computes covariance via the prior estimator, reduces the prior, the optimiser and the returns data to the Investable Mask with [`investable_reduction`](@ref), assigns weights inversely proportional to volatility (or variance when `iv.sq = true`), then applies weight bounds. [`NaiveOptimisationResult`](@ref) expands the weights back onto the full asset universe.

# Related

  - [`InverseVolatility`](@ref)
  - [`investable_reduction`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(iv::InverseVolatility, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, kwargs...)
    assert_dims(dims)
    iv = reset_time_dependent_estimator(iv)
    rd = returns_result_picker(rd, iv.brt)
    pr = prior(iv.pe, rd; dims = dims)
    # The prior fits on the coverage universe and returns a result on the full asset
    # universe, where an asset it could not estimate carries `NaN`. Reduce once, here: the
    # diagonal below then holds a finite variance at every position, and the bounds are
    # stated over the investable assets alone. `NaiveOptimisationResult` expands the
    # weights back.
    imsk, pr, iv, rd = investable_reduction(pr, iv, rd)
    X = pr.X
    w = LinearAlgebra.diag(pr.sigma)
    w = inv.(!iv.sq ? sqrt.(w) : w)
    w /= sum(w)
    # `pr.X` is always observations by assets, whatever `dims` the caller passed, so the
    # asset count is `size(X, 2)` unconditionally.
    wb = weight_bounds_constraints(iv.wb, iv.sets; N = size(X, 2), strict = iv.strict,
                                   datatype = eltype(X))
    retcode, w = finalise_weight_bounds(iv.wf, wb, w)
    return NaiveOptimisationResult(; pr = pr, wb = wb, retcode = retcode, w = w,
                                   imsk = imsk, fb = nothing)
end
"""
    optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; dims::Int = 1, kwargs...) -> NaiveOptimisationResult

Run the inverse volatility portfolio optimisation.

# Arguments

  - `iv`: The inverse volatility optimiser to use.
  - $(arg_dict[:rd]) If `isa(iv.pe, AbstractPriorResult)`, `rd` is not necessary.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Validation

  - No field in the tree of `iv` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise, through [`assert_batch_entry`](@ref): a plain `optimise` is a batch fit, and a wrapper resolves only at the warm-up of the fold loop's online arm.
"""
function optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)::NaiveOptimisationResult
    assert_batch_entry(iv, "`optimise`")
    return _optimise(iv, rd; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Allocates the same weight to every asset in the universe.

The asset count comes from `rd.X`, so this optimiser reads no prior and no covariance matrix.

``N`` is the size of the **Coverage Universe** of the window, not of the full asset universe. An asset is in it when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window, so an asset that is not yet listed, is delisted, or carries a stale finite price during an inactive spell, weights nothing and holds a zero. The result carries that universe as its `imsk`. An all-dead window throws an `IsEmptyError`.

# Mathematical definition

```math
\\begin{align}
w_i &= \\frac{1}{N} \\quad \\forall i\\,.
\\end{align}
```

Where:

  - ``w_i``: Portfolio weight of asset ``i`` before the weight bounds are applied.
  - ``N``: Number of assets.

The weight finaliser `wf` then imposes the resolved weight bounds on ``\\boldsymbol{w}``, so the returned weights equal ``1/N`` only when no bound binds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EqualWeighted(;
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        sets::TD_Option{<:UniverseSets} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> EqualWeighted

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the weight bounds, asset sets, weight finaliser and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default. `strict` is execution control and stays static.

## Validation

  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.
  - `fb` schedules: `bind !== :nearest`.

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

  - [`optimise`](@ref)
  - [`NaiveOptimisationResult`](@ref)
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
    """
    $(field_dict[:cache_rows])
    """
    @fprop @vprop cache
    function EqualWeighted(wb::TD_Option{<:WbE_Wb}, sets::TD_Option{<:UniverseSets},
                           wf::TD{<:WeightFinaliser}, fb::TDO_Option{<:OptE_Opt},
                           strict::Bool, cache::Option{<:ReturnsBufferState})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :EqualWeighted)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb is a WeightBoundsEstimator"))
        end
        assert_time_dependent_substitution(EqualWeighted, (; wb, sets, wf, fb, strict),
                                           naive_optimiser_td_defaults())
        return new{typeof(wb), typeof(sets), typeof(wf), typeof(fb), typeof(strict),
                   typeof(cache)}(wb, sets, wf, fb, strict, cache)
    end
end
function EqualWeighted(; wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                       sets::TD_Option{<:UniverseSets} = nothing,
                       wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                       fb::TDO_Option{<:OptE_Opt} = nothing, strict::Bool = false,
                       cache::Option{<:ReturnsBufferState} = nothing)::EqualWeighted
    return EqualWeighted(wb, sets, wf, fb, strict, cache)
end
function non_investable_universe(ew::EqualWeighted, ni::VecStr)::EqualWeighted
    return rebuild_estimator(ew, (; sets = non_investable_sets(ew.sets, ni)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the equal-weighted portfolio optimisation.

Internal dispatch called by [`optimise`](@ref). Reduces the optimiser and the returns data to the Coverage Universe of the window with [`coverage_reduction`](@ref), assigns equal weights to the assets it keeps, then applies weight bounds. [`NaiveOptimisationResult`](@ref) expands the weights back onto the full asset universe.

This head fits no prior, so no Prior Result yields an Investable Mask for it. The Coverage Universe is the mask it derives instead: an asset is kept when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window. A stale finite price during an inactive spell weights nothing, and an asset outside the mask holds a zero. The result carries the mask as `imsk`, so a reader of a walk-forward has the same idiom here as in every other family. An all-dead window throws an `IsEmptyError`.

# Related

  - [`EqualWeighted`](@ref)
  - [`coverage_reduction`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(ew::EqualWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    assert_returns_result_dims(dims)
    ew = reset_time_dependent_estimator(ew)
    # This head fits no prior, so it derives the Coverage Universe of its own window and
    # reduces once, here: the weight bounds and the sets are stated over the full universe
    # and are viewed by the same index. `NaiveOptimisationResult` expands the weights back.
    cmsk, ew, rd = coverage_reduction(ew, rd; dims = dims)
    # `rd.X` is always observations by assets, whatever `dims` the caller passed, so the
    # asset count is `size(rd.X, 2)` unconditionally.
    N = size(rd.X, 2)
    w = fill(inv(N), N)
    wb = weight_bounds_constraints(ew.wb, ew.sets; N = N, strict = ew.strict,
                                   datatype = eltype(rd.X))
    retcode, w = finalise_weight_bounds(ew.wf, wb, w)
    return NaiveOptimisationResult(; pr = rd, wb = wb, retcode = retcode, w = w,
                                   imsk = cmsk, fb = nothing)
end
"""
    optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; dims::Int = 1, kwargs...) -> NaiveOptimisationResult

Run the equal-weighted portfolio optimisation.

# Arguments

  - `ew`: The equal-weighted optimiser to use.
  - $(arg_dict[:rd]) Its returns matrix and its Asset Panel give the Coverage Universe of the window, which is the universe the weights are spread over.
  - `dims`: Must be `1`. A `ReturnsResult` is always observations × assets, so `dims == 2` throws `ConflictingArgumentError`; build one in this layout with `prices_to_returns`.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  dims::Int = 1, kwargs...)::NaiveOptimisationResult
    return _optimise(ew, rd; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Draws portfolio weights at random from a Dirichlet distribution with concentration parameter `alpha`.

Use it for simulation, benchmarking, or stress testing. A scalar `alpha` draws from the symmetric Dirichlet distribution over ``N`` assets; a vector `alpha` must be one entry per asset.

``N`` is the size of the **Coverage Universe** of the window, not of the full asset universe. An asset is in it when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window, so an asset that is not yet listed, is delisted, or carries a stale finite price during an inactive spell, weights nothing and holds a zero. A vector `alpha` is still stated over the full universe, and is sliced to the Coverage Universe. The result carries that universe as its `imsk`. An all-dead window throws an `IsEmptyError`.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w} &\\sim \\mathrm{Dirichlet}(\\boldsymbol{\\alpha})\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: Portfolio weight vector before the weight bounds are applied. A Dirichlet draw is non-negative and sums to one.
  - ``\\boldsymbol{\\alpha}``: Concentration parameter, one entry per asset. A larger value concentrates the distribution near equal weights.

The weight finaliser `wf` then imposes the resolved weight bounds on ``\\boldsymbol{w}``, which by default are absent for this optimiser.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RandomWeighted(;
        alpha::Num_VecNum = 1,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        wb::TD_Option{<:WbE_Wb} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> RandomWeighted

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the weight bounds, asset sets, weight finaliser and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default (`nothing` for `wb`, `sets` and `fb`). `rng`, `seed` and `strict` are execution control and stay static.

## Validation

  - `alpha`: non-empty, and every element is positive and finite.
  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.
  - `fb` schedules: `bind !== :nearest`.

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

  - [`optimise`](@ref)
  - [`NaiveOptimisationResult`](@ref)
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
    """
    $(field_dict[:cache_rows])
    """
    @fprop @vprop cache
    function RandomWeighted(alpha::Num_VecNum, rng::Random.AbstractRNG,
                            seed::Option{<:Integer}, wb::TD_Option{<:WbE_Wb},
                            sets::TD_Option{<:UniverseSets}, wf::TD{<:WeightFinaliser},
                            fb::TDO_Option{<:OptE_Opt}, strict::Bool,
                            cache::Option{<:ReturnsBufferState})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :RandomWeighted)
        assert_nonempty_gt0_finite_val(alpha, :alpha)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb is a WeightBoundsEstimator"))
        end
        assert_time_dependent_substitution(RandomWeighted,
                                           (; alpha, rng, seed, wb, sets, wf, fb, strict),
                                           (; wf = IterativeWeightFinaliser()))
        return new{typeof(alpha), typeof(rng), typeof(seed), typeof(wb), typeof(sets),
                   typeof(wf), typeof(fb), typeof(strict), typeof(cache)}(alpha, rng, seed,
                                                                          wb, sets, wf, fb,
                                                                          strict, cache)
    end
end
function RandomWeighted(; alpha::Num_VecNum = 1,
                        rng::Random.AbstractRNG = Random.default_rng(),
                        seed::Option{<:Integer} = nothing,
                        wb::TD_Option{<:WbE_Wb} = nothing,
                        sets::TD_Option{<:UniverseSets} = nothing,
                        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                        fb::TDO_Option{<:OptE_Opt} = nothing, strict::Bool = false,
                        cache::Option{<:ReturnsBufferState} = nothing)::RandomWeighted
    return RandomWeighted(alpha, rng, seed, wb, sets, wf, fb, strict, cache)
end
function non_investable_universe(rw::RandomWeighted, ni::VecStr)::RandomWeighted
    return rebuild_estimator(rw, (; sets = non_investable_sets(rw.sets, ni)))
end
function time_dependent_field_defaults(::RandomWeighted)::NamedTuple
    return (; wf = IterativeWeightFinaliser())
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the random-weighted portfolio optimisation.

Internal dispatch called by [`optimise`](@ref). Reduces the optimiser and the returns data to the Coverage Universe of the window with [`coverage_reduction`](@ref), draws weights over the assets it keeps from a Dirichlet distribution parameterised by `rw.alpha`, then applies weight bounds. [`NaiveOptimisationResult`](@ref) expands the weights back onto the full asset universe.

This head fits no prior, so no Prior Result yields an Investable Mask for it. The Coverage Universe is the mask it derives instead: an asset is kept when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window. A stale finite price during an inactive spell weights nothing, and an asset outside the mask holds a zero. The result carries the mask as `imsk`, so a reader of a walk-forward has the same idiom here as in every other family. An all-dead window throws an `IsEmptyError`.

A vector `alpha` is one concentration per asset of the **full** universe, because that is the universe the caller states it over. Its length is therefore checked against the full width, before the reduction, and [`port_opt_view`](@ref) then slices it to the Coverage Universe with the rest of the estimator.

# Related

  - [`RandomWeighted`](@ref)
  - [`coverage_reduction`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(rw::RandomWeighted, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    assert_returns_result_dims(dims)
    rw = reset_time_dependent_estimator(rw)
    # `rd.X` is always observations by assets, whatever `dims` the caller passed, so the
    # asset count is `size(rd.X, 2)` unconditionally.
    if isa(rw.alpha, VecNum)
        # The caller states one concentration per asset of the full universe, so the check
        # reads the full width. The reduction below slices `alpha` with the rest.
        Nf = size(rd.X, 2)
        @argcheck(length(rw.alpha) == Nf,
                  DimensionMismatch("rw.alpha ($(length(rw.alpha))) must match N ($Nf)"))
    end
    # This head fits no prior, so it derives the Coverage Universe of its own window and
    # reduces once, here: the concentrations, the weight bounds and the sets are stated over
    # the full universe and are viewed by the same index. `NaiveOptimisationResult` expands
    # the weights back.
    cmsk, rw, rd = coverage_reduction(rw, rd; dims = dims)
    N = size(rd.X, 2)
    dist = if isa(rw.alpha, Number)
        Distributions.Dirichlet(N, rw.alpha)
    else
        Distributions.Dirichlet(rw.alpha)
    end
    rng = resolve_rng(rw.rng, rw.seed)
    w = rand(rng, dist)
    wb = weight_bounds_constraints(rw.wb, rw.sets; N = N, strict = rw.strict,
                                   datatype = eltype(rd.X))
    retcode, w = finalise_weight_bounds(rw.wf, wb, w)
    return NaiveOptimisationResult(; pr = rd, wb = wb, retcode = retcode, w = w,
                                   imsk = cmsk, fb = nothing)
end
"""
    optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; dims::Int = 1, kwargs...) -> NaiveOptimisationResult

Run the random-weighted portfolio optimisation.

# Arguments

  - `rw`: The random-weighted optimiser to use.
  - $(arg_dict[:rd]) Its returns matrix and its Asset Panel give the Coverage Universe of the window, which is the universe the draw is taken over.
  - `dims`: Must be `1`. A `ReturnsResult` is always observations × assets, so `dims == 2` throws `ConflictingArgumentError`; build one in this layout with `prices_to_returns`.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)::NaiveOptimisationResult
    return _optimise(rw, rd; dims = dims, kwargs...)
end

"""
$(DocStringExtensions.TYPEDEF)

Holds the weights it was handed, and solves nothing.

The hold-only head. Its weights are the previous fold's, threaded into `w` by the fold loop through [`factory`](@ref) exactly as they reach a [`TurnoverEstimator`](@ref), and it returns them verbatim on the full asset universe: no prior, no Coverage Universe, no weight bounds, so a hold is never rewritten. Its one use is as the fallback `fb` of an optimiser inside a walk-forward — the reference's `fallback = "previous_weights"` — where a failed solve then holds the book instead of writing `NaN` weights and losing the fold; a weight on an asset that left the panel is still held, and its returns are zeroed as a Held Gap. It is also a primary: `PreviousWeights(; w = w)` is a walk-forward that holds `w` on every fold. It refuses nothing at construction and fails at solve time when `w` is `nothing`, which is what fold 1 of a walk-forward, and a fold-less `optimise` with no `w`, hand it.

`needs_previous_weights` is `true`, so an optimiser that carries it as a fallback runs sequentially, and the loop threads the previous weights into it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PreviousWeights(; w::Option{<:VecNum} = nothing, fb::TDO_Option{<:OptE_Opt} = nothing) -> PreviousWeights

Keywords correspond to the struct's fields. `fb` may hold a [`TimeDependent`](@ref) per-fold schedule.

## Validation

  - `w`: `all(isfinite, w)`, else a `DomainError` is raised.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fb`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> PreviousWeights()
PreviousWeights
   w ┼ nothing
  fb ┴ nothing
```

# Related

  - [`optimise`](@ref)
  - [`factory(pw::PreviousWeights, w::VecNum)`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
  - [`needs_previous_weights`](@ref)
  - [`TurnoverEstimator`](@ref): Receives the previous weights through the same [`factory`](@ref) pass.
  - [`fold_loop`](@ref): Threads the previous fold's weights into `w`.
"""
@propagatable @concrete struct PreviousWeights <: NaiveOptimisationEstimator
    """
    Weights to hold, or `nothing` before the fold loop threads any.
    """
    w
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function PreviousWeights(w::Option{<:VecNum}, fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :PreviousWeights)
        if !isnothing(w)
            assert_finite(w, :w)
        end
        assert_time_dependent_substitution(PreviousWeights, (; w, fb),
                                           naive_optimiser_td_defaults())
        return new{typeof(w), typeof(fb)}(w, fb)
    end
end
function PreviousWeights(; w::Option{<:VecNum} = nothing,
                         fb::TDO_Option{<:OptE_Opt} = nothing)::PreviousWeights
    return PreviousWeights(w, fb)
end
function needs_previous_weights(::PreviousWeights)
    return true
end
"""
    factory(pw::PreviousWeights, w::VecNum) -> PreviousWeights

Thread the previous fold's weights into the hold-only head, and on into its fallback.

# Arguments

  - `pw`: The head.
  - `w`: The weights the fold loop threads.

# Returns

  - `PreviousWeights`: The head holding `w`, with `fb` propagated through [`factory`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.factory(PreviousWeights(), [0.25, 0.75])
PreviousWeights
   w ┼ Vector{Float64}: [0.25, 0.75]
  fb ┴ nothing
```

# Related

  - [`PreviousWeights`](@ref)
  - [`factory`](@ref)
  - [`fold_loop`](@ref)
"""
function factory(pw::PreviousWeights, w::VecNum)::PreviousWeights
    return PreviousWeights(; w = w, fb = factory(pw.fb, w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the held weights as a result, or a failure when there are none.

Internal dispatch called by [`optimise`](@ref). The head reads nothing off `rd` but its width: the weights are returned verbatim, on the universe they were threaded on, with `imsk = nothing` and no weight bounds. A head whose `w` is `nothing` answers an [`OptimisationFailure`](@ref) naming the missing weights, so a fallback chain that reaches it walks on, and its weights are a `NaN` vector of the carrier's width — what every failed solve carries, so a fold reads the failure as it reads any other — or `nothing` when the carrier has no returns to take a width from.

# Algorithm

 1. With `w` set, give a [`NaiveOptimisationResult`](@ref) carrying `w` and an [`OptimisationSuccess`](@ref).
 2. With `w` unset, give one carrying an [`OptimisationFailure`](@ref) and `NaN` weights, one per column of `rd.X`, or `nothing` when `rd.X` is `nothing`.

# Related

  - [`PreviousWeights`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(pw::PreviousWeights, rd::ReturnsResult = ReturnsResult(); kwargs...)
    pw = reset_time_dependent_estimator(pw)
    (retcode, w) = if isnothing(pw.w)
        (OptimisationFailure(;
                             res = "`PreviousWeights` holds no weights: `w` is `nothing`. The fold loop threads the previous fold's weights into it from fold 2 on, so fold 1 of a walk-forward, and a fold-less `optimise`, must set `w` themselves."),
         failed_hold_weights(rd.X))
    else
        (OptimisationSuccess(), pw.w)
    end
    return NaiveOptimisationResult(; pr = rd, wb = nothing, retcode = retcode, w = w,
                                   fb = nothing)
end
"""
    failed_hold_weights(X::Nothing)
    failed_hold_weights(X::MatNum)

The weights a [`PreviousWeights`](@ref) with nothing to hold answers: `NaN` at every asset of the carrier, or `nothing` when the carrier has no returns to take a width from.

# Related

  - [`PreviousWeights`](@ref)
  - [`_optimise(pw::PreviousWeights, rd::ReturnsResult = ReturnsResult(); kwargs...)`](@ref)
"""
function failed_hold_weights(::Nothing)
    return nothing
end
function failed_hold_weights(X::MatNum)
    return fill(convert(eltype(X), NaN), size(X, 2))
end
"""
    optimise(pw::PreviousWeights{<:Any, Nothing}, rd::ReturnsResult = ReturnsResult();
             kwargs...) -> NaiveOptimisationResult

Hold the weights the head carries.

# Arguments

  - `pw`: The hold-only head.
  - $(arg_dict[:rd]) Read for nothing, and recorded on the result as `pr`.
  - `kwargs`: Additional keyword arguments, ignored.
"""
function optimise(pw::PreviousWeights{<:Any, Nothing}, rd::ReturnsResult = ReturnsResult();
                  kwargs...)::NaiveOptimisationResult
    return _optimise(pw, rd; kwargs...)
end
export NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeighted,
       PreviousWeights
