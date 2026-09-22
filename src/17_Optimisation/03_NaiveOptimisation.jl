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

Return whether the naive optimiser requires previous portfolio weights: through its fees, a per-fold schedule on any field, or its fallback estimator.

Every naive head carries a `fees` field, and a static [`Fees`](@ref) whose turnover term is not fixed reads the previous weights through [`factory`](@ref) exactly as it does on a [`JuMPOptimiser`](@ref); a [`TimeDependent`](@ref) `fees` is read through the schedule scan instead.

# Related

  - [`needs_previous_weights`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
"""
function needs_previous_weights(opt::NaiveOptimisationEstimator)
    return any(f -> needs_previous_weights(getfield(opt, f)), time_dependent_fields(opt)) ||
           needs_previous_weights(opt.fees) ||
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
        wb::Option{<:WeightBounds}, fees::Option{<:Fees} = nothing,
        retcode::OptimisationReturnCode, w::Option{<:VecNum},
        imsk::Option{<:BitVector} = nothing, fb::Option{<:OptE_Opt_FbChain}
    ) -> NaiveOptimisationResult

Keywords correspond to the struct's fields. The keyword constructor expands `w` onto the full asset universe through [`expand_investable_weights`](@ref), which is the one door [`_optimise`](@ref) exits through. The positional constructor never expands. `fees` is the fee the head was charged with, on the universe it solved on, so a walk-forward fold charges it through [`extract_fees`](@ref) exactly as it charges a hierarchical head's; it is `nothing` for a head that carries none.

# Examples

```jldoctest
julia> NaiveOptimisationResult(; pr = nothing, wb = nothing, retcode = OptimisationSuccess(),
                               w = [0.5, 0.5], fb = nothing)
NaiveOptimisationResult
       pr ┼ nothing
       wb ┼ nothing
     fees ┼ nothing
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
    $(field_dict[:fees_res])
    """
    fees
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
                                     fees::Option{<:Fees}, retcode::OptimisationReturnCode,
                                     w::Option{<:VecNum}, imsk::Option{<:BitVector},
                                     fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(pr), typeof(wb), typeof(fees), typeof(retcode), typeof(w),
                   typeof(imsk), typeof(fb)}(pr, wb, fees, retcode, w, imsk, fb)
    end
end
function NaiveOptimisationResult(; pr::Option{<:Pr_RR}, wb::Option{<:WeightBounds},
                                 fees::Option{<:Fees} = nothing,
                                 retcode::OptimisationReturnCode, w::Option{<:VecNum},
                                 imsk::Option{<:BitVector} = nothing,
                                 fb::Option{<:OptE_Opt_FbChain})::NaiveOptimisationResult
    return NaiveOptimisationResult(pr, wb, fees, retcode,
                                   expand_investable_weights(imsk, w), imsk, fb)
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
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        sq::Bool = false,
        brt::Bool = false,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> InverseVolatility

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the prior estimator, weight bounds, fees, asset sets, weight finaliser and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default. `sq`, `brt` and `strict` are execution control and stay static. `fees` is the fee the result carries and a walk-forward fold charges; a turnover fee reads the previous weights the loop threads through [`factory`](@ref).

## Validation

  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.
  - If `fees` is a [`FeesEstimator`](@ref): `!isnothing(sets)`.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `fees`: Recursively viewed via [`port_opt_view`](@ref).
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
    fees ┼ nothing
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
    $(field_dict[:fees])
    """
    @fprop @vprop fees
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
                               fees::TD_Option{<:FeesE_Fees},
                               sets::TD_Option{<:UniverseSets}, wf::TD{<:WeightFinaliser},
                               fb::TDO_Option{<:OptE_Opt}, sq::Bool, brt::Bool,
                               strict::Bool, cache::Option{<:ReturnsBufferState})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :InverseVolatility)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb is a WeightBoundsEstimator"))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when fees is a FeesEstimator"))
        end
        assert_time_dependent_substitution(InverseVolatility,
                                           (; pe, wb, fees, sets, wf, fb, sq, brt, strict),
                                           merge(naive_optimiser_td_defaults(),
                                                 (; pe = EmpiricalPrior())))
        return new{typeof(pe), typeof(wb), typeof(fees), typeof(sets), typeof(wf),
                   typeof(fb), typeof(sq), typeof(brt), typeof(strict), typeof(cache)}(pe,
                                                                                       wb,
                                                                                       fees,
                                                                                       sets,
                                                                                       wf,
                                                                                       fb,
                                                                                       sq,
                                                                                       brt,
                                                                                       strict,
                                                                                       cache)
    end
end
function InverseVolatility(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
                           wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                           fees::TD_Option{<:FeesE_Fees} = nothing,
                           sets::TD_Option{<:UniverseSets} = nothing,
                           wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                           fb::TDO_Option{<:OptE_Opt} = nothing, sq::Bool = false,
                           brt::Bool = false, strict::Bool = false,
                           cache::Option{<:ReturnsBufferState} = nothing)::InverseVolatility
    return InverseVolatility(pe, wb, fees, sets, wf, fb, sq, brt, strict, cache)
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
    # Resolve the fee on the caller's own universe, before the door below narrows `sets`,
    # for the reason the hierarchical heads do; `investable_fees_view` then places it on
    # the axes the mask leaves.
    fees = investable_fees_view(fees_constraints(iv.fees, iv.sets; strict = iv.strict,
                                                 datatype = eltype(pr.X)),
                                investable_mask(pr), size(pr.X, 2))
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
    return NaiveOptimisationResult(; pr = pr, wb = wb, fees = fees, retcode = retcode,
                                   w = w, imsk = imsk, fb = nothing)
end
"""
    optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
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
function optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
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
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> EqualWeighted

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the weight bounds, fees, asset sets, weight finaliser and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default. `strict` is execution control and stays static. `fees` is the fee the result carries and a walk-forward fold charges; a turnover fee reads the previous weights the loop threads through [`factory`](@ref).

## Validation

  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.
  - If `fees` is a [`FeesEstimator`](@ref): `!isnothing(sets)`.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `fees`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> EqualWeighted()
EqualWeighted
      wb ┼ WeightBounds
         │   lb ┼ Float64: 0.0
         │   ub ┴ Float64: 1.0
    fees ┼ nothing
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
    $(field_dict[:fees])
    """
    @fprop @vprop fees
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
    function EqualWeighted(wb::TD_Option{<:WbE_Wb}, fees::TD_Option{<:FeesE_Fees},
                           sets::TD_Option{<:UniverseSets}, wf::TD{<:WeightFinaliser},
                           fb::TDO_Option{<:OptE_Opt}, strict::Bool,
                           cache::Option{<:ReturnsBufferState})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :EqualWeighted)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb is a WeightBoundsEstimator"))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when fees is a FeesEstimator"))
        end
        assert_time_dependent_substitution(EqualWeighted,
                                           (; wb, fees, sets, wf, fb, strict),
                                           naive_optimiser_td_defaults())
        return new{typeof(wb), typeof(fees), typeof(sets), typeof(wf), typeof(fb),
                   typeof(strict), typeof(cache)}(wb, fees, sets, wf, fb, strict, cache)
    end
end
function EqualWeighted(; wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                       fees::TD_Option{<:FeesE_Fees} = nothing,
                       sets::TD_Option{<:UniverseSets} = nothing,
                       wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                       fb::TDO_Option{<:OptE_Opt} = nothing, strict::Bool = false,
                       cache::Option{<:ReturnsBufferState} = nothing)::EqualWeighted
    return EqualWeighted(wb, fees, sets, wf, fb, strict, cache)
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
    # The fee is resolved on the caller's own universe before the door below narrows
    # `sets`, and placed on the axes the mask leaves after it.
    Nf = size(rd.X, 2)
    fees = fees_constraints(ew.fees, ew.sets; strict = ew.strict, datatype = eltype(rd.X))
    # This head fits no prior, so it derives the Coverage Universe of its own window and
    # reduces once, here: the weight bounds and the sets are stated over the full universe
    # and are viewed by the same index. `NaiveOptimisationResult` expands the weights back.
    cmsk, ew, rd = coverage_reduction(ew, rd; dims = dims)
    fees = investable_fees_view(fees, cmsk, Nf)
    # `rd.X` is always observations by assets, whatever `dims` the caller passed, so the
    # asset count is `size(rd.X, 2)` unconditionally.
    N = size(rd.X, 2)
    w = fill(inv(N), N)
    wb = weight_bounds_constraints(ew.wb, ew.sets; N = N, strict = ew.strict,
                                   datatype = eltype(rd.X))
    retcode, w = finalise_weight_bounds(ew.wf, wb, w)
    return NaiveOptimisationResult(; pr = rd, wb = wb, fees = fees, retcode = retcode,
                                   w = w, imsk = cmsk, fb = nothing)
end
"""
    optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; dims::Int = 1, kwargs...) -> NaiveOptimisationResult

Run the equal-weighted portfolio optimisation.

# Arguments

  - `ew`: The equal-weighted optimiser to use.
  - $(arg_dict[:rd]) Its returns matrix and its Asset Panel give the Coverage Universe of the window, which is the universe the weights are spread over.
  - `dims`: Must be `1`. A `ReturnsResult` is always observations × assets, so `dims == 2` throws `ConflictingArgumentError`; build one in this layout with `prices_to_returns`.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
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
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> RandomWeighted

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the weight bounds, fees, asset sets, weight finaliser and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default (`nothing` for `wb`, `fees`, `sets` and `fb`). `rng`, `seed` and `strict` are execution control and stay static. `fees` is the fee the result carries and a walk-forward fold charges; a turnover fee reads the previous weights the loop threads through [`factory`](@ref).

## Validation

  - `alpha`: non-empty, and every element is positive and finite.
  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.
  - If `fees` is a [`FeesEstimator`](@ref): `!isnothing(sets)`.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `alpha`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `fees`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> RandomWeighted()
RandomWeighted
   alpha ┼ Int64: 1
     rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
    seed ┼ nothing
      wb ┼ nothing
    fees ┼ nothing
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
    $(field_dict[:fees])
    """
    @fprop @vprop fees
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
                            fees::TD_Option{<:FeesE_Fees}, sets::TD_Option{<:UniverseSets},
                            wf::TD{<:WeightFinaliser}, fb::TDO_Option{<:OptE_Opt},
                            strict::Bool, cache::Option{<:ReturnsBufferState})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :RandomWeighted)
        assert_nonempty_gt0_finite_val(alpha, :alpha)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb is a WeightBoundsEstimator"))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when fees is a FeesEstimator"))
        end
        assert_time_dependent_substitution(RandomWeighted,
                                           (; alpha, rng, seed, wb, fees, sets, wf, fb,
                                            strict), (; wf = IterativeWeightFinaliser()))
        return new{typeof(alpha), typeof(rng), typeof(seed), typeof(wb), typeof(fees),
                   typeof(sets), typeof(wf), typeof(fb), typeof(strict), typeof(cache)}(alpha,
                                                                                        rng,
                                                                                        seed,
                                                                                        wb,
                                                                                        fees,
                                                                                        sets,
                                                                                        wf,
                                                                                        fb,
                                                                                        strict,
                                                                                        cache)
    end
end
function RandomWeighted(; alpha::Num_VecNum = 1,
                        rng::Random.AbstractRNG = Random.default_rng(),
                        seed::Option{<:Integer} = nothing,
                        wb::TD_Option{<:WbE_Wb} = nothing,
                        fees::TD_Option{<:FeesE_Fees} = nothing,
                        sets::TD_Option{<:UniverseSets} = nothing,
                        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                        fb::TDO_Option{<:OptE_Opt} = nothing, strict::Bool = false,
                        cache::Option{<:ReturnsBufferState} = nothing)::RandomWeighted
    return RandomWeighted(alpha, rng, seed, wb, fees, sets, wf, fb, strict, cache)
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
    Nf = size(rd.X, 2)
    if isa(rw.alpha, VecNum)
        # The caller states one concentration per asset of the full universe, so the check
        # reads the full width. The reduction below slices `alpha` with the rest.
        @argcheck(length(rw.alpha) == Nf,
                  DimensionMismatch("rw.alpha ($(length(rw.alpha))) must match N ($Nf)"))
    end
    # The fee is resolved on the caller's own universe before the door below narrows
    # `sets`, and placed on the axes the mask leaves after it.
    fees = fees_constraints(rw.fees, rw.sets; strict = rw.strict, datatype = eltype(rd.X))
    # This head fits no prior, so it derives the Coverage Universe of its own window and
    # reduces once, here: the concentrations, the weight bounds and the sets are stated over
    # the full universe and are viewed by the same index. `NaiveOptimisationResult` expands
    # the weights back.
    cmsk, rw, rd = coverage_reduction(rw, rd; dims = dims)
    fees = investable_fees_view(fees, cmsk, Nf)
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
    return NaiveOptimisationResult(; pr = rd, wb = wb, fees = fees, retcode = retcode,
                                   w = w, imsk = cmsk, fb = nothing)
end
"""
    optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; dims::Int = 1, kwargs...) -> NaiveOptimisationResult

Run the random-weighted portfolio optimisation.

# Arguments

  - `rw`: The random-weighted optimiser to use.
  - $(arg_dict[:rd]) Its returns matrix and its Asset Panel give the Coverage Universe of the window, which is the universe the draw is taken over.
  - `dims`: Must be `1`. A `ReturnsResult` is always observations × assets, so `dims == 2` throws `ConflictingArgumentError`; build one in this layout with `prices_to_returns`.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                     Nothing}, rd::ReturnsResult; dims::Int = 1,
                  kwargs...)::NaiveOptimisationResult
    return _optimise(rw, rd; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Allocates the constant rebalanced portfolio that maximises the log wealth of the window it is fit on, by Cover's fixed point and no solver.

The Hindsight Comparator of the online selection family with no solver attached: fit it on the rows a strategy is scored on and predict it in sample over the same rows, and its log wealth is the ceiling every constant portfolio, and every strategy that pays no attention to order, is measured against with [`log_wealth_regret`](@ref). It is also the solver-free optimiser a follow-the-leader rule re-solves on the rows it selects. A caller with a JuMP solver reaches the same portfolio, and a bounded or otherwise constrained one, through [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref); the parity of the two is tested to the solver's tolerance.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}^{\\star} &= \\underset{\\boldsymbol{w} \\in \\Delta^{N}}{\\arg\\max} \\sum_{t=1}^{T} \\log\\left(1 + \\boldsymbol{x}_t^{\\intercal} \\boldsymbol{w}\\right)\\,, \\\\
w_i^{(k+1)} &= w_i^{(k)} \\, \\frac{1}{T} \\sum_{t=1}^{T} \\frac{1 + x_{t,i}}{1 + \\boldsymbol{x}_t^{\\intercal} \\boldsymbol{w}^{(k)}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}^{\\star}``: The best constant rebalanced portfolio, on the simplex ``\\Delta^{N}``.
  - $(math_dict[:x_t_obs]) ``1 + x_{t,i}`` is asset ``i``'s price relative.
  - $(math_dict[:T])
  - ``w_i^{(k)}``: Weight of asset ``i`` at iteration ``k``, started from ``1/N``.

The objective is concave on the simplex, so its maximum is global. The multiplier of an asset is the average ratio of its price relative to the portfolio's own; an asset that beats the portfolio on average grows, and at a fixed point every held asset has multiplier one, which is the first-order condition. The iteration is a minorise–maximise scheme, so the log wealth is non-decreasing at every step. A weight that starts at zero stays at zero, so the uniform start is what lets every asset compete.

The iteration stops on a certificate, or after `iters` steps. The certificate is ``T \\log \\max_i g_i`` over the multipliers ``g_i`` of the current weights. By concavity and Jensen's inequality it bounds from above the shortfall of the log wealth from the optimum, and it reaches zero at the optimum, a corner included. So `converged = true` means that the log wealth is within `tol * max(1, |log wealth|)` of the optimum. The certificate bounds the log wealth, not the weights. At a leader that holds no weight on an asset, the weight of that asset decays sublinearly, so the default budget can stop first and report `converged = false`. The result's `retcode.res` carries `converged`, `iterations` and the certificate `gap`. The exact form, on a solver, is [`MeanRisk`](@ref)`(; obj = MaximumReturn(), opt = JuMPOptimiser(; slv, ret = LogarithmicReturn()))`: one exponential cone per row, exact to the solver's tolerance, and the only form that honours a bound.

The head is simplex-only. The fixed point knows no bound other than the simplex, so a `wb` that binds is imposed by the weight finaliser `wf` **after** the fixed point: the returned weights are then a repair of the unconstrained optimum, not the constrained one. A bounded or constrained Hindsight Comparator is [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref), with the bounds stated on its optimiser. The default bounds are the simplex, and the finaliser leaves the fixed point untouched.

The head fits no prior and derives the Coverage Universe of its own window: an asset is kept when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row, and every other asset holds a zero. The result carries that universe as `imsk`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BestConstantRebalancedPortfolio(;
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        iters::Integer = 20_000,
        tol::Number = 1e-12,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> BestConstantRebalancedPortfolio

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the weight bounds, fees, asset sets, weight finaliser and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default. `iters`, `tol` and `strict` are execution control and stay static. `fees` is the fee the result carries and a walk-forward fold charges; a turnover fee reads the previous weights the loop threads through [`factory`](@ref).

## Validation

  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.
  - If `fees` is a [`FeesEstimator`](@ref): `!isnothing(sets)`.
  - `fb` schedules: `bind !== :nearest`.
  - `iters > 0`.
  - `tol > 0`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `fees`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> BestConstantRebalancedPortfolio()
BestConstantRebalancedPortfolio
      wb ┼ WeightBounds
         │   lb ┼ Float64: 0.0
         │   ub ┴ Float64: 1.0
    fees ┼ nothing
    sets ┼ nothing
      wf ┼ IterativeWeightFinaliser
         │   iter ┴ Int64: 100
      fb ┼ nothing
   iters ┼ Int64: 20000
     tol ┼ Float64: 1.0e-12
  strict ┴ Bool: false
```

# Related

  - [`optimise`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
  - [`log_wealth_regret`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`EqualWeighted`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cover1984])
"""
@propagatable @concrete struct BestConstantRebalancedPortfolio <: NaiveOptimisationEstimator
    """
    $(field_dict[:wb])
    """
    @vprop wb
    """
    $(field_dict[:fees])
    """
    @fprop @vprop fees
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
    Maximum number of fixed-point iterations.
    """
    iters
    """
    Convergence tolerance on the certificate of the shortfall in log wealth from the optimum, relative to the log wealth.
    """
    tol
    """
    $(field_dict[:strict_opt])
    """
    strict
    """
    $(field_dict[:cache_rows])
    """
    @fprop @vprop cache
    function BestConstantRebalancedPortfolio(wb::TD_Option{<:WbE_Wb},
                                             fees::TD_Option{<:FeesE_Fees},
                                             sets::TD_Option{<:UniverseSets},
                                             wf::TD{<:WeightFinaliser},
                                             fb::TDO_Option{<:OptE_Opt}, iters::Integer,
                                             tol::Number, strict::Bool,
                                             cache::Option{<:ReturnsBufferState})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :BestConstantRebalancedPortfolio)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb is a WeightBoundsEstimator"))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when fees is a FeesEstimator"))
        end
        @argcheck(iters > 0, DomainError(iters, "`iters` must be positive"))
        @argcheck(tol > zero(tol), DomainError(tol, "`tol` must be positive"))
        assert_time_dependent_substitution(BestConstantRebalancedPortfolio,
                                           (; wb, fees, sets, wf, fb, iters, tol, strict),
                                           naive_optimiser_td_defaults())
        return new{typeof(wb), typeof(fees), typeof(sets), typeof(wf), typeof(fb),
                   typeof(iters), typeof(tol), typeof(strict), typeof(cache)}(wb, fees,
                                                                              sets, wf, fb,
                                                                              iters, tol,
                                                                              strict, cache)
    end
end
function BestConstantRebalancedPortfolio(; wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                                         fees::TD_Option{<:FeesE_Fees} = nothing,
                                         sets::TD_Option{<:UniverseSets} = nothing,
                                         wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                                         fb::TDO_Option{<:OptE_Opt} = nothing,
                                         iters::Integer = 20_000, tol::Number = 1e-12,
                                         strict::Bool = false,
                                         cache::Option{<:ReturnsBufferState} = nothing)::BestConstantRebalancedPortfolio
    return BestConstantRebalancedPortfolio(wb, fees, sets, wf, fb, iters, tol, strict,
                                           cache)
end
function non_investable_universe(bcrp::BestConstantRebalancedPortfolio,
                                 ni::VecStr)::BestConstantRebalancedPortfolio
    return rebuild_estimator(bcrp, (; sets = non_investable_sets(bcrp.sets, ni)))
end
"""
    cover_fixed_point(X::MatNum, iters::Integer, tol::Number) -> NamedTuple

Run Cover's fixed-point iteration for the best constant rebalanced portfolio over a matrix of price relatives.

The kernel of [`BestConstantRebalancedPortfolio`](@ref), on a bare matrix so it can be tested against the closed form on a hand example and reused by a rule that re-solves it. Starts from the uniform portfolio and multiplies each weight by the average ratio of its price relative to the portfolio's, renormalising after every step. Before each step it reads the duality-gap certificate ``T \\log \\max_i g_i`` over the multipliers ``g_i``, an upper bound on the shortfall of the log wealth from the optimum, and it stops when the certificate is at most `tol * max(1, |log wealth|)`, or after `iters` steps.

# Arguments

  - `X`: Price relatives, `observations × assets`, every entry positive.
  - `iters`: Maximum number of iterations.
  - `tol`: Relative tolerance on the certificate of the shortfall in log wealth.

# Returns

  - `fp::NamedTuple`: `w`, the weights on the simplex; `log_wealth`, ``\\sum_t \\log(\\boldsymbol{x}_t^{\\intercal} \\boldsymbol{w})``; `gap`, the certificate at `w`, which bounds the optimal log wealth minus `log_wealth`; `converged`, whether the certificate met the tolerance; `iterations`, the number of multiplicative steps taken, zero when the uniform start already meets it.

# Related

  - [`BestConstantRebalancedPortfolio`](@ref)
"""
function cover_fixed_point(X::MatNum, iters::Integer, tol::Number)
    T, N = size(X)
    w = fill(one(eltype(X)) / N, N)
    wc = w
    local lw, gap
    converged = false
    iterations = 0
    for k in 0:iters
        # The certificate is read at `wc`, the weights the kernel returns. Cover's multiplier
        # of each asset is the mean over observations of its price relative against the
        # portfolio's own. By concavity and Jensen's inequality the largest one bounds the
        # shortfall, `lw* - lw <= T log(max_j g_j)`, and the bound reaches zero at the
        # optimum, a corner included, where every multiplier is at most one.
        wc = w
        p = X * wc
        lw = sum(log, p)
        g = vec(Statistics.mean(X ./ p; dims = 1))
        gap = max(zero(lw), T * log(maximum(g)))
        iterations = k
        converged = gap <= tol * max(one(lw), abs(lw))
        if converged
            break
        end
        # The multiplicative step, renormalised onto the simplex. The step after the last
        # certificate of the budget is not returned.
        w = wc .* g
        w ./= sum(w)
    end
    return (; w = wc, log_wealth = lw, gap = gap, converged = converged,
            iterations = iterations)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the best constant rebalanced portfolio optimisation.

Internal dispatch called by [`optimise`](@ref). Reduces the optimiser and the returns data to the Coverage Universe of the window with [`coverage_reduction`](@ref), forms the price relatives `1 .+ rd.X`, runs [`cover_fixed_point`](@ref) from the uniform portfolio, then applies weight bounds through the finaliser. [`NaiveOptimisationResult`](@ref) expands the weights back onto the full asset universe. The return code is the finaliser's; a successful one carries the fixed point's `converged` flag, iteration count and certificate `gap` in its `res`, so a run that stopped at `iters` is read off the result rather than guarded.

# Related

  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`cover_fixed_point`](@ref)
  - [`coverage_reduction`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(bcrp::BestConstantRebalancedPortfolio, rd::ReturnsResult; dims::Int = 1,
                   kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    assert_returns_result_dims(dims)
    bcrp = reset_time_dependent_estimator(bcrp)
    # The fee is resolved on the caller's own universe before the door below narrows
    # `sets`, and placed on the axes the mask leaves after it.
    Nf = size(rd.X, 2)
    fees = fees_constraints(bcrp.fees, bcrp.sets; strict = bcrp.strict,
                            datatype = eltype(rd.X))
    # This head fits no prior, so it derives the Coverage Universe of its own window and
    # reduces once, here, as `EqualWeighted` does. `NaiveOptimisationResult` expands back.
    cmsk, bcrp, rd = coverage_reduction(bcrp, rd; dims = dims)
    fees = investable_fees_view(fees, cmsk, Nf)
    X = one(eltype(rd.X)) .+ rd.X
    N = size(X, 2)
    fp = cover_fixed_point(X, bcrp.iters, bcrp.tol)
    wb = weight_bounds_constraints(bcrp.wb, bcrp.sets; N = N, strict = bcrp.strict,
                                   datatype = eltype(X))
    retcode, w = finalise_weight_bounds(bcrp.wf, wb, fp.w)
    if isa(retcode, OptimisationSuccess)
        retcode = OptimisationSuccess(;
                                      res = (; converged = fp.converged,
                                             iterations = fp.iterations, gap = fp.gap))
    end
    return NaiveOptimisationResult(; pr = rd, wb = wb, fees = fees, retcode = retcode,
                                   w = w, imsk = cmsk, fb = nothing)
end
"""
    optimise(bcrp::BestConstantRebalancedPortfolio{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; dims::Int = 1, kwargs...) -> NaiveOptimisationResult

Run the best constant rebalanced portfolio optimisation.

A Hindsight Comparator is fit on the rows it is scored on, so `predict(optimise(bcrp, rd_test), rd_test)` is the comparator's prediction result over `rd_test`, and `cross_val_predict(bcrp, rd, cv)` is the causal constant portfolio over the same rows.

# Arguments

  - `bcrp`: The best constant rebalanced portfolio optimiser to use.
  - $(arg_dict[:rd]) Its returns matrix and its Asset Panel give the Coverage Universe of the window, which is the universe the fixed point runs over.
  - `dims`: Must be `1`. A `ReturnsResult` is always observations × assets, so `dims == 2` throws `ConflictingArgumentError`; build one in this layout with `prices_to_returns`.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(bcrp::BestConstantRebalancedPortfolio{<:Any, <:Any, <:Any, <:Any,
                                                        Nothing}, rd::ReturnsResult;
                  dims::Int = 1, kwargs...)::NaiveOptimisationResult
    return _optimise(bcrp, rd; dims = dims, kwargs...)
end

"""
$(DocStringExtensions.TYPEDEF)

Holds the weights it was handed, and solves nothing.

The hold-only head. Its weights are the previous fold's, threaded into `w` by the fold loop through [`factory`](@ref) exactly as they reach a [`TurnoverEstimator`](@ref), and it returns them verbatim on the full asset universe: no prior, no Coverage Universe, no weight bounds, so a hold is never rewritten. Its one use is as the fallback `fb` of an optimiser inside a walk-forward, where a failed solve then holds the book instead of writing `NaN` weights and losing the fold; a weight on an asset that left the panel is still held, and its returns are zeroed as a Held Gap. It is also a primary: `PreviousWeights(; w = w)` is a walk-forward that holds `w` on every fold. It refuses nothing at construction and fails at solve time when `w` is `nothing`, which is what fold 1 of a walk-forward, and a fold-less `optimise` with no `w`, hand it.

`needs_previous_weights` is `true`, so an optimiser that carries it as a fallback runs sequentially, and the loop threads the previous weights into it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PreviousWeights(; w::Option{<:VecNum} = nothing, fees::Option{<:Fees} = nothing, fb::TDO_Option{<:OptE_Opt} = nothing) -> PreviousWeights

Keywords correspond to the struct's fields. `fb` may hold a [`TimeDependent`](@ref) per-fold schedule. `fees` is a resolved [`Fees`](@ref), never an estimator, because the head holds no `sets` to resolve one over; it is the fee the result carries and a walk-forward fold charges, and under a Previous-Weights Source its turnover term prices the rebalance from the drifted book back to the held target.

## Validation

  - `w`: `all(isfinite, w)`, else a `DomainError` is raised.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> PreviousWeights()
PreviousWeights
     w ┼ nothing
  fees ┼ nothing
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
    $(field_dict[:fees_res])
    """
    @fprop fees
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function PreviousWeights(w::Option{<:VecNum}, fees::Option{<:Fees},
                             fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :PreviousWeights)
        if !isnothing(w)
            assert_finite(w, :w)
        end
        assert_time_dependent_substitution(PreviousWeights, (; w, fees, fb),
                                           naive_optimiser_td_defaults())
        return new{typeof(w), typeof(fees), typeof(fb)}(w, fees, fb)
    end
end
function PreviousWeights(; w::Option{<:VecNum} = nothing, fees::Option{<:Fees} = nothing,
                         fb::TDO_Option{<:OptE_Opt} = nothing)::PreviousWeights
    return PreviousWeights(w, fees, fb)
end
function needs_previous_weights(::PreviousWeights)
    return true
end
"""
    factory(pw::PreviousWeights, w::VecNum) -> PreviousWeights

Thread the previous fold's weights into the hold-only head, into its fee, and on into its fallback.

# Arguments

  - `pw`: The head.
  - `w`: The weights the fold loop threads.

# Returns

  - `PreviousWeights`: The head holding `w`, with `fees` and `fb` propagated through [`factory`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.factory(PreviousWeights(), [0.25, 0.75])
PreviousWeights
     w ┼ Vector{Float64}: [0.25, 0.75]
  fees ┼ nothing
    fb ┴ nothing
```

# Related

  - [`PreviousWeights`](@ref)
  - [`factory`](@ref)
  - [`fold_loop`](@ref)
"""
function factory(pw::PreviousWeights, w::VecNum)::PreviousWeights
    return PreviousWeights(; w = w, fees = factory(pw.fees, w), fb = factory(pw.fb, w))
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
    return NaiveOptimisationResult(; pr = rd, wb = nothing,
                                   fees = investable_fees_view(pw.fees, nothing, nothing),
                                   retcode = retcode, w = w, fb = nothing)
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
    optimise(pw::PreviousWeights{<:Any, <:Any, Nothing}, rd::ReturnsResult = ReturnsResult();
             kwargs...) -> NaiveOptimisationResult

Hold the weights the head carries.

# Arguments

  - `pw`: The hold-only head.
  - $(arg_dict[:rd]) Read for nothing, and recorded on the result as `pr`.
  - `kwargs`: Additional keyword arguments, ignored.
"""
function optimise(pw::PreviousWeights{<:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); kwargs...)::NaiveOptimisationResult
    return _optimise(pw, rd; kwargs...)
end
export NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeighted,
       PreviousWeights, BestConstantRebalancedPortfolio
