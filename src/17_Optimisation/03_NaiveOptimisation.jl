"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the optimisers that compute portfolio weights without a solver.

Each subtype gives its weights by a closed form, a random draw, a fixed-point iteration or a copy of the weights it holds, and then imposes its weight bounds with its weight finaliser. [`PreviousWeights`](@ref) is the exception. It holds its weights as it was given them, and imposes no bounds. Every subtype carries a `fees` field and a fallback `fb`, and gives a [`NaiveOptimisationResult`](@ref).

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`InverseVolatility`](@ref)
  - [`EqualWeighted`](@ref)
  - [`RandomWeighted`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`PreviousWeights`](@ref)
"""
abstract type NaiveOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the naive optimiser needs the previous portfolio weights.

The answer is `true` when one of three parts needs them. The first is a [`TimeDependent`](@ref) schedule on any field. The second is the `fees`. A static [`Fees`](@ref) whose turnover or liquidation term is not fixed reads the previous weights through [`factory`](@ref), as it does on a [`JuMPOptimiser`](@ref). The third is the fallback `fb`. [`PreviousWeights`](@ref) overrides this method and always answers `true`.

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

Assert internal validity for a naive optimisation estimator.

The default method checks nothing and returns `nothing`.

# Related

  - [`assert_internal_optimiser`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
"""
function assert_internal_optimiser(::NaiveOptimisationEstimator)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert external validity for a naive optimisation estimator.

The default method checks nothing and returns `nothing`. [`InverseVolatility`](@ref) overrides it, because its prior estimator can hold a result.

# Related

  - [`assert_external_optimiser`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
"""
function assert_external_optimiser(::NaiveOptimisationEstimator)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when a field of the naive optimiser holds a [`TimeDependent`](@ref) schedule, or when its fallback `fb` is time-dependent.

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

Resolve the time-dependent fields of a naive optimiser for the fold that `ctx` describes.

The answer is a static optimiser, built through the keyword constructor, so the constructor checks every resolved value.

# Algorithm

 1. Return `opt` unchanged when [`is_time_dependent`](@ref) is `false`.
 2. Replace each [`TimeDependent`](@ref) field of `opt` with its value for the fold, with `update_time_dependent_fields`, giving `opt`.
 3. Resolve the fallback `opt.fb` for the same fold, and rebuild `opt` with it.

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

Return the static defaults of the naive-optimiser fields that can hold a [`TimeDependent`](@ref).

The constructors pass these defaults to [`assert_time_dependent_substitution`](@ref), and [`time_dependent_field_defaults`](@ref) returns them, so the value of a field outside a fold is written once. A field whose static default is `nothing` has no entry. [`RandomWeighted`](@ref) overrides the trait, because its `wb` default is `nothing`. [`InverseVolatility`](@ref) adds its prior estimator `pe`.

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

Replace every [`TimeDependent`](@ref) field of a naive optimiser with its static default, and do the same to its fallback `fb`.

A fold-less `optimise` calls it first, so the fit runs with each schedule at its static default.

# Algorithm

 1. Return `opt` unchanged when [`is_time_dependent`](@ref) is `false`.
 2. Replace each [`TimeDependent`](@ref) field of `opt` with its entry of [`time_dependent_field_defaults`](@ref), with `reset_time_dependent_fields`, giving `opt`.
 3. Reset the fallback `opt.fb` the same way, and rebuild `opt` with it.

# Related

  - [`NaiveOptimisationEstimator`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`naive_optimiser_td_defaults`](@ref)
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

Holds the weights, the bounds, the fee and the return code of a naive optimiser's fit.

Every subtype of [`NaiveOptimisationEstimator`](@ref) gives this result. `pr` is the prior result that [`InverseVolatility`](@ref) fits, or the returns data that a prior-free head reads.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NaiveOptimisationResult(;
        pr::Option{<:Pr_RR},
        wb::Option{<:WeightBounds}, fees::Option{<:Fees} = nothing,
        retcode::OptimisationReturnCode, w::Option{<:VecNum},
        imsk::Option{<:BitVector} = nothing, fb::Option{<:OptE_Opt_FbChain}
    ) -> NaiveOptimisationResult

Keywords correspond to the struct's fields. The keyword constructor expands `w` from the assets of `imsk` onto the full asset universe with [`expand_investable_weights`](@ref), and every [`_optimise`](@ref) of the family builds its result through it. The positional constructor never expands. `fees` is the fee of the head, on the universe that it solved on. A walk-forward fold charges it through [`extract_fees`](@ref), as it charges the fee of a hierarchical head. It is `nothing` for a head that carries no fee.

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

The variances are the diagonal of the covariance matrix that the prior estimator `pe` gives, so every choice inside `pe` reaches the weights. The inverse-variance weights, `sq = true`, are the naive risk parity allocation. [`HierarchicalRiskParity`](@ref) gives the assets of each half these weights under [`Variance`](@ref), and the inverse-volatility weights under [`StandardDeviation`](@ref), when it measures the risk of the half.

The weight finaliser `wf` then imposes the weight bounds, so the weights of the result equal the closed form below only when no bound binds.

# Mathematical definition

```math
\\begin{align}
w_i &= \\frac{\\sigma_i^{-1}}{\\sum_{j=1}^N \\sigma_j^{-1}} \\quad \\textrm{when } \\texttt{sq = false}\\,,\\\\
w_i &= \\frac{\\sigma_i^{-2}}{\\sum_{j=1}^N \\sigma_j^{-2}} \\quad \\textrm{when } \\texttt{sq = true}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_i_asset])
  - $(math_dict[:sigma_i_asset]) Its square ``\\sigma_i^2`` is entry ``i`` of the diagonal of the prior covariance matrix.
  - $(math_dict[:N])

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

Keywords correspond to the struct's fields. A field typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) per-fold schedule in place of a static value. The prior estimator, the weight bounds, the fees, the asset sets, the weight finaliser and the fallback define the problem, so a cross-validation fold loop resolves them for each fold. A fold-less `optimise` sets each of them to its static default. `sq`, `brt` and `strict` control the run and stay static. The result carries `fees`, and a walk-forward fold charges it. A turnover fee reads the previous weights that the loop writes through [`factory`](@ref).

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
  - `fb`: Recursively viewed via [`port_opt_view`](@ref), except a precomputed result, which is kept (see [`view_child`](@ref)).

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
    @fprop @vprop fb
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

It throws an `ArgumentError` when `opt.pe` holds an `AbstractPriorResult`, directly or through a [`TimeDependent`](@ref) schedule.

# Related

  - [`InverseVolatility`](@ref)
  - [`assert_external_optimiser`](@ref)
  - [`assert_estimated_prior`](@ref)
"""
function assert_external_optimiser(opt::InverseVolatility)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    assert_estimated_prior(opt.pe, "opt.pe")
    assert_internal_optimiser(opt)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the inverse volatility portfolio optimisation.

[`optimise`](@ref) calls this method. The weights follow the closed form of [`InverseVolatility`](@ref) on the assets of the Investable Mask, and every other asset holds a zero.

# Algorithm

 1. Set every [`TimeDependent`](@ref) field of `iv` to its static default, with [`reset_time_dependent_estimator`](@ref).
 2. Replace `rd` with its returns in excess of the benchmark when `iv.brt` is `true`, with `returns_result_picker`.
 3. Fit the prior `pr` with `iv.pe` on `rd`.
 4. Resolve the fee on the full universe of `rd`, and view it on the Investable Mask of `pr`, giving `fees`.
 5. Reduce `pr`, `iv` and `rd` to the Investable Mask with [`investable_reduction`](@ref), giving the mask `imsk`.
 6. Read the variances off the diagonal of `pr.sigma`, and take their inverse square roots, or their inverses when `iv.sq` is `true`, giving `w`.
 7. Divide `w` by its sum.
 8. Resolve the weight bounds `wb` over the reduced assets.
 9. Impose `wb` on `w` with the weight finaliser `iv.wf`, giving `retcode` and `w`.
10. Build the [`NaiveOptimisationResult`](@ref), which expands `w` onto the full universe.

# Related

  - [`InverseVolatility`](@ref)
  - [`investable_reduction`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(iv::InverseVolatility, rd::ReturnsResult = ReturnsResult(); kwargs...)
    iv = reset_time_dependent_estimator(iv)
    rd = returns_result_picker(rd, iv.brt)
    pr = prior(iv.pe, rd)
    # A weight, a bound and a fee hold fractions and infinities, so an integer sample takes
    # a float type for them, and every other sample keeps its own type.
    Tf = float_if_integer(eltype(pr.X))
    # Resolve the fee on the caller's own universe, before the door below narrows `sets`,
    # for the reason the hierarchical heads do; `investable_fees_view` then places it on
    # the axes the mask leaves.
    fees = investable_fees_view(fees_constraints(iv.fees, iv.sets; strict = iv.strict,
                                                 datatype = Tf), investable_mask(pr),
                                size(pr.X, 2))
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
    wb = weight_bounds_constraints(iv.wb, iv.sets; N = size(X, 2), strict = iv.strict,
                                   datatype = Tf)
    retcode, w = finalise_weight_bounds(iv.wf, wb, w)
    return NaiveOptimisationResult(; pr = pr, wb = wb, fees = fees, retcode = retcode,
                                   w = w, imsk = imsk, fb = nothing)
end
"""
    optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; kwargs...) -> NaiveOptimisationResult

Run the inverse volatility portfolio optimisation.

# Arguments

  - `iv`: The inverse volatility optimiser to use.
  - $(arg_dict[:rd]) When `iv.pe` is a prior result, the fit reads no returns from `rd`, and an empty `ReturnsResult()` is enough.
  - `kwargs`: Ignored.

# Validation

  - No field in the tree of `iv` holds an [`Online`](@ref). [`assert_batch_entry`](@ref) throws an `ArgumentError` that names the field otherwise. A plain `optimise` is a batch fit, and only the warm-up of the online arm of the fold loop resolves a wrapper.
"""
function optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; kwargs...)::NaiveOptimisationResult
    assert_batch_entry(iv, "`optimise`")
    return _optimise(iv, rd; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Allocates the same weight to every asset of the Coverage Universe of the window.

The optimiser fits no prior and reads no covariance matrix. It reads the asset count off the returns data.

``N`` is the size of the Coverage Universe of the window, not of the full asset universe. An asset is in it when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window. So an asset that is not yet listed, that is delisted, or that carries a stale finite price during an inactive spell holds a zero. The result carries the Coverage Universe as its `imsk`. A window in which no asset is covered throws an `IsEmptyError`.

The weight finaliser `wf` then imposes the weight bounds, so the weights of the result equal ``1/N`` only when no bound binds.

# Mathematical definition

```math
\\begin{align}
w_i &= \\frac{1}{N} \\quad \\forall i\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_i_asset])
  - $(math_dict[:N])

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

Keywords correspond to the struct's fields. A field typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) per-fold schedule in place of a static value. The weight bounds, the fees, the asset sets, the weight finaliser and the fallback define the problem, so a cross-validation fold loop resolves them for each fold. A fold-less `optimise` sets each of them to its static default. `strict` controls the run and stays static. The result carries `fees`, and a walk-forward fold charges it. A turnover fee reads the previous weights that the loop writes through [`factory`](@ref).

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
  - `fb`: Recursively viewed via [`port_opt_view`](@ref), except a precomputed result, which is kept (see [`view_child`](@ref)).

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
    @fprop @vprop fb
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

[`optimise`](@ref) calls this method. The head fits no prior, so no Prior Result gives it an Investable Mask. It takes the Coverage Universe of the window as its mask, and the result carries that mask as `imsk`, as the result of a prior-fitting head carries its Investable Mask.

# Algorithm

 1. Refuse a missing `rd.X` with an `IsNothingError`.
 2. Set every [`TimeDependent`](@ref) field of `ew` to its static default, with [`reset_time_dependent_estimator`](@ref).
 3. Resolve the fee on the full universe of `rd`, giving `fees`.
 4. Reduce `ew` and `rd` to the Coverage Universe with [`coverage_reduction`](@ref), giving the mask `cmsk`. A window in which no asset is covered throws an `IsEmptyError`.
 5. View `fees` on `cmsk`.
 6. Fill `w` with ``1/N`` over the ``N`` assets that `rd` keeps.
 7. Resolve the weight bounds `wb` over the kept assets.
 8. Impose `wb` on `w` with the weight finaliser `ew.wf`, giving `retcode` and `w`.
 9. Build the [`NaiveOptimisationResult`](@ref), which expands `w` onto the full universe.

# Related

  - [`EqualWeighted`](@ref)
  - [`coverage_reduction`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(ew::EqualWeighted, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    ew = reset_time_dependent_estimator(ew)
    # The fee is resolved on the caller's own universe before the door below narrows
    # `sets`, and placed on the axes the mask leaves after it.
    Nf = size(rd.X, 2)
    # A weight, a bound and a fee hold fractions and infinities, so an integer sample takes
    # a float type for them, and every other sample keeps its own type.
    Tf = float_if_integer(eltype(rd.X))
    fees = fees_constraints(ew.fees, ew.sets; strict = ew.strict, datatype = Tf)
    # This head fits no prior, so it derives the Coverage Universe of its own window and
    # reduces once, here: the weight bounds and the sets are stated over the full universe
    # and are viewed by the same index. `NaiveOptimisationResult` expands the weights back.
    cmsk, ew, rd = coverage_reduction(ew, rd)
    fees = investable_fees_view(fees, cmsk, Nf)
    N = size(rd.X, 2)
    # The weight takes the type of the data, not the `Float64` of `inv(N)`.
    w = fill(one(Tf) / N, N)
    wb = weight_bounds_constraints(ew.wb, ew.sets; N = N, strict = ew.strict, datatype = Tf)
    retcode, w = finalise_weight_bounds(ew.wf, wb, w)
    return NaiveOptimisationResult(; pr = rd, wb = wb, fees = fees, retcode = retcode,
                                   w = w, imsk = cmsk, fb = nothing)
end
"""
    optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; kwargs...) -> NaiveOptimisationResult

Run the equal-weighted portfolio optimisation.

# Arguments

  - `ew`: The equal-weighted optimiser to use.
  - $(arg_dict[:rd]) Its returns matrix and its Asset Panel give the Coverage Universe of the window, and the weights are spread over that universe.
  - `kwargs`: Ignored.

# Validation

  - `rd.X` is not `nothing`. The method throws an `IsNothingError` otherwise.
"""
function optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  kwargs...)::NaiveOptimisationResult
    return _optimise(ew, rd; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Draws portfolio weights at random from a Dirichlet distribution with concentration `alpha`.

A scalar `alpha` gives every asset the same concentration, so the draw is from the symmetric Dirichlet distribution. A vector `alpha` states one concentration per asset of the full universe. The draw takes the float type of `alpha`, not of the returns data, so the default `alpha = 1` gives `Float64` weights and `alpha = 1.0f0` gives `Float32` weights.

``N`` is the size of the Coverage Universe of the window, not of the full asset universe. An asset is in it when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window. So an asset that is not yet listed, that is delisted, or that carries a stale finite price during an inactive spell holds a zero. A vector `alpha` is sliced to the Coverage Universe. The result carries the Coverage Universe as its `imsk`. A window in which no asset is covered throws an `IsEmptyError`.

The weight finaliser `wf` then imposes the weight bounds. The default `wb` is `nothing`, so by default no bound binds and the weights of the result are the draw.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w} &\\sim \\mathrm{Dirichlet}(\\boldsymbol{\\alpha})\\,, \\\\
\\mathbb{E}[\\boldsymbol{w}] &= \\frac{\\boldsymbol{\\alpha}}{\\boldsymbol{1}^\\intercal \\boldsymbol{\\alpha}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: Portfolio weights vector, one entry per asset of the Coverage Universe.
  - $(math_dict[:alpha_dirichlet_conc])
  - $(math_dict[:N])

A Dirichlet draw is non-negative and sums to one. A larger ``\\boldsymbol{1}^\\intercal \\boldsymbol{\\alpha}`` concentrates the draws nearer their mean, which is ``1/N`` for every asset when all the entries of ``\\boldsymbol{\\alpha}`` are equal.

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

Keywords correspond to the struct's fields. A field typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) per-fold schedule in place of a static value. The weight bounds, the fees, the asset sets, the weight finaliser and the fallback define the problem, so a cross-validation fold loop resolves them for each fold. A fold-less `optimise` sets each of them to its static default, which is `nothing` for `wb`, `fees`, `sets` and `fb`. `alpha` is static. `rng`, `seed` and `strict` control the run and stay static. The result carries `fees`, and a walk-forward fold charges it. A turnover fee reads the previous weights that the loop writes through [`factory`](@ref).

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
  - `fb`: Recursively viewed via [`port_opt_view`](@ref), except a precomputed result, which is kept (see [`view_child`](@ref)).

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
    @fprop @vprop fb
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

[`optimise`](@ref) calls this method. The head fits no prior, so no Prior Result gives it an Investable Mask. It takes the Coverage Universe of the window as its mask, and the result carries that mask as `imsk`, as the result of a prior-fitting head carries its Investable Mask. A vector `alpha` states one concentration per asset of the full universe, so the method checks its length against the full width, before the reduction slices it.

# Algorithm

 1. Refuse a missing `rd.X` with an `IsNothingError`.
 2. Set every [`TimeDependent`](@ref) field of `rw` to its static default, with [`reset_time_dependent_estimator`](@ref).
 3. Refuse a vector `rw.alpha` whose length is not the width `Nf` of `rd.X`, with a `DimensionMismatch`.
 4. Resolve the fee on the full universe of `rd`, giving `fees`.
 5. Reduce `rw` and `rd` to the Coverage Universe with [`coverage_reduction`](@ref), giving the mask `cmsk`. [`port_opt_view`](@ref) slices a vector `rw.alpha` with the rest of `rw`. A window in which no asset is covered throws an `IsEmptyError`.
 6. View `fees` on `cmsk`.
 7. Build the Dirichlet distribution `dist` over the ``N`` kept assets, symmetric for a scalar `rw.alpha`.
 8. Draw `w` from `dist` with the generator that `resolve_rng` gives for `rw.rng` and `rw.seed`.
 9. Resolve the weight bounds `wb` over the kept assets.
10. Impose `wb` on `w` with the weight finaliser `rw.wf`, giving `retcode` and `w`.
11. Build the [`NaiveOptimisationResult`](@ref), which expands `w` onto the full universe.

# Related

  - [`RandomWeighted`](@ref)
  - [`coverage_reduction`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(rw::RandomWeighted, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    rw = reset_time_dependent_estimator(rw)
    Nf = size(rd.X, 2)
    if isa(rw.alpha, VecNum)
        # The caller states one concentration per asset of the full universe, so the check
        # reads the full width. The reduction below slices `alpha` with the rest.
        @argcheck(length(rw.alpha) == Nf,
                  DimensionMismatch("rw.alpha ($(length(rw.alpha))) must match N ($Nf)"))
    end
    # The fee is resolved on the caller's own universe before the door below narrows
    # `sets`, and placed on the axes the mask leaves after it.
    # A weight, a bound and a fee hold fractions and infinities, so an integer sample takes
    # a float type for them, and every other sample keeps its own type.
    Tf = float_if_integer(eltype(rd.X))
    fees = fees_constraints(rw.fees, rw.sets; strict = rw.strict, datatype = Tf)
    # This head fits no prior, so it derives the Coverage Universe of its own window and
    # reduces once, here: the concentrations, the weight bounds and the sets are stated over
    # the full universe and are viewed by the same index. `NaiveOptimisationResult` expands
    # the weights back.
    cmsk, rw, rd = coverage_reduction(rw, rd)
    fees = investable_fees_view(fees, cmsk, Nf)
    N = size(rd.X, 2)
    dist = if isa(rw.alpha, Number)
        Distributions.Dirichlet(N, rw.alpha)
    else
        Distributions.Dirichlet(rw.alpha)
    end
    rng = resolve_rng(rw.rng, rw.seed)
    w = rand(rng, dist)
    wb = weight_bounds_constraints(rw.wb, rw.sets; N = N, strict = rw.strict, datatype = Tf)
    retcode, w = finalise_weight_bounds(rw.wf, wb, w)
    return NaiveOptimisationResult(; pr = rd, wb = wb, fees = fees, retcode = retcode,
                                   w = w, imsk = cmsk, fb = nothing)
end
"""
    optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult; kwargs...) -> NaiveOptimisationResult

Run the random-weighted portfolio optimisation.

# Arguments

  - `rw`: The random-weighted optimiser to use.
  - $(arg_dict[:rd]) Its returns matrix and its Asset Panel give the Coverage Universe of the window, and the draw is over that universe.
  - `kwargs`: Ignored.

# Validation

  - `rd.X` is not `nothing`. The method throws an `IsNothingError` otherwise.
  - A vector `rw.alpha` has one entry per column of `rd.X`. The method throws a `DimensionMismatch` otherwise.
"""
function optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                     Nothing}, rd::ReturnsResult;
                  kwargs...)::NaiveOptimisationResult
    return _optimise(rw, rd; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Allocates the constant rebalanced portfolio that maximises the log wealth of its window, through Cover's fixed point and without a solver.

It is the Hindsight Comparator of the online selection family that needs no solver. Fit it on the rows that a strategy is scored on, and predict it in sample over the same rows. Its log wealth is then the largest that a constant rebalanced portfolio reaches on those rows, and [`log_wealth_regret`](@ref) measures a strategy against it. A follow-the-leader rule can also re-solve it on the rows that the rule selects. [`MeanRisk`](@ref)`(; obj = MaximumReturn(), opt = JuMPOptimiser(; slv, ret = LogarithmicReturn()))` gives the same portfolio to the tolerance of its solver, with one exponential cone per row.

The fixed point knows no bound other than the simplex. The weight finaliser `wf` imposes a `wb` that binds after the fixed point, so the weights of the result are then a repair of the unconstrained optimum, and not the constrained optimum. The default bounds are the simplex, and the finaliser leaves the fixed point unchanged under them. For a bounded or constrained Hindsight Comparator, state the bounds on the [`MeanRisk`](@ref) form above.

The head fits no prior. It reduces to the Coverage Universe of its window. An asset is kept when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row. Every other asset holds a zero, and the result carries the Coverage Universe as `imsk`.

The iteration stops when the certificate of the last line below is at most `tol * max(1, |log wealth|)`, or after `iters` steps. So `converged = true` means that the log wealth is within that tolerance of the optimum. The certificate bounds the log wealth, not the weights. The weight of an asset that the optimum does not hold falls by the factor ``g_i(\\boldsymbol{b}^{\\star})`` at each step. When that multiplier is near one, the default budget can stop first with `converged = false`. The `retcode.res` of the result holds `converged`, `iterations` and the certificate `gap`.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{b}^{\\star} &= \\underset{\\boldsymbol{b} \\in \\Delta_N}{\\arg\\max} \\log S_T(\\boldsymbol{b}) = \\underset{\\boldsymbol{b} \\in \\Delta_N}{\\arg\\max} \\sum_{t=1}^{T} \\log \\langle \\boldsymbol{b}, \\boldsymbol{x}_t \\rangle\\,, \\\\
g_i(\\boldsymbol{b}) &= \\frac{1}{T} \\sum_{t=1}^{T} \\frac{x_{t,i}}{\\langle \\boldsymbol{b}, \\boldsymbol{x}_t \\rangle}\\,, \\\\
b_i^{(k+1)} &= b_i^{(k)} \\, g_i\\left(\\boldsymbol{b}^{(k)}\\right)\\,, \\\\
\\log S_T(\\boldsymbol{b}^{\\star}) - \\log S_T(\\boldsymbol{b}) &\\leq T \\log \\max_i g_i(\\boldsymbol{b})\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{b}^{\\star}``: Best constant rebalanced portfolio, the weights of the result when no bound binds.
  - $(math_dict[:b_crp])
  - $(math_dict[:Delta_N_simplex])
  - $(math_dict[:S_t_crp])
  - $(math_dict[:x_t_rel]) Its entry ``x_{t,i}`` is one plus the return of asset ``i``.
  - $(math_dict[:T])
  - $(math_dict[:N])
  - ``g_i(\\boldsymbol{b})``: Multiplier of asset ``i`` at ``\\boldsymbol{b}``, the mean over the periods of the price relative of asset ``i`` divided by the gross return of ``\\boldsymbol{b}``.
  - ``\\boldsymbol{b}^{(k)}``: Iterate ``k`` of Cover's map.

The objective is concave on the simplex, so its maximum is global. The multipliers satisfy ``\\sum_i b_i g_i(\\boldsymbol{b}) = 1``, so Cover's map keeps an iterate on the simplex. The map is a minorise-maximise step, so ``\\log S_T`` does not fall from one iterate to the next. An entry at zero stays at zero. At ``\\boldsymbol{b}^{\\star}`` every held asset has ``g_i = 1`` and every other asset has ``g_i \\leq 1``, which is the first-order condition. The last line holds at every ``\\boldsymbol{b}`` in the simplex, by concavity and Jensen's inequality, and its right side is zero at ``\\boldsymbol{b}^{\\star}``, a corner included.

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

Keywords correspond to the struct's fields. A field typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) per-fold schedule in place of a static value. The weight bounds, the fees, the asset sets, the weight finaliser and the fallback define the problem, so a cross-validation fold loop resolves them for each fold. A fold-less `optimise` sets each of them to its static default. `iters`, `tol` and `strict` control the run and stay static. The result carries `fees`, and a walk-forward fold charges it. A turnover fee reads the previous weights that the loop writes through [`factory`](@ref).

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
  - `fb`: Recursively viewed via [`port_opt_view`](@ref), except a precomputed result, which is kept (see [`view_child`](@ref)).

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
  - [`cover_fixed_point`](@ref)
  - [`MeanRisk`](@ref): Reaches the same portfolio with a solver under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref), and takes bounds.
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
    @fprop @vprop fb
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

It is the kernel of [`BestConstantRebalancedPortfolio`](@ref), on a bare matrix, so a rule that re-solves the portfolio can call it directly. The map, the multipliers ``g_i`` and the certificate are the ones that [`BestConstantRebalancedPortfolio`](@ref) states.

# Algorithm

 1. Start from the uniform portfolio `w`, ``1/N`` for every asset, so that every asset can take weight.
 2. Compute the gross returns `p = X * w`, the log wealth `lw`, the multipliers `g` as `X' * inv.(p) / T`, and the certificate `gap`, ``T \\log \\max_i g_i`` clamped at zero.
 3. Stop, with `converged = true`, when `gap` is at most `tol * max(1, abs(lw))`.
 4. Stop, with `converged = false`, after `iters` steps.
 5. Otherwise multiply `w` by `g` entry by entry, divide it by its sum to remove the round-off, add one to `iterations`, and go back to step 2.

# Arguments

  - `X`: Price relatives, `observations × assets`, every entry positive.
  - `iters`: Maximum number of iterations.
  - `tol`: Relative tolerance on the certificate of the shortfall in log wealth.

# Returns

  - `fp::NamedTuple`: Five entries.

      + `w`: The weights, on the simplex.
      + `log_wealth`: ``\\sum_t \\log(\\boldsymbol{x}_t^{\\intercal} \\boldsymbol{w})`` at `w`.
      + `gap`: The certificate at `w`, an upper bound on the optimal log wealth minus `log_wealth`.
      + `converged`: Whether `gap` met the tolerance.
      + `iterations`: The number of multiplicative steps, zero when the uniform start already meets the tolerance.

# Related

  - [`BestConstantRebalancedPortfolio`](@ref)
"""
function cover_fixed_point(X::MatNum, iters::Integer, tol::Number)
    T, N = size(X)
    w = fill(one(eltype(X)) / N, N)
    # The certificate is read at `w`, the weights the kernel returns, before each step, so the
    # uniform start is certified before the loop. Cover's multiplier of each asset is the mean
    # over observations of its price relative against the portfolio's own. By concavity and
    # Jensen's inequality the largest one bounds the shortfall,
    # `lw* - lw <= T log(max_j g_j)`, and the bound reaches zero at the optimum, a corner
    # included, where every multiplier is at most one.
    p = X * w
    lw = sum(log, p)
    # The mean over observations of `X[t, i] / p[t]` is one matrix-vector product, so no
    # observations-by-assets temporary is built at each step.
    g = X' * inv.(p) / T
    gap = max(zero(lw), T * log(maximum(g)))
    converged = gap <= tol * max(one(lw), abs(lw))
    iterations = 0
    while !converged && iterations < iters
        # The multiplicative step, renormalised onto the simplex, then the certificate at the
        # new weights.
        w = w .* g
        w ./= sum(w)
        p = X * w
        lw = sum(log, p)
        g = X' * inv.(p) / T
        gap = max(zero(lw), T * log(maximum(g)))
        converged = gap <= tol * max(one(lw), abs(lw))
        iterations += 1
    end
    return (; w = w, log_wealth = lw, gap = gap, converged = converged,
            iterations = iterations)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the best constant rebalanced portfolio optimisation.

[`optimise`](@ref) calls this method. The return code is the one of the weight finaliser. A successful return code holds the `converged` flag, the iteration count and the certificate `gap` of the fixed point in its `res`, so the caller reads a run that stopped at `iters` off the result, and the method throws nothing for it.

# Algorithm

 1. Refuse a missing `rd.X` with an `IsNothingError`.
 2. Set every [`TimeDependent`](@ref) field of `bcrp` to its static default, with [`reset_time_dependent_estimator`](@ref).
 3. Resolve the fee on the full universe of `rd`, giving `fees`.
 4. Reduce `bcrp` and `rd` to the Coverage Universe with [`coverage_reduction`](@ref), giving the mask `cmsk`. A window in which no asset is covered throws an `IsEmptyError`.
 5. View `fees` on `cmsk`.
 6. Form the price relatives `X`, one plus `rd.X`.
 7. Run [`cover_fixed_point`](@ref) on `X` with `bcrp.iters` and `bcrp.tol`, giving `fp`.
 8. Resolve the weight bounds `wb` over the kept assets.
 9. Impose `wb` on `fp.w` with the weight finaliser `bcrp.wf`, giving `retcode` and `w`.
10. When `retcode` is an [`OptimisationSuccess`](@ref), replace it with one whose `res` holds `fp.converged`, `fp.iterations` and `fp.gap`.
11. Build the [`NaiveOptimisationResult`](@ref), which expands `w` onto the full universe.

# Related

  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`cover_fixed_point`](@ref)
  - [`coverage_reduction`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(bcrp::BestConstantRebalancedPortfolio, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    bcrp = reset_time_dependent_estimator(bcrp)
    # The fee is resolved on the caller's own universe before the door below narrows
    # `sets`, and placed on the axes the mask leaves after it.
    Nf = size(rd.X, 2)
    # A weight, a bound and a fee hold fractions and infinities, so an integer sample takes
    # a float type for them, and every other sample keeps its own type.
    Tf = float_if_integer(eltype(rd.X))
    fees = fees_constraints(bcrp.fees, bcrp.sets; strict = bcrp.strict, datatype = Tf)
    # This head fits no prior, so it derives the Coverage Universe of its own window and
    # reduces once, here, as `EqualWeighted` does. `NaiveOptimisationResult` expands back.
    cmsk, bcrp, rd = coverage_reduction(bcrp, rd)
    fees = investable_fees_view(fees, cmsk, Nf)
    X = one(eltype(rd.X)) .+ rd.X
    N = size(X, 2)
    fp = cover_fixed_point(X, bcrp.iters, bcrp.tol)
    wb = weight_bounds_constraints(bcrp.wb, bcrp.sets; N = N, strict = bcrp.strict,
                                   datatype = Tf)
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
             rd::ReturnsResult; kwargs...) -> NaiveOptimisationResult

Run the best constant rebalanced portfolio optimisation.

A Hindsight Comparator is fit on the rows that it is scored on. So `predict(optimise(bcrp, rd_test), rd_test)` is the prediction result of the comparator over `rd_test`, and `cross_val_predict(bcrp, rd, cv)` is the causal constant portfolio over the same rows.

# Arguments

  - `bcrp`: The best constant rebalanced portfolio optimiser to use.
  - $(arg_dict[:rd]) Its returns matrix and its Asset Panel give the Coverage Universe of the window, and the fixed point runs over that universe.
  - `kwargs`: Ignored.

# Validation

  - `rd.X` is not `nothing`. The method throws an `IsNothingError` otherwise.
"""
function optimise(bcrp::BestConstantRebalancedPortfolio{<:Any, <:Any, <:Any, <:Any,
                                                        Nothing}, rd::ReturnsResult;
                  kwargs...)::NaiveOptimisationResult
    return _optimise(bcrp, rd; kwargs...)
end

"""
$(DocStringExtensions.TYPEDEF)

Holds the weights it was handed, and solves nothing.

The fold loop writes the weights of the previous fold into `w` through [`factory`](@ref), as it writes them into a [`TurnoverEstimator`](@ref). The head returns `w` unchanged, on the full asset universe. It fits no prior, takes no Coverage Universe and imposes no weight bounds, so nothing rewrites a hold.

Its main use is as the fallback `fb` of an optimiser inside a walk-forward. A failed solve then holds the book, and the fold keeps finite weights in place of `NaN` weights. The head still holds a weight on an asset that left the panel, and the returns of that asset are zeroed as a Held Gap. It is also a primary optimiser: `PreviousWeights(; w = w)` is a walk-forward that holds `w` on every fold.

A view to an asset subset, such as a cluster of [`NestedClustered`](@ref) or the asset subset of a [`MultipleRandomised`](@ref) fold, slices `w` to the subset. The head holds that slice as it is, and does not rescale it to the budget of the subset.

The constructor refuses a `w` with a non-finite entry. A fit with `w = nothing` gives an [`OptimisationFailure`](@ref). Fold 1 of a walk-forward and a fold-less `optimise` with no `w` meet that case. [`needs_previous_weights`](@ref) is `true`, so an optimiser that carries the head as a fallback runs its folds in sequence, and the loop writes the previous weights into it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PreviousWeights(; w::Option{<:VecNum} = nothing, fees::Option{<:Fees} = nothing, fb::TDO_Option{<:OptE_Opt} = nothing) -> PreviousWeights

Keywords correspond to the struct's fields. `fb` can hold a [`TimeDependent`](@ref) per-fold schedule. `fees` is a resolved [`Fees`](@ref), never an estimator, because the head holds no `sets` to resolve an estimator over. The result carries `fees`, and a walk-forward fold charges it. Under a Previous-Weights Source, its turnover term prices the rebalance from the drifted book back to the held target.

## Validation

  - `w`: `all(isfinite, w)` when `w` is not `nothing`. The constructor throws a `DomainError` otherwise.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `fees`: Recursively viewed via [`port_opt_view`](@ref).
  - `fb`: Recursively viewed via [`port_opt_view`](@ref), except a precomputed result, which is kept (see [`view_child`](@ref)).

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
    @vprop w
    """
    $(field_dict[:fees_res])
    """
    @fprop @vprop fees
    """
    $(field_dict[:fb])
    """
    @fprop @vprop fb
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

[`optimise`](@ref) calls this method. It reads only the width of `rd`. The weights of the result are `w` unchanged, on the universe that the loop wrote them on, with `imsk = nothing` and no weight bounds.

A head whose `w` is `nothing` gives an [`OptimisationFailure`](@ref) that names the missing weights, so a fallback chain that reaches it goes on to the next fallback. Its weights are then `NaN` at every column of `rd.X`, as the weights of every failed solve are, so a fold reads this failure as it reads any other. They are `nothing` when `rd.X` is `nothing`.

# Algorithm

 1. Set every [`TimeDependent`](@ref) field of `pw` to its static default, with [`reset_time_dependent_estimator`](@ref).
 2. With `w` set, give a [`NaiveOptimisationResult`](@ref) that carries `w` and an [`OptimisationSuccess`](@ref).
 3. With `w` unset, give one that carries an [`OptimisationFailure`](@ref) and the weights of [`failed_hold_weights`](@ref).

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

Return the weights of a [`PreviousWeights`](@ref) that has nothing to hold.

The weights are `NaN` at every column of `X`, or `nothing` when `X` is `nothing`. They take the element type of `X`, or a float type when `X` holds integers, because an integer cannot hold `NaN`.

# Related

  - [`PreviousWeights`](@ref)
  - [`_optimise(pw::PreviousWeights, rd::ReturnsResult = ReturnsResult(); kwargs...)`](@ref)
"""
function failed_hold_weights(::Nothing)
    return nothing
end
function failed_hold_weights(X::MatNum)
    # An integer sample cannot hold `NaN`, so it takes a float type, and every other sample
    # keeps its own type.
    return fill(convert(float_if_integer(eltype(X)), NaN), size(X, 2))
end
"""
    optimise(pw::PreviousWeights{<:Any, <:Any, Nothing}, rd::ReturnsResult = ReturnsResult();
             kwargs...) -> NaiveOptimisationResult

Hold the weights the head carries.

# Arguments

  - `pw`: The hold-only head.
  - $(arg_dict[:rd]) The result records it as `pr`. The head reads only the width of `rd.X`, and only when `pw.w` is `nothing`.
  - `kwargs`: Ignored.
"""
function optimise(pw::PreviousWeights{<:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); kwargs...)::NaiveOptimisationResult
    return _optimise(pw, rd; kwargs...)
end
export NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeighted,
       PreviousWeights, BestConstantRebalancedPortfolio
