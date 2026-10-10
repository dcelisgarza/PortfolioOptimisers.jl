"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the optimisers that combine several inner optimisers through one outer optimiser.

A stacking optimiser treats the portfolio of each inner optimiser as one synthetic asset, and an outer optimiser allocates across these synthetic assets. So the returns of the inner portfolios set their weights, and no fixed rule averages them. This is the stacked generalisation of [wolpert1992](@cite), with portfolios in place of predictors.

No method of the library dispatches on this type, so a subtype implements no method for it. [`Stacking`](@ref) is its one subtype.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`Stacking`](@ref)
  - [`StackingResult`](@ref)

# References

  - $(ref_dict[:wolpert1992])
"""
abstract type BaseStackingOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Holds the inner results, the outer result and the stacked weights of a stacking optimisation.

[`optimise`](@ref) returns it for a [`Stacking`](@ref). `resi` holds one result for each inner optimiser, in the order of `opti`. `reso` is the result of the outer optimiser on the synthetic assets, so `reso.w` holds one entry for each inner optimiser, not one for each asset. `w` holds the stacked weights over the assets.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StackingResult(;
        pr::Option{<:AbstractPriorResult},
        wb::Option{<:WeightBounds},
        fees::Option{<:Fees},
        resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
        reso::OptimisationResult,
        cv::Option{<:OptimisationCrossValidation},
        retcode::OptRetCode_VecOptRetCode,
        w::VecNum_VecVecNum,
        imsk::Option{<:BitVector} = nothing,
        fb::Option{<:OptE_Opt_FbChain}
    ) -> StackingResult

Keywords correspond to the struct's fields. The keyword constructor expands `w` from the assets of the Investable Mask `imsk` onto the full asset universe, through [`expand_investable_weights`](@ref). An asset outside the mask gets a zero weight. [`_optimise`](@ref) returns through this constructor. The positional constructor does not expand `w`, so [`set_retcode`](@ref) and [`factory`](@ref) rebuild a result and do not expand it a second time.

# Related

  - [`Stacking`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`NestedClusteredResult`](@ref)
  - [`combination_weights`](@ref)
  - [`expand_investable_weights`](@ref)

# References

  - $(ref_dict[:wolpert1992])
"""
@concrete struct StackingResult <: NonJuMPOptimisationResult
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:fees])
    """
    fees
    """
    $(field_dict[:resi])
    """
    resi
    """
    $(field_dict[:reso])
    """
    reso
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    Stacked portfolio weights over the full asset universe, after the weight finaliser. An outer efficient frontier gives one weight vector for each point.
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
    function StackingResult(pr::Option{<:AbstractPriorResult}, wb::Option{<:WeightBounds},
                            fees::Option{<:Fees},
                            resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                            reso::OptimisationResult,
                            cv::Option{<:OptimisationCrossValidation},
                            retcode::OptRetCode_VecOptRetCode, w::VecNum_VecVecNum,
                            imsk::Option{<:BitVector}, fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(pr), typeof(wb), typeof(fees), typeof(resi), typeof(reso),
                   typeof(cv), typeof(retcode), typeof(w), typeof(imsk), typeof(fb)}(pr, wb,
                                                                                     fees,
                                                                                     resi,
                                                                                     reso,
                                                                                     cv,
                                                                                     retcode,
                                                                                     w,
                                                                                     imsk,
                                                                                     fb)
    end
end
function StackingResult(; pr::Option{<:AbstractPriorResult}, wb::Option{<:WeightBounds},
                        fees::Option{<:Fees},
                        resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                        reso::OptimisationResult, cv::Option{<:OptimisationCrossValidation},
                        retcode::OptRetCode_VecOptRetCode, w::VecNum_VecVecNum,
                        imsk::Option{<:BitVector} = nothing,
                        fb::Option{<:OptE_Opt_FbChain})::StackingResult
    return StackingResult(pr, wb, fees, resi, reso, cv, retcode,
                          expand_investable_weights(imsk, w), imsk, fb)
end
# The stacking family carries the mask on the result itself, so the fold reads it directly.
function result_investable_mask(res::StackingResult)
    return res.imsk
end
"""
    set_retcode(res::StackingResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`StackingResult`](@ref) with a different return code.

An outer efficient frontier gives the result one return code for each point, and a fold drops a point when it replaces the code of that point with a failure. The method keeps every other field of `res`.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one return code for each point of an efficient frontier.

# Returns

  - [`StackingResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`StackingResult`](@ref)
"""
function set_retcode(res::StackingResult, retcode::OptRetCode_VecOptRetCode)
    return StackingResult(res.pr, res.wb, res.fees, res.resi, res.reso, res.cv, retcode,
                          res.w, res.imsk, res.fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`Stacking`](@ref) fields that can hold a [`TimeDependent`](@ref) schedule.

The constructor reads them when it tests each entry of a schedule, and [`time_dependent_field_defaults`](@ref) returns them. `opti` and `opto` are required and have no static default, so their entry is [`NoDefault`](@ref). A schedule in one of them must carry its own `default`, or it cannot run outside a fold loop. The entries of `pe` and `wf` are their keyword defaults. The tuple holds no entry for `wb`, `fees`, `sets`, `scale` and `fb`, because their static default is `nothing`.

# Returns

  - `NamedTuple`: The defaults of `pe`, `opti`, `opto` and `wf`.

# Related

  - [`Stacking`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function stacking_td_defaults()::NamedTuple
    return (; pe = EmpiricalPrior(), opti = NoDefault(), opto = NoDefault(),
            wf = IterativeWeightFinaliser())
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Narrow the element type of a vector of inner optimisers, so that the [`Stacking`](@ref) constructor accepts it.

A literal such as `[MeanRisk(), TimeDependent(…)]` infers the element type `AbstractEstimator`. The [`VecOptE_Opt_TD`](@ref) bound refuses that type, but it accepts each element. The keyword constructor of [`Stacking`](@ref) calls this method on `opti`.

# Algorithm

 1. When `opti` satisfies [`VecOptE_Opt_TD`](@ref), return it unchanged.
 2. Check that every element of `opti` is an [`OptE_Opt_TD`](@ref).
 3. Convert `opti` to a vector whose element type is the union of the types of its elements.

The method for a [`TimeDependent`](@ref) returns the schedule unchanged.

# Arguments

  - `opti`: Vector of inner optimisers, or a [`TimeDependent`](@ref) schedule for the whole field.

# Validation

  - Every element of `opti` is an [`OptE_Opt_TD`](@ref). The method throws an `ArgumentError` otherwise.

# Returns

  - The vector with the narrowed element type, or `opti` itself.

# Related

  - [`VecOptE_Opt_TD`](@ref)
  - [`Stacking`](@ref)
"""
function narrow_optimiser_vector(opti::AbstractVector)
    if isa(opti, VecOptE_Opt_TD)
        return opti
    end
    @argcheck(all(x -> isa(x, OptE_Opt_TD), opti),
              ArgumentError("every element of opti must be an optimisation estimator, a precomputed optimisation result, or a TimeDependent schedule standing in for one"))
    return convert(Vector{Union{unique(typeof.(opti))...}}, opti)
end
function narrow_optimiser_vector(opti::TimeDependent)
    return opti
end
"""
$(DocStringExtensions.TYPEDEF)

Combines several inner optimisers through one outer optimiser that allocates across their portfolios.

`Stacking` solves each inner optimiser of `opti` on the whole sample. The portfolio of each inner optimiser is one synthetic asset, and the outer optimiser `opto` allocates across these synthetic assets. The stacked weights are the inner weights, combined by the outer weights. This is the stacked generalisation of [wolpert1992](@cite), with portfolios in place of predictors.

`cv` selects the returns of the synthetic assets that `opto` reads. With `cv`, they are out-of-sample returns. Each observation comes from inner solves that did not see it, which is the level-one data of [wolpert1992](@cite). Without `cv`, they are the in-sample returns of the full-sample inner solves. Then `opto` reads the observations that fitted the inner weights, and it does not see the returns of an inner optimiser on data that it did not fit.

# Mathematical definition

Without `cv`:

```math
\\begin{align}
\\mathbf{R}_{\\cdot k} &= \\mathbf{X} \\mathbf{W}_{\\cdot k} - F(\\mathbf{W}_{\\cdot k})\\,,\\\\
\\boldsymbol{v} &= \\mathcal{O}(\\mathbf{R})\\,,\\\\
c_k &= \\begin{cases}
\\dfrac{s_k v_k}{\\sum_{j=1}^{K} s_j v_j} \\sum_{j=1}^{K} v_j & \\text{if } \\sum_{j=1}^{K} s_j v_j \\neq 0 \\text{ and } \\sum_{j=1}^{K} v_j \\neq 0\\,,\\\\
s_k v_k & \\text{otherwise}\\,,
\\end{cases}\\\\
\\boldsymbol{w} &= \\mathbf{W} \\boldsymbol{c}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{R}``: Returns matrix of the synthetic assets, ``T \\times K``, with column ``k`` for inner optimiser ``k``.
  - $(math_dict[:X_returns])
  - $(math_dict[:W_inner]) Sub-portfolio ``k`` is inner optimiser ``k``, solved on the whole sample.
  - $(math_dict[:F_fee_series]) Here ``\\boldsymbol{w}`` is ``\\mathbf{W}_{\\cdot k}``.
  - ``\\mathcal{O}``: The outer optimiser, which maps the returns matrix of the synthetic assets to their weights.
  - $(math_dict[:v_outer])
  - $(math_dict[:c_k_comb])
  - $(math_dict[:s_k_comb]) Without a Combination Weight, ``\\boldsymbol{c} = \\boldsymbol{v}``.
  - $(math_dict[:K_sub]) Here it is the number of inner optimisers.
  - ``\\boldsymbol{w}``: Stacked weights over the ``N`` assets.
  - $(math_dict[:T])
  - $(math_dict[:N])

With `cv`, row ``t`` of ``\\mathbf{R}_{\\cdot k}`` is the return at observation ``t`` of inner optimiser ``k``, solved on the training observations of the fold whose test window holds ``t``. Then ``\\mathbf{R}`` has one row for each observation that a test window holds. ``\\mathbf{W}`` is the full-sample matrix in both cases.

``\\mathcal{O}`` reads ``\\mathbf{R}``, which does not depend on ``s_k``. So the Combination Weight acts at the combination alone, and a run with `cv` applies it in the same way as a run without `cv`. The ratios ``c_k / c_j = s_k v_k / (s_j v_j)`` hold in both cases of ``c_k``, and the first case keeps the total, ``\\sum_k c_k = \\sum_k v_k``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Stacking(;
        pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
        wb::TD_Option{<:WbE_Wb} = nothing,
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        scale::TD_Option{<:VecNum} = nothing,
        opti::Union{<:AbstractVector, <:TD_VecOptE_Opt},
        opto::OptE_TD,
        cv::Option{<:OptimisationCrossValidation} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        brt::Bool = false,
        strict::Bool = false,
        pcol::AbstractPanelCollapseAlgorithm = RenormaliseActive(),
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> Stacking

Keywords correspond to the struct's fields.

## Time-dependent fields

`pe`, `wb`, `fees`, `sets`, `scale`, `wf`, `opto` and `fb` can hold a [`TimeDependent`](@ref) schedule. No inner fold loop of `Stacking` reads them, so the fold loop that reaches the `Stacking` resolves them. A schedule in `opto` or `fb` must bind `:outermost`. `opti` takes a schedule at two levels:

  - **Element**, as in `opti = [static, TimeDependent(…)]`. The inner cross-validation solves each element on its own, so an element schedule can bind `:nearest`. Such a schedule needs an explicit `default` and a `cv` that is not `nothing`, because the full-sample solve of the element always resolves it to its `default`. See [`assert_nearest_optimiser_schedule`](@ref).
  - **Field**, as in `opti = TimeDependent([[…], […]])`, one vector of inner optimisers for each fold. See [`TD_VecOptE_Opt`](@ref). The schedule must bind `:outermost`. The inner cross-validation receives the elements and never the field, and a vector that changes with the fold changes the number and the order of the synthetic assets that `opto` reads.

## Validation

  - When `opti` is a vector: `!isempty(opti)`. The constructor throws an `IsEmptyError` otherwise.
  - Every element of `opti` is an [`OptE_Opt_TD`](@ref). The keyword constructor throws an `ArgumentError` otherwise, through [`narrow_optimiser_vector`](@ref).
  - An element schedule of `opti` that binds `:nearest` has an explicit `default`, and `cv !== nothing`. The constructor throws a [`TimeDependentDefaultError`](@ref) or an `ArgumentError` otherwise, through [`assert_nearest_optimiser_schedule`](@ref).
  - When `opti` is a [`TimeDependent`](@ref): `opti.bind !== :nearest`. The constructor throws an `ArgumentError` otherwise.
  - When `scale` is a vector: `all(isfinite, scale)`, and `length(scale) == length(opti)` when `opti` is a vector. The constructor throws an `IsNonFiniteError` or a `DimensionMismatch` otherwise.
  - A schedule in `opto` or `fb` has `bind !== :nearest`. The constructor throws an `ArgumentError` otherwise.
  - `opto` passes [`assert_external_optimiser`](@ref), and every element of `opti` passes it when `cv !== nothing`. The constructor throws an `ArgumentError` otherwise. Such an optimiser runs on the synthetic assets or on the training rows of a fold, which it does not know in advance. So it cannot hold a precomputed prior or a precomputed constraint.
  - When `wb` is a [`WeightBoundsEstimator`](@ref) or `fees` is a [`FeesEstimator`](@ref): `!isnothing(sets)`. The constructor throws an `IsNothingError` otherwise.
  - Each entry and the `default` of each schedule give a valid `Stacking`. [`assert_time_dependent_substitution`](@ref) runs the constructor on each of them.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `opti`: Recursively updated via [`factory`](@ref).
  - `opto`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

`Stacking` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `pe` holds a prior result, the method replaces `X` with `pe.X`, so it views the children against the observations of that prior.
  - `pe`, `wb`, `sets` and `cache` recurse through [`port_opt_view`](@ref) with the index alone.
  - `fees`, `opti` and `opto` recurse through [`port_opt_view`](@ref) with that matrix.
  - `fb` recurses through [`view_child`](@ref) with that matrix.
  - `scale` stays unchanged. It holds one entry for each inner optimiser, not one for each asset.

# Related

  - [`optimise`](@ref)
  - [`StackingResult`](@ref)
  - [`BaseStackingOptimisationEstimator`](@ref)
  - [`NestedClustered`](@ref)
  - [`combination_weights`](@ref)
  - [`predict_outer_returns`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:wolpert1992])
"""
@propagatable @concrete struct Stacking <: BaseStackingOptimisationEstimator
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:wb_jmp])
    """
    wb
    """
    $(field_dict[:feese])
    """
    @fprop fees
    """
    $(field_dict[:sets])
    """
    sets
    """
    Combination Weight of the inner optimisers, one entry for each element of `opti`, or `nothing`. Entry ``k`` is the weight that inner optimiser ``k`` carries in the combination that the answer of `opto` defines. Only the ratios between the entries matter, because [`combination_weights`](@ref) rescales the tilted coefficients to the total of `opto`. So a common factor cancels. A uniform weight and the weight of a lone inner optimiser change nothing. `nothing` keeps the answer of `opto`.
    """
    scale
    """
    $(field_dict[:opti])
    """
    @fprop opti
    """
    $(field_dict[:opto])
    """
    @fprop opto
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:fb])
    """
    @fprop fb
    """
    $(field_dict[:brt])
    """
    brt
    """
    $(field_dict[:strict_opt])
    """
    strict
    """
    $(field_dict[:pcol])
    """
    pcol
    """
    $(field_dict[:cache_opt])
    """
    @fprop cache
    function Stacking(pe::Onl{<:TD{<:PrE_Pr}}, wb::TD_Option{<:WbE_Wb},
                      fees::TD_Option{<:FeesE_Fees}, sets::TD_Option{<:UniverseSets},
                      scale::TD_Option{<:VecNum},
                      opti::Union{<:VecOptE_Opt_TD, <:TD_VecOptE_Opt}, opto::OptE_TD,
                      cv::Option{<:OptimisationCrossValidation}, wf::TD{<:WeightFinaliser},
                      ex::FLoops.Transducers.Executor, fb::TDO_Option{<:OptE_Opt},
                      brt::Bool, strict::Bool, pcol::AbstractPanelCollapseAlgorithm,
                      cache::Option{<:ReturnsBufferState})
        if isa(opti, TimeDependent)
            @argcheck(opti.bind !== :nearest,
                      ArgumentError("opti of Stacking cannot hold a `bind = :nearest` schedule at the field level: Stacking's inner cross-validation is entered per candidate (`cross_val_predict(opti[k], …)`), so the fold loop is handed the elements, never the field — and a per-fold candidate vector would change the number and identity of the returns-proxy columns opto sees. Schedule individual elements instead (`opti = [static, TimeDependent(…, :nearest; default = …)]`), or use `bind = :outermost` to vary the whole vector with the fold loop that reaches the Stacking."))
        else
            @argcheck(!isempty(opti), IsEmptyError("opti cannot be empty"))
            for (i, x) in pairs(opti)
                assert_nearest_optimiser_schedule(x, Symbol("opti[$i]"), cv, :Stacking)
            end
        end
        if !isnothing(scale) && !isa(scale, TimeDependent)
            if isa(opti, AbstractVector)
                @argcheck(length(scale) == length(opti),
                          DimensionMismatch("scale ($(length(scale))) must match opti ($(length(opti)))"))
            end
            @argcheck(all(isfinite, scale),
                      IsNonFiniteError("all elements of scale must be finite"))
        end
        assert_no_nearest_bind_optimiser_schedule(opto, :opto, :Stacking)
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :Stacking)
        assert_external_optimiser(opto)
        if !isnothing(cv)
            assert_external_optimiser(opti)
        end
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        assert_time_dependent_substitution(Stacking,
                                           (; pe, wb, fees, sets, scale, opti, opto, cv, wf,
                                            ex, fb, brt, strict, pcol),
                                           stacking_td_defaults())
        return new{typeof(pe), typeof(wb), typeof(fees), typeof(sets), typeof(scale),
                   typeof(opti), typeof(opto), typeof(cv), typeof(wf), typeof(ex),
                   typeof(fb), typeof(brt), typeof(strict), typeof(pcol), typeof(cache)}(pe,
                                                                                         wb,
                                                                                         fees,
                                                                                         sets,
                                                                                         scale,
                                                                                         opti,
                                                                                         opto,
                                                                                         cv,
                                                                                         wf,
                                                                                         ex,
                                                                                         fb,
                                                                                         brt,
                                                                                         strict,
                                                                                         pcol,
                                                                                         cache)
    end
end
function Stacking(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
                  wb::TD_Option{<:WbE_Wb} = nothing,
                  fees::TD_Option{<:FeesE_Fees} = nothing,
                  sets::TD_Option{<:UniverseSets} = nothing,
                  scale::TD_Option{<:VecNum} = nothing,
                  opti::Union{<:AbstractVector, <:TD_VecOptE_Opt}, opto::OptE_TD,
                  cv::Option{<:OptimisationCrossValidation} = nothing,
                  wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                  ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                  fb::TDO_Option{<:OptE_Opt} = nothing, brt::Bool = false,
                  strict::Bool = false,
                  pcol::AbstractPanelCollapseAlgorithm = RenormaliseActive(),
                  cache::Option{<:ReturnsBufferState} = nothing)::Stacking
    return Stacking(pe, wb, fees, sets, scale, narrow_optimiser_vector(opti), opto, cv, wf,
                    ex, fb, brt, strict, pcol, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse a precomputed result among the inner optimisers of a [`Stacking`](@ref) that a [`NestedClustered`](@ref) holds.

[`NestedClustered`](@ref) solves its inner optimiser on each cluster, and its outer optimiser on the synthetic assets of the clusters. So a `Stacking` inside it runs on a universe that it does not know in advance. A [`NonFiniteAllocationOptimisationResult`](@ref) holds weights over the universe of its own solve, so it cannot give the returns of a synthetic asset on a new universe. A `Stacking` that runs alone accepts a result in `opti`.

[`assert_special_nco_requirements`](@ref) calls this method on `opti`, and on each entry and the `default` of a schedule in `opti`.

# Arguments

  - `opti`: Inner optimisers of the `Stacking`.

# Validation

  - No element of `opti` is a [`NonFiniteAllocationOptimisationResult`](@ref). The method throws an `ArgumentError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`Stacking`](@ref)
  - [`assert_special_nco_requirements`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function assert_special_nco_requirements_stacking_opti(opti::AbstractVector)::Nothing
    @argcheck(!any(x -> isa(x, NonFiniteAllocationOptimisationResult), opti),
              ArgumentError("opti cannot contain NonFiniteAllocationOptimisationResult elements"))
    return nothing
end
function assert_special_nco_requirements(opt::Stacking)::Nothing
    opti = opt.opti
    if isa(opti, TimeDependent)
        for v in time_dependent_entries(opti)
            assert_special_nco_requirements_stacking_opti(v)
        end
    else
        assert_special_nco_requirements_stacking_opti(opti)
    end
    return nothing
end
function assert_external_optimiser(opt::Stacking)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    assert_estimated_prior(opt.pe, "opt.pe")
    assert_external_optimiser(opt.opto)
    if !isnothing(opt.cv)
        assert_external_optimiser(opt.opti)
    end
    return nothing
end
function assert_internal_optimiser(opt::Stacking)::Nothing
    assert_external_optimiser(opt.opto)
    assert_internal_optimiser(opt.opti)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`Stacking`](@ref) needs the previous portfolio weights.

Returns `true` when `opt.fees`, `opt.opti`, `opt.opto` or `opt.fb` needs them, or when a [`TimeDependent`](@ref) schedule in a field of `opt` holds a value that needs them. A [`TurnoverRiskMeasure`](@ref) and a turnover fee both need them.

# Related

  - [`needs_previous_weights`](@ref)
  - [`Stacking`](@ref)
"""
function needs_previous_weights(opt::Stacking)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.opti) ||
            needs_previous_weights(opt.opto) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`Stacking`](@ref) holds a [`TimeDependent`](@ref) schedule that a fold loop must resolve.

Returns `true` when [`time_dependent_fields`](@ref) reports a field of `opt`, or when `opt.opti`, `opt.opto` or `opt.fb` holds a schedule at any depth.

# Related

  - [`is_time_dependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`Stacking`](@ref)
"""
function is_time_dependent(opt::Stacking)
    return (!isempty(time_dependent_fields(opt)) ||
            is_time_dependent(opt.opti) ||
            is_time_dependent(opt.opto) ||
            is_time_dependent(opt.fb))
end
function time_dependent_field_defaults(::Stacking)::NamedTuple
    return stacking_td_defaults()
end
function inner_fold_fields(::Stacking)::Tuple
    return (:opti,)
end
function assert_time_dependent_fold_count(opt::Stacking, n::Integer,
                                          all_binds::Bool = true)::Nothing
    assert_time_dependent_fields_fold_count(opt, n, all_binds)
    if isa(opt.opti, AbstractVector)
        assert_time_dependent_fold_count(opt.opti, n, false)
    end
    assert_time_dependent_fold_count(opt.opto, n, all_binds)
    assert_time_dependent_fold_count(opt.fb, n, all_binds)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the [`TimeDependent`](@ref) schedules of a [`Stacking`](@ref) for the fold that `ctx` describes.

# Algorithm

 1. When [`is_time_dependent`](@ref) is `false` for `opt`, return `opt` unchanged.
 2. Resolve the schedules in the fields of `opt` with [`update_time_dependent_fields`](@ref), under `all_binds`.
 3. Resolve `opti` with `all_binds = false`, so that an element schedule that binds `:nearest` stays for the inner cross-validation of `opt`.
 4. Resolve `opto` and `fb` under `all_binds`.
 5. Rebuild `opt` with the resolved `opti`, `opto` and `fb`.

# Arguments

  - `opt`: The stacking optimiser.
  - `ctx`: The [`TimeDependentContext`](@ref) of the fold.
  - `all_binds`: `true` resolves every schedule. `false` resolves the schedules that bind `:outermost` alone.

# Returns

  - `Stacking`: The optimiser with the schedules of the fold resolved.

# Related

  - [`is_time_dependent`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
"""
function update_time_dependent_estimator(opt::Stacking, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    if !is_time_dependent(opt)
        return opt
    end
    opt = update_time_dependent_fields(opt, ctx, all_binds)
    return rebuild_estimator(opt,
                             (;
                              opti = update_time_dependent_estimator(opt.opti, ctx, false),
                              opto = update_time_dependent_estimator(opt.opto, ctx,
                                                                     all_binds),
                              fb = update_time_dependent_estimator(opt.fb, ctx, all_binds)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace each [`TimeDependent`](@ref) field of a [`Stacking`](@ref) with its static default.

The method does not recurse into the optimisers that `opt` holds. The inner cross-validation of `opt` resolves the element schedules of `opti` for each fold, and each full-sample inner solve resets its own schedules in its own [`_optimise`](@ref). An element schedule of `opti` stays in place, because `opti` is then a vector and not a schedule. A field schedule of `opti` binds `:outermost`, and the method replaces it with its `default`.

# Arguments

  - `opt`: The stacking optimiser.

# Returns

  - `Stacking`: The optimiser with no schedule in its own fields.

# Related

  - [`stacking_td_defaults`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`update_time_dependent_estimator`](@ref)
"""
function reset_time_dependent_estimator(opt::Stacking)
    return reset_time_dependent_fields(opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the [`Stacking`](@ref) `st` sliced to the asset indices `i`.

When `st.pe` holds a prior result, the view reads the `X` of that prior in place of the `X` it receives. The struct's `## View parameters` states what each field does.

# Related

  - [`Stacking`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(st::Stacking, i, X::MatNum, args...)::Stacking
    X = isa(st.pe, AbstractPriorResult) ? st.pe.X : X
    pe = port_opt_view(st.pe, i)
    wb = port_opt_view(st.wb, i)
    fees = port_opt_view(st.fees, i, X)
    sets = port_opt_view(st.sets, i)
    opti = port_opt_view(st.opti, i, X)
    opto = port_opt_view(st.opto, i, X)
    return Stacking(; pe = pe, wb = wb, fees = fees, sets = sets, scale = st.scale,
                    opti = opti, opto = opto, cv = st.cv, wf = st.wf, ex = st.ex,
                    fb = view_child(st.fb, i, X), brt = st.brt, strict = st.strict,
                    pcol = st.pcol, cache = port_opt_view(st.cache, i))
end
function non_investable_universe(st::Stacking, ni::VecStr)::Stacking
    return rebuild_estimator(st, (; sets = non_investable_sets(st.sets, ni)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the stacking optimisation.

[`optimise`](@ref) calls this method. The struct [`Stacking`](@ref) states the mathematics.

# Algorithm

 1. Resolve every [`TimeDependent`](@ref) field of `st` to its static default, with [`reset_time_dependent_estimator`](@ref).
 2. Pick the returns `rd` that `st.brt` selects, with [`returns_result_picker`](@ref).
 3. Fit the prior `pr` with `st.pe`, and take the element type `Tf` of the weights, fees and bounds from `pr.X` with [`float_if_integer`](@ref).
 4. Find the Investable Mask `imsk` of `pr`. Resolve the fees `fees` on the full universe, and place them on the assets of `imsk` with [`investable_fees_view`](@ref).
 5. Remove the two liquidation charges, `lq` and `flq`, from `fees`, giving `cfees`, the fees that the returns of the synthetic assets pay.
 6. Reduce `pr`, `st` and `rd` to the assets of `imsk` with [`investable_reduction`](@ref), giving `rdr`. `X` is the returns matrix of the reduced `pr`.
 7. Solve each inner optimiser of `st.opti` on `rdr` under `st.ex`, giving the inner results `resi`. Write the weights of result ``k`` to column ``k`` of `wi`.
 8. Build the returns of the synthetic assets, `rdo`, with [`predict_outer_returns`](@ref) under `st.cv`.
 9. Solve `st.opto` on `rdo`, giving the outer result `reso`.
10. Resolve the weight bounds `wb` from `st.wb` and `st.sets`.
11. Apply the Combination Weight `st.scale` to `reso.w` with [`combination_weights`](@ref).
12. Combine the coefficients with `wi`, and finalise the weights under `wb` and `st.wf` with [`outer_optimisation_finaliser`](@ref), giving `retcode` and `w`.
13. Return a [`StackingResult`](@ref). Its keyword constructor expands `w` onto the full asset universe through `imsk`.

# Arguments

  - `st`: The stacking optimiser.
  - $(arg_dict[:rd])
  - `branchorder`, `str_names`, `save`, `kwargs`: Passed to the solve of each inner optimiser and to the solve of `st.opto`.

# Validation

  - No inner optimiser returns an efficient frontier. The method throws an `ArgumentError` otherwise.

# Returns

  - `StackingResult`: The stacked portfolio.

# Related

  - [`Stacking`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(st::Stacking, rd::ReturnsResult; branchorder::Symbol = :optimal,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    st = reset_time_dependent_estimator(st)
    rd = returns_result_picker(rd, st.brt)
    pr = prior(st.pe, rd)
    # A weight and a fee are fractions, so integer returns take a floating point type.
    Tf = float_if_integer(eltype(pr.X))
    # Resolve the fee on the caller's universe before `investable_reduction` narrows `sets`.
    # A name stated over that universe must not be refused because the data delisted the
    # asset. A liquidation charge keyed by name cannot resolve at all once its `w` sits on
    # the complement while `sets` sits on the mask. `investable_fees_view` then places the
    # resolved fee on the axes the mask leaves.
    imsk = investable_mask(pr)
    fees = investable_fees_view(fees_constraints(st.fees, st.sets; datatype = Tf,
                                                 strict = st.strict), imsk, pr.X)
    # A forced exit is charged once, against the full-universe weight vector the fit
    # rebuilds, so only the result charges it. No sub-problem below holds that vector —
    # the exiting asset is in no cluster, its column being `NaN` — so none prices an exit.
    cfees = strip_liquidation_charges(fees, nothing)
    # The prior fits on the coverage universe and returns a result on the full asset
    # universe, where an asset it could not estimate carries `NaN`. Reduce once, here,
    # before the candidate solves: every candidate then sees the investable universe alone,
    # and each composes its own mask inside its own solve. `StackingResult` expands the
    # combined weights back.
    # The reduced `rd` takes a name of its own: a variable that is reassigned and then
    # captured by the fold's closure is boxed, which FLoops reports as a correctness and
    # performance problem on every call.
    _, pr, st, rdr = investable_reduction(imsk, pr, st, rd)
    X = pr.X
    opti = st.opti
    Ni = length(opti)
    wi = zeros(Tf, size(X, 2), Ni)
    resi = Vector{NonFiniteAllocationOptimisationResult}(undef, Ni)
    FLoops.@floop st.ex for (i, opt) in pairs(opti)
        res = optimise(opt, rdr; branchorder = branchorder, str_names = str_names,
                       save = save, kwargs...)
        #! Support efficient frontier?
        @argcheck(!isa(res.retcode, AbstractVector),
                  ArgumentError("res.retcode cannot be an AbstractVector; efficient frontier results are not supported here"))
        wi[:, i] = res.w
        resi[i] = res
    end
    rdo = predict_outer_returns(st.cv, st, FullUniverse(), rdr, pr, cfees, wi, resi)
    reso = optimise(st.opto, rdo; branchorder = branchorder, str_names = str_names,
                    save = save, kwargs...)
    wb = weight_bounds_constraints(st.wb, st.sets; N = size(X, 2), strict = st.strict,
                                   datatype = Tf)
    retcode, w = outer_optimisation_finaliser(wb, st.wf, resi, reso.retcode,
                                              combination_weights(st.scale, reso.w), wi)
    return StackingResult(; pr = pr, wb = wb, fees = fees, resi = resi, reso = reso,
                          cv = st.cv, retcode = retcode, w = w, imsk = imsk, fb = nothing)
end
"""
    optimise(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                     <:Any, <:Any, Nothing
                 }, rd::ReturnsResult;
             branchorder::Symbol = :optimal, str_names::Bool = false,
             save::Bool = true, kwargs...) -> StackingResult

Run the stacking portfolio optimisation.

# Arguments

  - `st`: The stacking optimiser.
  - $(arg_dict[:rd])
  - `branchorder`: Passed to each inner optimiser and to the outer optimiser. The branch order of a clusterisation.
  - `str_names`: Passed to each inner optimiser and to the outer optimiser. When `true`, the optimisation uses string names for the assets.
  - `save`: Passed to each inner optimiser and to the outer optimiser. When `true`, a JuMP result keeps its model.
  - `kwargs`: Passed to each inner optimiser and to the outer optimiser.

# Validation

  - No field in the tree of `st` holds an [`Online`](@ref). The method throws an `ArgumentError` that names the field otherwise, through [`assert_batch_entry`](@ref). A plain `optimise` is a batch fit, and a wrapper resolves only at the warm-up of the online arm of the fold loop.
  - No inner optimiser returns an efficient frontier. The method throws an `ArgumentError` otherwise.

# Returns

  - `res::StackingResult`: The stacked portfolio. `retcode` is an [`OptimisationFailure`](@ref) when an inner solve, the outer solve or the weight finaliser failed. An outer efficient frontier gives one weight vector and one return code for each point.

# Related

  - [`Stacking`](@ref)
  - [`StackingResult`](@ref)
  - [`_optimise`](@ref)
  - [`combination_weights`](@ref)
"""
function optimise(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                               <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    assert_batch_entry(st, "`optimise`")
    return _optimise(st, rd; branchorder = branchorder, str_names = str_names, save = save,
                     kwargs...)
end

export StackingResult, Stacking
