"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that average one optimiser over random subsets of the assets.

A subtype draws random subsets of the assets, solves one base optimiser on each subset, and averages the subset weights into weights over all the assets. The average has less estimation error than one solve over all the assets [shen2017](@cite).

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`SubsetResampling`](@ref)
  - [`SubsetResamplingResult`](@ref)

# References

  - $(ref_dict[:shen2017])
"""
abstract type BaseSubsetResamplingOptimisationEstimator <:
              NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Result of a [`SubsetResampling`](@ref) optimisation.

`ress[m]` is the result of the subset whose asset indices are `idx[:, m]`. `w` is the average of the subset weights over all the assets, after the weight finaliser.

`idx`, `pr`, `wb` and `fees` are on the universe that the subsets come from. When `imsk` is not `nothing`, this universe is the investable universe. `w` is on the full asset universe.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SubsetResamplingResult(;
        pr::Option{<:AbstractPriorResult},
        wb::Option{<:WeightBounds},
        fees::Option{<:Fees},
        ress::AbstractVector{<:NonFiniteAllocationOptimisationResult},
        idx::MatNum,
        retcode::OptRetCode_VecOptRetCode,
        w::VecNum_VecVecNum,
        imsk::Option{<:BitVector} = nothing,
        fb::Option{<:OptE_Opt_FbChain}
    ) -> SubsetResamplingResult

Keywords correspond to the struct's fields. The keyword constructor expands `w` onto the full asset universe with [`expand_investable_weights`](@ref), and [`_optimise`](@ref) calls it. The positional constructor does not expand `w`. [`set_retcode`](@ref) and [`factory`](@ref) call it, because their `w` is already on the full asset universe.

# Related

  - [`SubsetResampling`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`subset_resampling_finaliser`](@ref)
  - [`expand_investable_weights`](@ref)

# References

  - $(ref_dict[:shen2017])
"""
@concrete struct SubsetResamplingResult <: NonJuMPOptimisationResult
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
    Optimisation result of each asset subset, one entry for each subset, in the column order of `idx`.
    """
    ress
    """
    Asset indices of the subsets, one column for each subset and one row for each asset of a subset. When `imsk` is not `nothing`, the indices are into the investable universe.
    """
    idx
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    Averaged and finalised portfolio weights over the full asset universe, or one vector for each point of an efficient frontier.
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
    function SubsetResamplingResult(pr::Option{<:AbstractPriorResult},
                                    wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                                    ress::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                                    idx::MatNum, retcode::OptRetCode_VecOptRetCode,
                                    w::VecNum_VecVecNum, imsk::Option{<:BitVector},
                                    fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(pr), typeof(wb), typeof(fees), typeof(ress), typeof(idx),
                   typeof(retcode), typeof(w), typeof(imsk), typeof(fb)}(pr, wb, fees, ress,
                                                                         idx, retcode, w,
                                                                         imsk, fb)
    end
end
function SubsetResamplingResult(; pr::Option{<:AbstractPriorResult},
                                wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                                ress::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                                idx::MatNum, retcode::OptRetCode_VecOptRetCode,
                                w::VecNum_VecVecNum, imsk::Option{<:BitVector} = nothing,
                                fb::Option{<:OptE_Opt_FbChain})::SubsetResamplingResult
    return SubsetResamplingResult(pr, wb, fees, ress, idx, retcode,
                                  expand_investable_weights(imsk, w), imsk, fb)
end
# The subset-resampling family carries the mask on the result itself, so the fold reads it
# directly.
function result_investable_mask(res::SubsetResamplingResult)
    return res.imsk
end
"""
    set_retcode(res::SubsetResamplingResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`SubsetResamplingResult`](@ref) with a different return code.

On an efficient frontier the result has one return code for each point, so a failure code on one entry marks that point alone. The method copies every other field unchanged.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one return code for each point of an efficient frontier.

# Returns

  - [`SubsetResamplingResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`SubsetResamplingResult`](@ref)
"""
function set_retcode(res::SubsetResamplingResult, retcode::OptRetCode_VecOptRetCode)
    return SubsetResamplingResult(res.pr, res.wb, res.fees, res.ress, res.idx, retcode,
                                  res.w, res.imsk, res.fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a [`SubsetResamplingResult`](@ref) with the fallback chain `fb`.

[`optimise`](@ref) calls this method when a `SubsetResampling` fallback gives the answer. The method copies every other field unchanged, and it does not expand `sr.w` again.

# Related

  - [`SubsetResamplingResult`](@ref)
  - [`FbChain`](@ref)
"""
function factory(sr::SubsetResamplingResult, fb::Option{<:OptE_Opt_FbChain})
    # The positional constructor, because `sr.w` is already on the full asset universe: the
    # keyword one expands, and a second pass over an expanded vector is a length error.
    return SubsetResamplingResult(sr.pr, sr.wb, sr.fees, sr.ress, sr.idx, sr.retcode, sr.w,
                                  sr.imsk, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Averages the weights of one base optimiser over random subsets of the assets.

The estimator draws `n_subsets` distinct subsets of `subset_size` assets, solves `opt` on each subset, and averages the subset weights. The average then goes through the weight finaliser `wf` under the bounds `wb`. [shen2017](@cite) states the method. When the number of possible subsets is larger than `max_comb`, the draw is approximate, and two subsets can be equal.

# Mathematical definition

Draw ``M`` distinct subsets ``S_1, \\ldots, S_M`` of ``k`` assets each from the ``N`` assets. Solve the base optimiser on each subset, and average the subset weights over all the assets:

```math
\\begin{align}
k &= \\begin{cases}
       \\mathrm{subset\\_size} & \\text{if } \\mathrm{subset\\_size} \\in \\mathbb{Z}\\,, \\\\
       \\max\\!\\left(\\mathrm{round}(s N),\\, 1\\right) & \\text{if } s = \\mathrm{subset\\_size} \\in (0, 1)\\,, \\\\
       f(\\mathcal{P}) & \\text{if } f = \\mathrm{subset\\_size} \\text{ is callable}\\,,
     \\end{cases} \\\\
\\boldsymbol{w}^* &= \\frac{1}{M} \\sum_{m=1}^{M} \\boldsymbol{e}_{S_m}(\\boldsymbol{w}_{S_m})\\,, \\\\
\\left[\\boldsymbol{e}_{S}(\\boldsymbol{v})\\right]_i &= \\begin{cases}
       v_j & \\text{if } i \\text{ is the } j\\text{-th asset of } S\\,, \\\\
       0 & \\text{if } i \\notin S\\,.
     \\end{cases}
\\end{align}
```

Where:

  - ``\\boldsymbol{w}^*``: Averaged portfolio weights, before the weight finaliser.
  - ``M``: Number of subsets, `n_subsets`. A callable `n_subsets` gives its value on ``\\mathcal{P}``.
  - ``k``: Number of assets in each subset.
  - ``s``: Fractional `subset_size`.
  - ``f``: Callable `subset_size`.
  - ``\\mathcal{P}``: Prior result of `pe` on the investable universe.
  - ``S_m``: The ``m``-th subset, with its assets in ascending order.
  - ``\\boldsymbol{w}_{S_m}``: Weights that `opt` gives the assets of ``S_m``.
  - ``\\boldsymbol{e}_{S}(\\cdot)``: Map that puts the weights of a subset ``S`` at the assets of ``S``, and zero at every other asset.
  - ``\\boldsymbol{v}``: Weight vector over the assets of a subset ``S``, with entries ``v_j``.
  - ``i``: Index of an asset of the ``N`` assets.
  - $(math_dict[:N])

The subsets are distinct, so ``M \\leq \\binom{N}{k}``. When each ``\\boldsymbol{w}_{S_m}`` sums to one, ``\\boldsymbol{w}^*`` sums to one. An asset that is in no subset has zero weight.

``\\mathrm{round}`` gives the nearest integer, and the even integer at a tie. So ``s = 0.8`` on ``N = 12`` assets gives ``k = 10``, and ``s = 0.5`` on ``N = 5`` assets gives ``k = 2``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SubsetResampling(;
        pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
        wb::TD_Option{<:WbE_Wb} = nothing,
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        opt::OptE_TD,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        subset_size::TD{<:SubsetSizeE} = 0.8,
        n_subsets::TD{<:NumberSubsetsE} = 2,
        max_comb::Integer = 1_000_000_000,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        fb::TDO_Option{<:OptE_Opt} = nothing,
        brt::Bool = false,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> SubsetResampling

Keywords correspond to the struct's fields.

## Time-dependent fields

`pe`, `wb`, `fees`, `sets`, `opt`, `wf`, `subset_size`, `n_subsets` and `fb` can hold a [`TimeDependent`](@ref) schedule with one entry for each fold. The other fields cannot hold a schedule.

The loop inside `SubsetResampling` is over asset subsets, not over time folds. No inner fold loop exists for a `bind = :nearest` schedule in `opt` or `fb`, so the constructor refuses one. A fold loop that reaches the estimator resolves each schedule. A solve with no fold loop resets each schedule to its own `default`. A schedule with no `default` resets `pe`, `wf`, `subset_size` and `n_subsets` to their keyword defaults, and `wb`, `fees`, `sets` and `fb` to `nothing`. A schedule in `opt` with no `default` throws a [`TimeDependentDefaultError`](@ref).

## Validation

  - `opt` passes [`assert_internal_optimiser`](@ref). A schedule applies the check to each entry and to its `default`.
  - If `wb` is a [`WeightBoundsEstimator`](@ref), `sets` is not `nothing`. The constructor throws an [`IsNothingError`](@ref) otherwise.
  - If `fees` is a [`FeesEstimator`](@ref), `sets` is not `nothing`. The constructor throws an [`IsNothingError`](@ref) otherwise.
  - If `subset_size` is an `Integer`: `subset_size >= 1`.
  - If `subset_size` is any other real number: `0 < subset_size < 1`.
  - If `n_subsets` is an `Integer`: `n_subsets >= 2`.
  - `max_comb > 0` and finite.
  - A schedule in `opt` or `fb` has `bind !== :nearest`. Each entry of a schedule passes the check of its field.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `opt`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

`SubsetResampling` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `pe` is a prior result, the method uses `pe.X` in place of `X`.
  - `pe`, `wb`, `sets` and `cache` recurse through [`port_opt_view`](@ref) with the index alone.
  - `fees` and `opt` recurse with the index and `X`.
  - `fb` recurses through [`view_child`](@ref) with the index and `X`.

# Related

  - [`optimise`](@ref)
  - [`SubsetResamplingResult`](@ref)
  - [`BaseSubsetResamplingOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`port_opt_view`](@ref)
  - [`subset_resampling_finaliser`](@ref)

# References

  - $(ref_dict[:shen2017])
"""
@propagatable @concrete struct SubsetResampling <: BaseSubsetResamplingOptimisationEstimator
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
    Base portfolio optimiser applied to each asset subset.
    """
    @fprop opt
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:subset_size])
    """
    subset_size
    """
    $(field_dict[:n_subsets])
    """
    n_subsets
    """
    $(field_dict[:max_comb])
    """
    max_comb
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
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
    $(field_dict[:cache_opt])
    """
    @fprop cache
    function SubsetResampling(pe::Onl{<:TD{<:PrE_Pr}}, wb::TD_Option{<:WbE_Wb},
                              fees::TD_Option{<:FeesE_Fees},
                              sets::TD_Option{<:UniverseSets}, opt::OptE_TD,
                              wf::TD{<:WeightFinaliser}, ex::FLoops.Transducers.Executor,
                              subset_size::TD{<:SubsetSizeE},
                              n_subsets::TD{<:NumberSubsetsE}, max_comb::Integer,
                              rng::Random.AbstractRNG, seed::Option{<:Integer},
                              fb::TDO_Option{<:OptE_Opt}, brt::Bool, strict::Bool,
                              cache::Option{<:ReturnsBufferState})
        assert_no_nearest_bind_optimiser_schedule(opt, :opt, :SubsetResampling)
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :SubsetResampling)
        assert_internal_optimiser(opt)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if isa(subset_size, Integer)
            assert_nonempty_nonneg_finite_val(subset_size - 1, "subset_size - 1")
        elseif isa(subset_size, Real)
            assert_unit_interval(subset_size, :subset_size)
        end
        if isa(n_subsets, Integer)
            assert_nonempty_nonneg_finite_val(n_subsets - 2, "n_subsets - 2")
        end
        assert_nonempty_gt0_finite_val(max_comb, :max_comb)
        assert_time_dependent_substitution(SubsetResampling,
                                           (; pe, wb, fees, sets, opt, wf, ex, subset_size,
                                            n_subsets, max_comb, rng, seed, fb, brt,
                                            strict), subset_resampling_td_defaults())
        return new{typeof(pe), typeof(wb), typeof(fees), typeof(sets), typeof(opt),
                   typeof(wf), typeof(ex), typeof(subset_size), typeof(n_subsets),
                   typeof(max_comb), typeof(rng), typeof(seed), typeof(fb), typeof(brt),
                   typeof(strict), typeof(cache)}(pe, wb, fees, sets, opt, wf, ex,
                                                  subset_size, n_subsets, max_comb, rng,
                                                  seed, fb, brt, strict, cache)
    end
end
function SubsetResampling(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
                          wb::TD_Option{<:WbE_Wb} = nothing,
                          fees::TD_Option{<:FeesE_Fees} = nothing,
                          sets::TD_Option{<:UniverseSets} = nothing, opt::OptE_TD,
                          wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                          ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                          subset_size::TD{<:SubsetSizeE} = 0.8,
                          n_subsets::TD{<:NumberSubsetsE} = 2,
                          max_comb::Integer = 1_000_000_000,
                          rng::Random.AbstractRNG = Random.default_rng(),
                          seed::Option{<:Integer} = nothing,
                          fb::TDO_Option{<:OptE_Opt} = nothing, brt::Bool = false,
                          strict::Bool = false,
                          cache::Option{<:ReturnsBufferState} = nothing)::SubsetResampling
    return SubsetResampling(pe, wb, fees, sets, opt, wf, ex, subset_size, n_subsets,
                            max_comb, rng, seed, fb, brt, strict, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`SubsetResampling`](@ref) fields that can hold a [`TimeDependent`](@ref).

The constructor passes the defaults to [`assert_time_dependent_substitution`](@ref), and [`time_dependent_field_defaults`](@ref) returns them. `opt` is a required keyword and has no static default, so its entry is [`NoDefault`](@ref). A schedule in `opt` must have its own `default` to run outside a fold loop. `pe`, `wf`, `subset_size` and `n_subsets` take their keyword defaults. The tuple omits `wb`, `fees`, `sets` and `fb`, because their static default is `nothing`.

# Related

  - [`SubsetResampling`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function subset_resampling_td_defaults()::NamedTuple
    return (; pe = EmpiricalPrior(), opt = NoDefault(), wf = IterativeWeightFinaliser(),
            subset_size = 0.8, n_subsets = 2)
end
function time_dependent_field_defaults(::SubsetResampling)::NamedTuple
    return subset_resampling_td_defaults()
end
function assert_external_optimiser(opt::SubsetResampling)::Nothing
    assert_estimated_prior(opt.pe, "opt.pe")
    return assert_external_optimiser(opt.opt)
end
function assert_internal_optimiser(opt::SubsetResampling)::Nothing
    return assert_internal_optimiser(opt.opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when a field of `opt` needs the previous portfolio weights.

The method reads each field that holds a [`TimeDependent`](@ref) schedule, then `fees`, `opt` and `fb`.

# Related

  - [`needs_previous_weights`](@ref)
  - [`SubsetResampling`](@ref)
"""
function needs_previous_weights(opt::SubsetResampling)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `opt` holds a [`TimeDependent`](@ref) schedule.

The method reads the fields of `opt` itself, then asks the same question of `opt.opt` and `opt.fb`.

# Related

  - [`is_time_dependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
"""
function is_time_dependent(opt::SubsetResampling)
    return (!isempty(time_dependent_fields(opt)) ||
            is_time_dependent(opt.opt) ||
            is_time_dependent(opt.fb))
end
function assert_time_dependent_fold_count(opt::SubsetResampling, n::Integer,
                                          all_binds::Bool = true)::Nothing
    assert_time_dependent_fields_fold_count(opt, n, all_binds)
    assert_time_dependent_fold_count(opt.opt, n, all_binds)
    assert_time_dependent_fold_count(opt.fb, n, all_binds)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every [`TimeDependent`](@ref) schedule of `opt` for the fold that `ctx` describes.

# Algorithm

 1. Return `opt` unchanged when [`is_time_dependent`](@ref) is `false` for it.
 2. Resolve the fields of `opt` itself with [`update_time_dependent_fields`](@ref), giving a new `opt`.
 3. Resolve `opt.opt` and `opt.fb` with this function, and rebuild `opt` with them through [`rebuild_estimator`](@ref).

# Related

  - [`is_time_dependent`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
  - [`TimeDependentContext`](@ref)
"""
function update_time_dependent_estimator(opt::SubsetResampling, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    if !is_time_dependent(opt)
        return opt
    end
    opt = update_time_dependent_fields(opt, ctx, all_binds)
    return rebuild_estimator(opt,
                             (;
                              opt = update_time_dependent_estimator(opt.opt, ctx,
                                                                    all_binds),
                              fb = update_time_dependent_estimator(opt.fb, ctx, all_binds)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reset each [`TimeDependent`](@ref) schedule in the fields of `opt` itself to its value outside a fold loop.

The method does not reset the schedules inside `opt.opt` or `opt.fb`. Each subset solve calls [`optimise`](@ref), which resets the schedules of the optimiser that it solves, and [`optimise`](@ref) resets a fallback before it solves it.

# Related

  - [`update_time_dependent_estimator`](@ref)
  - [`reset_time_dependent_fields`](@ref)
"""
function reset_time_dependent_estimator(opt::SubsetResampling)
    return reset_time_dependent_fields(opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of a [`SubsetResampling`](@ref) on the assets `i`.

`X` is the returns matrix that the views of `fees`, `opt` and `fb` read. When `sr.pe` is a prior result, the method uses `sr.pe.X` in place of `X`. The `## View parameters` subsection of [`SubsetResampling`](@ref) lists what the method does to each field.

# Related

  - [`port_opt_view`](@ref)
  - [`view_child`](@ref)
"""
function port_opt_view(sr::SubsetResampling, i, X::MatNum, args...)::SubsetResampling
    X = isa(sr.pe, AbstractPriorResult) ? sr.pe.X : X
    pe = port_opt_view(sr.pe, i)
    wb = port_opt_view(sr.wb, i)
    fees = port_opt_view(sr.fees, i, X)
    sets = port_opt_view(sr.sets, i)
    opt = port_opt_view(sr.opt, i, X)
    return SubsetResampling(; pe = pe, wb = wb, fees = fees, sets = sets, opt = opt,
                            wf = sr.wf, ex = sr.ex, subset_size = sr.subset_size,
                            n_subsets = sr.n_subsets, max_comb = sr.max_comb, rng = sr.rng,
                            seed = sr.seed, fb = view_child(sr.fb, i, X), brt = sr.brt,
                            strict = sr.strict, cache = port_opt_view(sr.cache, i))
end
function non_investable_universe(sr::SubsetResampling, ni::VecStr)::SubsetResampling
    return rebuild_estimator(sr, (; sets = non_investable_sets(sr.sets, ni)))
end
"""
    subset_resampling_retcode(ress::VecOpt, retcode::OptimisationReturnCode)

Combine the return codes of the subset optimisations with the return code of the weight finaliser.

# Algorithm

 1. Read the return code of each subset result, giving `resi_retcodes`.
 2. Return `retcode` unchanged when no entry of `resi_retcodes` and not `retcode` is an [`OptimisationFailure`](@ref).
 3. Otherwise build `msg`, with the line `opti failed.` when a subset failed, and then the line `weight bounds finalisation failed.` when the finaliser failed.
 4. Return an [`OptimisationFailure`](@ref) whose `res` is `(; msg, opti = resi_retcodes, wb = retcode)`. Each return code keeps its own diagnostics.

# Arguments

  - `ress`: Results of the subset optimisations.
  - `retcode`: Return code of the weight finaliser.

# Returns

  - `retcode::OptimisationReturnCode`: `retcode`, or the failure of step 4.

# Related

  - [`subset_resampling_finaliser`](@ref)
  - [`SubsetResampling`](@ref)
"""
function subset_resampling_retcode(ress::VecOpt, retcode::OptimisationReturnCode)
    resi_retcodes = getproperty.(ress, :retcode)
    resi_flag = any(x -> isa(x, OptimisationFailure), resi_retcodes)
    wb_flag = isa(retcode, OptimisationFailure)
    return if resi_flag || wb_flag
        msg = ""
        if resi_flag
            msg *= "opti failed.\n"
        end
        if wb_flag
            msg *= "weight bounds finalisation failed.\n"
        end
        OptimisationFailure(; res = (; msg = msg, opti = resi_retcodes, wb = retcode))
    else
        retcode
    end
end
"""
    subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                wb::WeightBounds, wf::WeightFinaliser,
                                ress::VecOpt, w::VecNum_VecVecNum)

Average the subset weights over all the assets, and finalise the average under the weight bounds.

The method computes ``\\boldsymbol{w}^*`` of the `# Mathematical definition` of [`SubsetResampling`](@ref).

# Algorithm

 1. Allocate `w`, a vector of ``N`` zeros with the element type of the weights of the first subset.
 2. For each subset `i`, add `ress[i].w` into `w` at the asset indices `asset_idx[:, i]`.
 3. Divide `w` by `n_subsets`.
 4. Finalise `w` with `wf` under `wb` through [`finalise_weight_bounds`](@ref), giving `retcode` and the finalised `w`.
 5. Combine `retcode` with the return codes of `ress` through [`subset_resampling_retcode`](@ref).

On an efficient frontier, the method runs the steps once for each point.

# Arguments

  - `N::Integer`: Number of assets in the universe that the subsets come from.
  - `n_subsets::Integer`: Number of asset subsets.
  - `asset_idx::MatNum`: Asset indices, one column for each subset.
  - `wb::WeightBounds`: Weight bounds of the finaliser.
  - `wf::WeightFinaliser`: Weight finaliser that repairs a bounds violation.
  - `ress::VecOpt`: Subset optimisation results, in the column order of `asset_idx`.
  - `w::VecNum_VecVecNum`: Weights of the first subset. The method reads only the type of `w`. A vector of vectors selects the efficient-frontier method.

# Returns

  - `retcode`: The return code from [`subset_resampling_retcode`](@ref), or one for each point of an efficient frontier.
  - `w`: The averaged and finalised weights, or one vector for each point of an efficient frontier.

# Related

  - [`SubsetResampling`](@ref)
  - [`subset_resampling_retcode`](@ref)
  - [`finalise_weight_bounds`](@ref)
"""
function subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                     wb::WeightBounds, wf::WeightFinaliser, ress::VecOpt,
                                     ::VecNum)
    w = zeros(eltype(ress[1].w), N)
    for i in 1:n_subsets
        idx = view(asset_idx, :, i)
        w[idx] .+= ress[i].w
    end
    w /= n_subsets
    retcode, w = finalise_weight_bounds(wf, wb, w)
    return subset_resampling_retcode(ress, retcode), w
end
function subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                     wb::WeightBounds, wf::WeightFinaliser, ress::VecOpt,
                                     ws::VecVecNum)
    M = length(ws)
    w = [zeros(eltype(ress[1].w[i]), N) for i in 1:M]
    for i in 1:n_subsets
        idx = view(asset_idx, :, i)
        for j in 1:M
            w[j][idx] .+= ress[i].w[j]
        end
    end
    for j in 1:M
        w[j] /= n_subsets
    end
    retcode_w = [finalise_weight_bounds(wf, wb, wi) for wi in w]
    return map(x -> subset_resampling_retcode(ress, x[1]), retcode_w),
           map(x -> x[2], retcode_w)
end
function _optimise(sr::SubsetResampling, rd::ReturnsResult; branchorder::Symbol = :optimal,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    sr = reset_time_dependent_estimator(sr)
    rd = returns_result_picker(rd, sr.brt)
    pr = prior(sr.pe, rd)
    # A weight and a fee are fractions, so integer returns take a floating point type.
    Tf = float_if_integer(eltype(pr.X))
    # Resolve the fee on the caller's universe before `investable_reduction` narrows `sets`.
    # A name stated over that universe must not be refused because the data delisted the
    # asset. A liquidation charge keyed by name cannot resolve at all once its `w` sits on
    # the complement while `sets` sits on the mask. `investable_fees_view` then places the
    # resolved fee on the axes the mask leaves.
    imsk = investable_mask(pr)
    fees = investable_fees_view(fees_constraints(sr.fees, sr.sets; datatype = Tf,
                                                 strict = sr.strict), imsk, pr.X)
    # The prior fits on the coverage universe and returns a result on the full asset
    # universe, where an asset it could not estimate carries `NaN`. Reduce once, here,
    # before the sample: `N` is then the count of investable assets, so every subset is
    # drawn from the investable universe alone and no subset can hold a dead asset.
    # `SubsetResamplingResult` expands the averaged weights back.
    # The reduced `rd` takes a name of its own: a variable that is reassigned and then
    # captured by the fold's closure is boxed, which FLoops reports as a correctness and
    # performance problem on every call.
    _, pr, sr, rdr = investable_reduction(imsk, pr, sr, rd)
    X = pr.X
    N = size(X, 2)
    (; subset_size, n_subsets, max_comb, rng, seed) = sr
    subset_size = get_subset_size(subset_size, pr)
    n_subsets = get_n_subsets(n_subsets, pr)
    asset_idx = sample_unique_assets(N, subset_size, n_subsets; max_comb = max_comb,
                                     rng = rng, seed = seed)
    opt = sr.opt
    ress = Vector{NonFiniteAllocationOptimisationResult}(undef, n_subsets)
    FLoops.@floop sr.ex for i in 1:n_subsets
        idx = view(asset_idx, :, i)
        opti = port_opt_view(opt, idx, X)
        rdi = port_opt_view(rdr, idx)
        ress[i] = optimise(opti, rdi; branchorder = branchorder, str_names = str_names,
                           save = save, kwargs...)
    end
    wb = weight_bounds_constraints(sr.wb, sr.sets; N = N, strict = sr.strict, datatype = Tf)
    retcode, w = subset_resampling_finaliser(N, n_subsets, asset_idx, wb, sr.wf, ress,
                                             ress[1].w)
    return SubsetResamplingResult(; pr = pr, wb = wb, fees = fees, ress = ress,
                                  idx = asset_idx, retcode = retcode, w = w, imsk = imsk,
                                  fb = nothing)
end
"""
    optimise(sr::SubsetResampling{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                     <:Any, <:Any, <:Any, <:Any, <:Any, Nothing
                 }, rd::ReturnsResult;
             branchorder::Symbol = :optimal, str_names::Bool = false,
             save::Bool = true, kwargs...) -> SubsetResamplingResult

Solve a [`SubsetResampling`](@ref) that has no fallback, and return the averaged portfolio.

A `SubsetResampling` with a fallback goes through the generic [`optimise`](@ref), which walks the fallback chain. The two methods give the same result when `fb` is `nothing`.

# Algorithm

 1. Refuse an [`Online`](@ref) in the tree of `sr` with [`assert_batch_entry`](@ref).
 2. Reset the schedules of `sr` with [`reset_time_dependent_estimator`](@ref).
 3. Pick the returns data that `sr.brt` selects with [`returns_result_picker`](@ref), giving `rd`.
 4. Fit the prior `sr.pe` on `rd` with [`prior`](@ref), giving `pr`, and read its investable mask with [`investable_mask`](@ref), giving `imsk`.
 5. Resolve the fees on the full asset universe with [`fees_constraints`](@ref), and put them on the investable universe with [`investable_fees_view`](@ref), giving `fees`.
 6. Reduce `pr`, `sr` and `rd` to the investable universe with [`investable_reduction`](@ref), giving `rdr`. `N` is the number of investable assets.
 7. Resolve `subset_size` and `n_subsets` on `pr` with [`get_subset_size`](@ref) and [`get_n_subsets`](@ref).
 8. Draw the subsets with [`sample_unique_assets`](@ref), giving `asset_idx`. When `binomial(N, subset_size)` is at most `max_comb`, the subsets are distinct. Otherwise the function draws each subset alone, and two subsets can be equal.
 9. For each column of `asset_idx`, view `sr.opt` and `rdr` on the subset with [`port_opt_view`](@ref) and optimise the view, giving `ress`. The loop runs on the executor `sr.ex`.
10. Build the weight bounds `wb` over the ``N`` assets with [`weight_bounds_constraints`](@ref).
11. Average and finalise the subset weights with [`subset_resampling_finaliser`](@ref), giving `retcode` and `w`.
12. Return a [`SubsetResamplingResult`](@ref). Its keyword constructor expands `w` onto the full asset universe.

# Arguments

  - `sr`: The subset resampling optimiser to use.
  - $(arg_dict[:rd])
  - `branchorder`: Branch order, passed to each subset optimisation.
  - `str_names`: Whether each subset optimisation uses string names for the assets.
  - `save`: Whether each subset optimisation saves its JuMP model in its result.
  - `kwargs`: Other keyword arguments, passed to each subset optimisation.

# Validation

  - No field in the tree of `sr` holds an [`Online`](@ref). [`assert_batch_entry`](@ref) throws an `ArgumentError` that names the field otherwise. A plain `optimise` is a batch fit, and the fold loop resolves the wrapper only at the warm-up of its online arm.
  - `n_subsets <= binomial(N, subset_size)`. [`sample_unique_assets`](@ref) throws an `ArgumentError` otherwise.

# Returns

  - `res::SubsetResamplingResult`: The averaged portfolio. `retcode` is an [`OptimisationFailure`](@ref) when a subset optimisation fails, or when the weight finaliser fails.

# Related

  - [`SubsetResampling`](@ref)
  - [`SubsetResamplingResult`](@ref)
  - [`subset_resampling_finaliser`](@ref)
"""
function optimise(sr::SubsetResampling{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; branchorder::Symbol = :optimal,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    assert_batch_entry(sr, "`optimise`")
    return _optimise(sr, rd; branchorder = branchorder, str_names = str_names, save = save,
                     kwargs...)
end

export SubsetResamplingResult, SubsetResampling
