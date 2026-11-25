"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that average one optimiser over random subsets of the assets.

A subtype draws random subsets of the assets, solves one base optimiser on each subset, and averages the subset weights into weights over all the assets. Each subset portfolio rests on parameters that are estimated more accurately than those of one solve over all the assets, and the average of the subset portfolios is diversified over all the assets, as in the abstract of [shen2017](@cite).

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`SubsetResampling`](@ref)
  - [`SubsetResamplingResult`](@ref)

# References

  - $(ref_dict[:shen2017]) Abstract.
"""
abstract type BaseSubsetResamplingOptimisationEstimator <:
              NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Result of a [`SubsetResampling`](@ref) optimisation.

`ress[m]` is the result of the subset whose asset indices are `idx[:, m]`. `w` is the average of the subset weights over all the assets, after the weight finaliser. The average is the last step of Algorithm 1 of [shen2017](@cite).

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

  - $(ref_dict[:shen2017]) Algorithm 1.
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

Sizes each asset subset as a power of the number of assets, the subset size of [shen2017](@cite).

An instance is the default `subset_size` of [`SubsetResampling`](@ref). A call of the instance on a prior result or a returns result gives the subset size.

# Mathematical definition

```math
\\begin{align}
k &= \\mathrm{round}\\left(N^{p}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:k_subset])
  - ``p``: Exponent of the number of assets, `p`.
  - $(math_dict[:N])

``\\mathrm{round}`` gives the nearest integer, and the even integer at a tie. When ``N \\geq 1`` and ``0 < p \\leq 1``, ``1 \\leq N^{p} \\leq N``, so ``1 \\leq k \\leq N``. So ``p = 0.8`` on ``N = 100`` assets gives ``k = 40``, and ``p = 0.7`` gives ``k = 25``.

[shen2017](@citet) state the subset size as ``b = n^{p}`` and give no rounding rule. The simulation study uses ``p = 0.8``, and the empirical study uses ``p = 0.7``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PowerSubsetSize(; p::Real = 0.8) -> PowerSubsetSize

Keywords correspond to the struct's fields. The default is the exponent of the simulation study of [shen2017](@cite).

## Validation

  - `0 < p <= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> PowerSubsetSize()
PowerSubsetSize
  p ┴ Float64: 0.8
```

# Related

  - [`SubsetResampling`](@ref)
  - [`SubsetSizeEstimator`](@ref)
  - [`get_subset_size`](@ref)

# References

  - $(ref_dict[:shen2017]) Page 1520, Figure 1 on page 1521, and the note of Table 2 on page 1522.
"""
@concrete struct PowerSubsetSize <: SubsetSizeEstimator
    """
    The exponent ``p`` of the number of assets.
    """
    p
    function PowerSubsetSize(p::Real)
        @argcheck(zero(p) < p <= one(p),
                  DomainError(p, "0 < p <= 1 must hold. Got\np => $p"))
        return new{typeof(p)}(p)
    end
end
function PowerSubsetSize(; p::Real = 0.8)::PowerSubsetSize
    return PowerSubsetSize(p)
end
function (s::PowerSubsetSize)(rd::Union{Pr_RR, AbstractPricesResult})::Integer
    return round(Int, size(rd.X, 2)^s.p)
end
"""
$(DocStringExtensions.TYPEDEF)

Averages the weights of one base optimiser over random subsets of the assets.

The estimator draws `n_subsets` distinct subsets of `subset_size` assets, solves `opt` on each subset, and averages the subset weights. The average then goes through the weight finaliser `wf` under the bounds `wb`. Algorithm 1 of [shen2017](@cite) states the method. It solves the mean-variance problem on each subset, and this type takes any base optimiser. It samples the assets of each subset without replacement, and it does not require the subsets to differ. This type draws distinct subsets. When the number of possible subsets is larger than `max_comb`, the draw is approximate, and two subsets can be equal.

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
  - $(math_dict[:k_subset])
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

A subset whose solve fails is dropped when enough of the others solve, see `min_solved`. The average is then over the ``M'`` subsets that solved, with ``1 / M'`` in place of ``1 / M``, so it keeps the budget. Each point of an efficient frontier drops its own subsets.

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
        min_solved::TD{<:Real_CountFraction} = FractionOf(1),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        subset_size::TD{<:SubsetSizeE} = PowerSubsetSize(),
        n_subsets::TD{<:NumberSubsetsE} = 2,
        max_comb::Integer = 1_000_000_000,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        fb::TDO_Option{<:OptE_Opt} = nothing,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> SubsetResampling

Keywords correspond to the struct's fields. The default `subset_size` is the subset size of the simulation study of [shen2017](@cite), ``k = \\mathrm{round}\\left(N^{0.8}\\right)``, see [`PowerSubsetSize`](@ref).

## Time-dependent fields

`pe`, `wb`, `fees`, `sets`, `opt`, `wf`, `min_solved`, `subset_size`, `n_subsets` and `fb` can hold a [`TimeDependent`](@ref) schedule with one entry for each fold. The other fields cannot hold a schedule.

The loop inside `SubsetResampling` is over asset subsets, not over time folds. No inner fold loop exists for a `bind = :nearest` schedule in `opt` or `fb`, so the constructor refuses one. A fold loop that reaches the estimator resolves each schedule. A solve with no fold loop resets each schedule to its own `default`. A schedule with no `default` resets `pe`, `wf`, `min_solved`, `subset_size` and `n_subsets` to their keyword defaults, and `wb`, `fees`, `sets` and `fb` to `nothing`. A schedule in `opt` with no `default` throws a [`TimeDependentDefaultError`](@ref).

## Validation

  - `opt` passes [`assert_internal_optimiser`](@ref). A schedule applies the check to each entry and to its `default`.
  - If `wb` is keyed by name ([`name_keyed`](@ref)), `sets` is not `nothing`. The constructor throws an [`IsNothingError`](@ref) otherwise.
  - If `fees` is a [`FeesEstimator`](@ref), `sets` is not `nothing`. The constructor throws an [`IsNothingError`](@ref) otherwise.
  - A plain number in `min_solved` or `subset_size` passes [`count_or_fraction`](@ref), and the field then holds a [`CountOf`](@ref) or a [`FractionOf`](@ref).
  - If `n_subsets` is an `Integer`: `n_subsets >= 2`.
  - `max_comb > 0` and finite.
  - A schedule in `opt` or `fb` has `bind !== :nearest`. Each entry of a schedule passes the check of its field.
  - A reader that states no source and reads the raw returns data, beside a reader that states `x_src = ExcessSource()`, logs one warning that names its path, see [`warn_raw_readers`](@ref). The head then builds as configured.

$(val_dict[:count_fraction_grid])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `fees`: Recursively updated through [`factory`](@ref).
  - `opt`: Recursively updated through [`factory`](@ref).
  - `fb`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

`SubsetResampling` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `pe` is a prior result, the method uses `pe.X` in place of `X`.
  - `pe`, `wb`, `sets` and `cache` recurse through [`port_opt_view`](@ref) with the index alone.
  - `fees` and `opt` recurse with the index and `X`.
  - `fb` recurses through [`view_child`](@ref) with the index and `X`.

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `pe`: Recursively indexed through [`obs_weights_view`](@ref).
  - `fees`: Recursively indexed through [`obs_weights_view`](@ref).
  - `opt`: Recursively indexed through [`obs_weights_view`](@ref).
  - `fb`: Recursively indexed through [`obs_weights_view`](@ref).
  - `cache`: Removed by [`obs_weights_view`](@ref), because a state has no slice on the observation axis.

# Related

  - [`optimise`](@ref)
  - [`SubsetResamplingResult`](@ref)
  - [`BaseSubsetResamplingOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)
  - [`subset_resampling_finaliser`](@ref)
  - [`PowerSubsetSize`](@ref)

# References

  - $(ref_dict[:shen2017]) Algorithm 1.
"""
@propagatable @concrete struct SubsetResampling <: BaseSubsetResamplingOptimisationEstimator
    """
    $(field_dict[:pe])
    """
    @oprop pe
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
    $(field_dict[:min_solved])
    """
    min_solved
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
                              wf::TD{<:WeightFinaliser},
                              min_solved::TD{<:Real_CountFraction},
                              ex::FLoops.Transducers.Executor,
                              subset_size::TD{<:SubsetSizeE},
                              n_subsets::TD{<:NumberSubsetsE}, max_comb::Integer,
                              rng::Random.AbstractRNG, seed::Option{<:Integer},
                              fb::TDO_Option{<:OptE_Opt}, strict::Bool,
                              cache::Option{<:ReturnsBufferState})
        assert_no_nearest_bind_optimiser_schedule(opt, :opt, :SubsetResampling)
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :SubsetResampling)
        assert_internal_optimiser(opt)
        if name_keyed(wb)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        min_solved = count_or_fraction(min_solved)
        subset_size = count_or_fraction(subset_size)
        if isa(n_subsets, Integer)
            assert_nonempty_nonneg_finite_val(n_subsets - 2, "n_subsets - 2")
        end
        assert_nonempty_gt0_finite_val(max_comb, :max_comb)
        assert_time_dependent_substitution(SubsetResampling,
                                           (; pe, wb, fees, sets, opt, wf, min_solved, ex,
                                            subset_size, n_subsets, max_comb, rng, seed, fb,
                                            strict), subset_resampling_td_defaults())
        est = new{typeof(pe), typeof(wb), typeof(fees), typeof(sets), typeof(opt),
                  typeof(wf), typeof(min_solved), typeof(ex), typeof(subset_size),
                  typeof(n_subsets), typeof(max_comb), typeof(rng), typeof(seed),
                  typeof(fb), typeof(strict), typeof(cache)}(pe, wb, fees, sets, opt, wf,
                                                             min_solved, ex, subset_size,
                                                             n_subsets, max_comb, rng, seed,
                                                             fb, strict, cache)
        warn_raw_readers(est)
        return est
    end
end
function SubsetResampling(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
                          wb::TD_Option{<:WbE_Wb} = nothing,
                          fees::TD_Option{<:FeesE_Fees} = nothing,
                          sets::TD_Option{<:UniverseSets} = nothing, opt::OptE_TD,
                          wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                          min_solved::TD{<:Real_CountFraction} = FractionOf(1),
                          ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                          subset_size::TD{<:SubsetSizeE} = PowerSubsetSize(),
                          n_subsets::TD{<:NumberSubsetsE} = 2,
                          max_comb::Integer = 1_000_000_000,
                          rng::Random.AbstractRNG = Random.default_rng(),
                          seed::Option{<:Integer} = nothing,
                          fb::TDO_Option{<:OptE_Opt} = nothing, strict::Bool = false,
                          cache::Option{<:ReturnsBufferState} = nothing)::SubsetResampling
    return SubsetResampling(pe, wb, fees, sets, opt, wf, min_solved, ex, subset_size,
                            n_subsets, max_comb, rng, seed, fb, strict, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`SubsetResampling`](@ref) fields that can hold a [`TimeDependent`](@ref).

The constructor passes the defaults to [`assert_time_dependent_substitution`](@ref), and [`time_dependent_field_defaults`](@ref) returns them. `opt` is a required keyword and has no static default, so its entry is [`NoDefault`](@ref). A schedule in `opt` must have its own `default` to run outside a fold loop. `pe`, `wf`, `min_solved`, `subset_size` and `n_subsets` take their keyword defaults. The tuple omits `wb`, `fees`, `sets` and `fb`, because their static default is `nothing`.

# Related

  - [`SubsetResampling`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function subset_resampling_td_defaults()::NamedTuple
    return (; pe = EmpiricalPrior(), opt = NoDefault(), wf = IterativeWeightFinaliser(),
            min_solved = FractionOf(1), subset_size = PowerSubsetSize(), n_subsets = 2)
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
 3. Resolve `opt.opt` and `opt.fb` with this function, and rebuild `opt` with them through [`rebuild_struct`](@ref).

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
    return rebuild_struct(opt,
                          (; opt = update_time_dependent_estimator(opt.opt, ctx, all_binds),
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
                            wf = sr.wf, min_solved = sr.min_solved, ex = sr.ex,
                            subset_size = sr.subset_size, n_subsets = sr.n_subsets,
                            max_comb = sr.max_comb, rng = sr.rng, seed = sr.seed,
                            fb = view_child(sr.fb, i, X), strict = sr.strict,
                            cache = port_opt_view(sr.cache, i))
end
function non_investable_universe(sr::SubsetResampling, ni::VecStr)::SubsetResampling
    return rebuild_struct(sr, (; sets = non_investable_sets(sr.sets, ni)))
end
"""
    subset_resampling_retcode(resi_retcodes::VecOptRetCode, retcode::OptimisationReturnCode,
                              partial::Bool)

Combine the return codes of the subset optimisations with the return code of the weight finaliser.

# Algorithm

 1. Return `retcode` unchanged when no entry of `resi_retcodes` and not `retcode` is a failure.
 2. Otherwise build `msg`, with the line `opti failed.` when a subset failed, and then the line `weight bounds finalisation failed.` when the finaliser failed.
 3. Return a code whose `res` is `(; msg, opti = resi_retcodes, wb = retcode)`. Each return code keeps its own diagnostics. The code is an [`OptimisationSuccess`](@ref) when `partial` holds and the finaliser did not fail, and an [`OptimisationFailure`](@ref) otherwise.

# Arguments

  - `resi_retcodes`: Return codes of the subset optimisations. On an efficient frontier they are the codes of one point.
  - `retcode`: Return code of the weight finaliser.
  - `partial`: Whether the average dropped the subsets that failed, see [`dropped_sub_portfolios`](@ref).

# Returns

  - `retcode::OptimisationReturnCode`: `retcode`, or the code of step 3.

# Related

  - [`subset_resampling_finaliser`](@ref)
  - [`SubsetResampling`](@ref)
"""
function subset_resampling_retcode(resi_retcodes::VecOptRetCode,
                                   retcode::OptimisationReturnCode, partial::Bool)
    resi_flag = !all(any_solved, resi_retcodes)
    wb_flag = !any_solved(retcode)
    if !(resi_flag || wb_flag)
        return retcode
    end
    msg = ""
    if resi_flag
        msg *= "opti failed.\n"
    end
    if wb_flag
        msg *= "weight bounds finalisation failed.\n"
    end
    res = (; msg = msg, opti = resi_retcodes, wb = retcode)
    return if partial && !wb_flag
        OptimisationSuccess(; res = res)
    else
        OptimisationFailure(; res = res)
    end
end
"""
    subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                wb::WeightBounds, wf::WeightFinaliser,
                                min_solved::AbstractCountOrFraction, ress::VecOpt,
                                j::Integer)
    subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                wb::WeightBounds, wf::WeightFinaliser,
                                min_solved::AbstractCountOrFraction, ress::VecOpt,
                                ws::VecVecNum)

Average the subset weights over all the assets, and finalise the average under the weight bounds.

The method computes ``\\boldsymbol{w}^*`` of the `# Mathematical definition` of [`SubsetResampling`](@ref).

On an efficient frontier, a subset whose every point failed can hold the one portfolio of its fallback. That portfolio and its return code stand at every point.

# Algorithm

The method for point `j` does these steps. One portfolio for each subset is the point `1`.

 1. Read the return codes of point `j` of `ress` with [`member_retcode`](@ref), and find the subsets to drop with [`dropped_sub_portfolios`](@ref), giving `dropped`.
 2. Allocate `w`, a vector of ``N`` zeros with the element type of the weights of the first subset.
 3. For each subset `i` that is not in `dropped`, add the weights of point `j` of `ress[i]`, from [`member_vector`](@ref), into `w` at the asset indices `asset_idx[:, i]`.
 4. Divide `w` by the number of these subsets.
 5. Finalise `w` with `wf` under `wb` through [`finalise_weight_bounds`](@ref), giving `retcode` and the finalised `w`.
 6. Combine `retcode` with the return codes of step 1 through [`subset_resampling_retcode`](@ref).

The method for a vector of vectors runs the steps once for each point. Its point `j` reads the codes of the subsets at `j` alone, so one point can drop a subset that another point keeps.

# Arguments

  - `N::Integer`: Number of assets in the universe that the subsets come from.
  - `n_subsets::Integer`: Number of asset subsets.
  - `asset_idx::MatNum`: Asset indices, one column for each subset.
  - `wb::WeightBounds`: Weight bounds of the finaliser.
  - `wf::WeightFinaliser`: Weight finaliser that repairs a bounds violation.
  - `min_solved::AbstractCountOrFraction`: Number of subsets that must solve.
  - `ress::VecOpt`: Subset optimisation results, in the column order of `asset_idx`.
  - `j::Integer`: The point of the efficient frontier, `1` when no subset holds one.
  - `ws::VecVecNum`: Weights of a subset that holds an efficient frontier. The method reads the number of points from it, and no weight.

# Validation

  - On an efficient frontier, every subset result that holds a frontier holds `length(ws)` points, `DimensionMismatch` otherwise.

# Returns

  - `retcode`: The return code from [`subset_resampling_retcode`](@ref), or one for each point of an efficient frontier.
  - `w`: The averaged and finalised weights, or one vector for each point of an efficient frontier.
  - `dropped::BitVector`: The subsets that the average drops, at one or more points of an efficient frontier.

# Related

  - [`SubsetResampling`](@ref)
  - [`dropped_sub_portfolios`](@ref)
  - [`subset_resampling_retcode`](@ref)
  - [`finalise_weight_bounds`](@ref)
  - [`member_vector`](@ref)
  - [`member_retcode`](@ref)
"""
function subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                     wb::WeightBounds, wf::WeightFinaliser,
                                     min_solved::AbstractCountOrFraction, ress::VecOpt,
                                     j::Integer)
    resi_retcodes = [member_retcode(res.retcode, j) for res in ress]
    dropped = dropped_sub_portfolios(min_solved, resi_retcodes)
    w = zeros(eltype(member_vector(ress[1].w, j)), N)
    for i in 1:n_subsets
        if !dropped[i]
            idx = view(asset_idx, :, i)
            w[idx] .+= member_vector(ress[i].w, j)
        end
    end
    w /= n_subsets - count(dropped)
    retcode, w = finalise_weight_bounds(wf, wb, w)
    return subset_resampling_retcode(resi_retcodes, retcode, any(dropped)), w, dropped
end
function subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                     wb::WeightBounds, wf::WeightFinaliser,
                                     min_solved::AbstractCountOrFraction, ress::VecOpt,
                                     ws::VecVecNum)
    M = length(ws)
    @argcheck(all(res -> !isa(res.w, VecVecNum) || length(res.w) == M, ress),
              DimensionMismatch("every subset that answers with an efficient frontier must have the same number of points.\nnumber of points of each subset => $([isa(res.w, VecVecNum) ? length(res.w) : 1 for res in ress])"))
    points = [subset_resampling_finaliser(N, n_subsets, asset_idx, wb, wf, min_solved, ress,
                                          j) for j in 1:M]
    return map(x -> x[1], points), map(x -> x[2], points),
           mapreduce(x -> x[3], (a, b) -> a .| b, points)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the subset resampling optimisation, and return the averaged portfolio.

[`optimise`](@ref) calls this method. The struct [`SubsetResampling`](@ref) states the mathematics.

# Algorithm

 1. Reset the schedules of `sr` with [`reset_time_dependent_estimator`](@ref).
 2. Fit the prior `sr.pe` on `rd` with [`prior`](@ref), giving `pr`, and read its investable mask with [`investable_mask`](@ref), giving `imsk`. The prior reads the returns in the frame that its own `x_src` states. `rd` keeps its raw returns, and the readers of the subset optimisers state their own sources.
 3. Resolve the fees on the full asset universe with [`fees_constraints`](@ref), and put them on the investable universe with [`investable_fees_view`](@ref), giving `fees`.
 4. Reduce `pr`, `sr` and `rd` to the investable universe with [`investable_reduction`](@ref), giving `rdr`. `N` is the number of investable assets.
 5. Resolve `subset_size` and `n_subsets` on `pr` with [`get_subset_size`](@ref) and [`get_n_subsets`](@ref).
 6. Draw the subsets with [`sample_unique_assets`](@ref), giving `asset_idx`. When `binomial(N, subset_size)` is at most `max_comb`, the subsets are distinct. Otherwise the function draws each subset alone, and two subsets can be equal.
 7. For each column of `asset_idx`, view `sr.opt` and `rdr` on the subset with [`port_opt_view`](@ref) and optimise the view, giving `ress`. The loop runs on the executor `sr.ex`.
 8. Build the weight bounds `wb` over the ``N`` assets with [`weight_bounds_constraints`](@ref).
 9. Average and finalise the subset weights with [`subset_resampling_finaliser`](@ref), giving `retcode` and `w`. The first subset result that holds an efficient frontier selects its frontier method. Name the subsets that the average drops in one warning.
10. Return a [`SubsetResamplingResult`](@ref). Its keyword constructor expands `w` onto the full asset universe.

# Arguments

  - `sr`: The subset resampling optimiser to use.
  - $(arg_dict[:rd])
  - `branchorder`: Branch order, passed to each subset optimisation.
  - `str_names`: Whether each subset optimisation uses string names for the assets.
  - `save`: Whether each subset optimisation saves its JuMP model in its result.
  - `kwargs`: Other keyword arguments, passed to each subset optimisation.

# Validation

  - `n_subsets <= binomial(N, subset_size)`. [`sample_unique_assets`](@ref) throws an `ArgumentError` otherwise.

# Returns

  - `res::SubsetResamplingResult`: The averaged portfolio. `retcode` is an [`OptimisationFailure`](@ref) when fewer subset optimisations solve than `sr.min_solved` asks for, or when the weight finaliser fails. On an efficient frontier it holds one code for each point. When the average of a point drops a subset, the `res` of its code holds the return code of each subset.

# Related

  - [`SubsetResampling`](@ref)
  - [`SubsetResamplingResult`](@ref)
  - [`subset_resampling_finaliser`](@ref)
  - [`optimise`](@ref)
"""
function _optimise(sr::SubsetResampling, rd::ReturnsResult; branchorder::Symbol = :optimal,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    sr = reset_time_dependent_estimator(sr)
    pr = prior(sr.pe, rd)
    # A weight and a fee take the element type of the returns.
    Tf = eltype(pr.X)
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
    _, pr, sr, rdr = investable_reduction(pr, sr, rd, imsk)
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
    wb = weight_bounds_constraints(sr.wb, sr.sets; rd = rdr, N = N, strict = sr.strict,
                                   datatype = Tf)
    # A subset whose every point failed answers with the one portfolio of its fallback, so
    # the first subset does not state whether the others hold a frontier.
    i = findfirst(res -> isa(res.w, VecVecNum), ress)
    points = isnothing(i) ? 1 : ress[i].w
    retcode, w, dropped = subset_resampling_finaliser(N, n_subsets, asset_idx, wb, sr.wf,
                                                      sr.min_solved, ress, points)
    warn_dropped_sub_portfolios(dropped)
    return SubsetResamplingResult(; pr = pr, wb = wb, fees = fees, ress = ress,
                                  idx = asset_idx, retcode = retcode, w = w, imsk = imsk,
                                  fb = nothing)
end
"""
    group_frames(opt::NestedClustered) -> ReturnsFrames
    group_frames(opt::Union{BaseStackingOptimisationEstimator, BaseSubsetResamplingOptimisationEstimator}) -> ReturnsFrames

Walk the readers of a meta-optimiser, for the warning of a forgotten raw reader and the gap rule of the head.

The group of a meta-optimiser is the head and every reader in its own slots: the prior `pe`, and the clustering `cle` of a [`NestedClustered`](@ref). [`field_frames`](@ref) walks every field. The walk stops at the inner optimisers, the outer optimiser and the fallback, because each one is a head with its own group.

# Arguments

  - `opt`: The meta-optimiser.

# Returns

  - `fr::ReturnsFrames`: The frames of the group of `opt`.

# Related

  - [`field_frames`](@ref)
  - [`warn_raw_readers`](@ref)
  - [`group_source`](@ref)
"""
function group_frames(opt::NestedClustered)
    return field_frames(opt, "", refits_regression(opt.pe))
end
function group_frames(opt::Union{BaseStackingOptimisationEstimator,
                                 BaseSubsetResamplingOptimisationEstimator})
    return field_frames(opt, "", refits_regression(opt.pe))
end

export SubsetResamplingResult, PowerSubsetSize, SubsetResampling
