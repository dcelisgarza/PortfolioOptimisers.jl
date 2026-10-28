"""
$(DocStringExtensions.TYPEDEF)

Updates its allocation from each realised price relative through the Online Selection Rule on `alg`.

The head is a naive optimiser. It fits no moment and solves no programme of its own. Every rule of the family is one recursion, which rebalances the allocation to its target at the start of every period, and the head does all the work that the rules share. `# Mathematical definition` states the recursion and the wealth it earns.

**The batch verb is the Causal Pass.** `optimise(opt, rd)` starts from the Start Allocation `w0`, applies the Online Update to every row of `rd` in order, and returns the **Next-Period Allocation**. After rows `1:T` that is the portfolio for period `T + 1`, so the pass takes one more update than a backtest driver that stops at the last row it holds. The pass rebalances at every row and pays no fee. `predict(res, rd)` on its Result charges the head's `fees`, with no drift.

**The online verbs run the same recursion.** `partial_fit!(opt, rd)` folds each row into the state as the Block Step. A fold of `k` rows is `k` single-row updates, so a walk-forward with `test_size = k` holds the Next-Period Allocation of the end of the block for `k` periods. `optimise(opt)` with no data is the **Recursion Result**. It wraps the allocation of the state in a Result and runs no batch path. So `optimise(opt)` after a fold of rows `1:t` equals `optimise(opt, rd[1:t])` exactly, at every `test_size`. The held path of a walk-forward differs from the batch pass in two ways. The target set at the start of a block drifts over the block, and the Block Step sets the cadence of the trades.

**The Start Allocation** is `w0`, `nothing` by default. When it is absent, the recursion starts at `1/N` over the whole pinned universe, and the unlisted names are included. If `k` of `N` assets are unlisted at the first row, the recursion puts `k/N` of its weight on legs that see `x = 1`, as cash does, until the rule moves that weight. `optimise(opt)` with no data renormalises over the listed legs, so the fund holds the listed legs alone. But the fund is not always the recursion that starts over the listed names alone, because the parked legs enter the gross return `⟨w, x⟩` that a rule reads. Buy-and-hold scales each leg by its own price relative, so its fund is the same. The exponentiated-gradient step divides by the gross return, so its fund is different. A caller who wants the recursion to be the fund from row one gives `w0` over the listed assets, with zeros elsewhere. The rule's step then decides what a zero does. The exponentiated-gradient step keeps a zero at zero, so that asset never gets weight. A Euclidean step that moves no weight onto the asset keeps the zero exact too, because [`project_simplex`](@ref) only renormalises a raw step that is on the simplex up to rounding. Buy-and-hold is such a step. A Euclidean step that moves weight onto the asset, as the passive-aggressive step does, gives it weight, and a gap at that asset is then a Held Gap.

A given `w0` is over the pinned names, and the head pins and views it with them. The first step projects it once onto the Allocation Set in the rule's geometry, as it projects the uniform start, so a start outside the set becomes feasible and is never refused. A set that reads the head's rows has no constraints before the first row, so it holds the start as given until the first update. The start reaches the first Online Update as `w`. A rule that steps from `w` continues from it. A rule whose next allocation is not a step from `w` replaces it at that update. The constant rebalanced portfolio returns its own `w`, an expert mixture returns the mix of its experts, and the Newton step solves its Gram. Such a rule can still carry `w0` forward. The gradient of the Newton step reads `w`, and an expert that steps from `w` starts at `w0`. Under every rule, the fund holds `w0` for exactly one period.

**The Online Update reads the recursion's own allocation, and no flag changes that.** The previous weights of the loop reach the head through [`factory`](@ref) alone. They go into `fees` and `fb`, and never into the recursion. The fund's held book is the one base against which a turnover fee on this family measures a trade. So the online arm of the fold loop refuses a head whose `fees` carry a `tn` term when the Previous-Weights Source of the walk-forward is `nothing`. It throws an `ArgumentError` before any fold, and `pws = DriftedWeights()` is the whole configuration. A head with no `tn` fee is not checked.

**Under `x_src = ExcessSource()` the head selects in the Excess Frame.** The rule's step reads the price relative of the return in excess of the benchmark, `x = 1 + r - b`. Here `b` is the benchmark return of the row, one value that every asset shares for a vector benchmark, and one value per asset for a matrix benchmark. Every reader of the rows inside a rule reads the same frame through [`head_frame_returns`](@ref): the refit and the fold of a forecaster, the scale of a [`ForecastReversion`](@ref), the window of an [`AntiCorrelation`](@ref) and the row selector of a [`FollowTheLeader`](@ref). The rows buffer keeps the raw returns and the benchmark, so the prior of a Risk Loss, the prior and each graph or exposure slot of a programme set, and a leader read the raw rows and state their own source. The source of the step, the readers of a Risk Loss and the readers of a programme set form one group. When one of them states `ExcessSource()` and another states no source and reads the raw rows, the head warns once, see [`warn_raw_readers`](@ref). A leader and the fallback are heads with their own groups. A gap in the benchmark is a gap in the frame, which the step reads as one and the head reports as a Held Gap. `predict(res, rd)` reports the raw wealth of the fund, because the fund holds the assets. Returns data with no benchmark is refused with an `ArgumentError`.

**A non-finite return is kept, and read in two ways.** The rule's step reads `x = 1` there, as if the leg held cash. At an asset that the panel marks inactive, or at one to which the recursion gives no weight, this is silent. At an active asset with a non-zero weight it is a Held Gap, which warns by default and refuses under `strict`. The rows buffer keeps the cell as `NaN`, with the row's active mask beside it. Every statistic over the rows reads them as the batch verb reads a Returns Result, and reduces to its own Coverage Universe. Such a statistic is the mean of a forecaster, the prior of a Risk Loss or of a programme set, or the re-solve of a leader. A plain estimator drops an asset with a gap anywhere in its window. A mask-aware estimator answers the asset from the rows it has. A kernel over price relatives reads the gap as one.

The head never forces a zero into the recursion's allocation, and a relisted asset comes back at the recursion's own weight. A programme set that fits on the rows writes a zero at a leg that its prior cannot price, as a batch head does. It admits the leg again when the prior can price it. Under a time-varying panel, the Investable Mask of the Recursion Result is the active mask of the last folded row. `optimise(opt)` with no data slices the full allocation to that mask and renormalises it, and the Result expands it back with a zero at every non-investable asset.

**The Allocation Set on `set` is the one the Constrained Update projects onto.** [`project`](@ref) projects the raw step of every rule onto it, in the rule's own Projection Geometry. The default [`BoundedAllocationSet`](@ref) is the simplex, and every projection onto it is closed form, so the default configuration solves nothing. A [`ProgrammeAllocationSet`](@ref) admits the full constraint vocabulary, and its projection is a programme. When a programme fails, the step is a **Held Step**. The projection returns the Price-Adjusted Allocation it was given, so the fund trades nothing that period, and the state of the rule still absorbs the row. The head warns once, with the row's timestamp. The retcode of the Recursion Result is an [`OptimisationSuccess`](@ref) that carries the [`HeldStep`](@ref) record of the last folded row, so a fallback chain never runs on a hold. The constructor refuses a negative lower bound under an entropic, Tsallis or log-barrier rule.

`merge_states` on the state and `Online(head)` each throw an `ArgumentError`. An update depends on the order of the rows, and the family never refits from a buffer.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}_1 &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{w}_0)\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q})\\,, \\\\
x_{tj} &= \\begin{cases} 1 + r_{tj} & \\text{if } r_{tj} \\text{ is finite}\\,, \\\\ 1 & \\text{otherwise}\\,, \\end{cases} \\\\
S_T &= \\prod_{t=1}^{T} \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_1_start])
  - ``\\boldsymbol{w}_0``: Given start, or ``\\boldsymbol{1}/N`` when none is given.
  - $(math_dict[:w_t_iter])
  - $(math_dict[:q_raw]) The rule makes it from ``\\boldsymbol{w}_t`` and ``\\boldsymbol{x}_t``.
  - $(math_dict[:W_aset])
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:Psi_pot])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:r_tj])
  - ``S_T``: Wealth of the fund after ``T`` periods, from a wealth of one.
  - $(math_dict[:t_period])
  - $(math_dict[:T])
  - $(math_dict[:N])

The wealth ``S_T`` is the wealth of Equation 1.1 of [cover1991](@cite), earned in the protocol of Section 2 of [lihoi2014](@cite). The Next-Period Allocation after ``T`` rows is ``\\boldsymbol{w}_{T+1}``. Both projections are one call of [`project`](@ref), so a [`ProgrammeAllocationSet`](@ref) adds its Objective Penalty to both objectives.

Under `x_src = ExcessSource()`, ``r_{tj}`` is the return of asset ``j`` in excess of the benchmark, ``r_{tj} - b_{tj}``, where ``b_{tj}`` is the benchmark return of period ``t`` for asset ``j``. A vector benchmark gives every asset the same ``b_{tj}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OnlinePortfolioSelection(;
        alg::AbstractOnlinePortfolioSelectionAlgorithm = ExponentiatedGradient(),
        set::TD{<:AbstractAllocationSet} = BoundedAllocationSet(),
        w0::Option{<:AbstractVector} = nothing,
        fees::TD_Option{<:FeesE_Fees} = nothing,
        fb::TDO_Option{<:OptE_Opt} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing,
        strict::Bool = false,
        cache::Option{<:OnlinePortfolioSelectionState} = nothing
    ) -> OnlinePortfolioSelection

Keywords correspond to the struct's fields. `set`, `fees` and `fb` can hold a [`TimeDependent`](@ref) per-fold schedule. `alg`, `w0`, `x_src` and `strict` are static.

## Validation

  - `w0`: non-empty and finite, when given. It need not lie in the set, because the first step projects it onto the set.
  - If `fees` is a [`FeesEstimator`](@ref): `!isnothing(set.sets)`. A schedule on `set` is held to it per entry.
  - Everything [`assert_geometry_admits_set`](@ref) refuses: a negative lower bound under an entropic, Tsallis or log-barrier rule. A schedule on `set` is held to it per entry.
  - Everything [`assert_rule_admits_set`](@ref) refuses: a solver-free leader under a programme set. A schedule on `set` is held to it per entry.
  - `fb` schedules: `bind !== :nearest`.
  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref). The step reads no prior, so it refuses `PriorSource()`.
  - A reader that states no source and reads the raw returns data, beside a reader that states `x_src = ExcessSource()`, logs one warning that names its path, see [`warn_raw_readers`](@ref). The head then builds as configured.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these fields:

  - `fees`: Recursively updated through [`factory`](@ref), which threads the previous weights into a turnover fee.
  - `fb`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of the fields with a per-asset axis at the selected indices: `alg` and `set` recursively, `w0` sliced and renormalised, `fees` viewed at the indices, and `cache` through the state's own view.

# Examples

```jldoctest
julia> OnlinePortfolioSelection()
OnlinePortfolioSelection
     alg ┼ MirrorDescent
         │     eta ┼ Float64: 0.05
         │    proj ┼ EntropicProjection()
         │   alpha ┼ Int64: 0
         │     obj ┼ LogWealth()
         │    grad ┴ nothing
     set ┼ BoundedAllocationSet
         │     wb ┼ WeightBounds
         │        │   lb ┼ Float64: 0.0
         │        │   ub ┴ Float64: 1.0
         │   sets ┴ nothing
      w0 ┼ nothing
    fees ┼ nothing
      fb ┼ nothing
   x_src ┼ nothing
  strict ┴ Bool: false
```

# Related

  - [`NaiveOptimisationEstimator`](@ref)
  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelectionState`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
  - [`HeldStep`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`optimise`](@ref)
  - [`partial_fit!`](@ref)

# References

  - $(ref_dict[:lihoi2014]) Section 2, Algorithm 1.
  - $(ref_dict[:cover1991]) Section 1, Equation 1.1.
"""
@concrete struct OnlinePortfolioSelection <: NaiveOptimisationEstimator
    """
    The Online Selection Rule: the update, its parameters and its Projection Geometry.
    """
    alg
    """
    The Allocation Set every allocation of the recursion lies in, or a [`TimeDependent`](@ref) schedule of one set per fold. Entry `i` is the complete set of fold `i`. The fold loop puts it in place before it folds the rows of fold `i`, so those rows step inside entry `i`, and the Recursion Result reports it.
    """
    set
    """
    The Start Allocation over the pinned universe, or `nothing` for `1/N` over every pinned name.
    """
    w0
    """
    $(field_dict[:fees])
    """
    fees
    """
    $(field_dict[:fb])
    """
    fb
    """
    $(field_dict[:x_src_ops])
    """
    x_src
    """
    $(field_dict[:strict_opt]) The flag reaches every constraint of the set, the fee and `optimise(opt)` with no data, and the constraints that a programme set resolves on the rows at each step. It also governs a Held Gap, which warns when it is `false` and raises an error when it is `true`.
    """
    strict
    """
    The Partial Fit State of the recursion, `nothing` until [`partial_fit!`](@ref) writes one. [`factory`](@ref) carries it unchanged and [`port_opt_view`](@ref) slices it to the selected assets.
    """
    cache
    function OnlinePortfolioSelection(alg::AbstractOnlinePortfolioSelectionAlgorithm,
                                      set::TD{<:AbstractAllocationSet},
                                      w0::Option{<:AbstractVector},
                                      fees::TD_Option{<:FeesE_Fees},
                                      fb::TDO_Option{<:OptE_Opt},
                                      x_src::Option{<:AbstractReturnsSource}, strict::Bool,
                                      cache::Option{<:OnlinePortfolioSelectionState})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :OnlinePortfolioSelection)
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :OnlinePortfolioSelection)
        if !isa(set, TimeDependent)
            # A schedule holds no set of its own, so the two set refusals and the fee's
            # read of `set.sets` run per entry, in the substitution pass below.
            assert_geometry_admits_set(projection_geometry(alg), set)
            assert_rule_admits_set(alg, set)
            if isa(fees, FeesEstimator)
                @argcheck(!isnothing(set.sets),
                          IsNothingError("set.sets cannot be nothing when fees is a FeesEstimator"))
            end
        end
        if !isnothing(w0)
            assert_nonempty(w0, :w0)
            assert_finite(w0, :w0)
        end
        assert_time_dependent_substitution(OnlinePortfolioSelection,
                                           (; alg, set, w0, fees, fb, x_src, strict),
                                           online_portfolio_selection_td_defaults())
        opt = new{typeof(alg), typeof(set), typeof(w0), typeof(fees), typeof(fb),
                  typeof(x_src), typeof(strict), typeof(cache)}(alg, set, w0, fees, fb,
                                                                x_src, strict, cache)
        warn_raw_readers(opt)
        return opt
    end
end
function OnlinePortfolioSelection(;
                                  alg::AbstractOnlinePortfolioSelectionAlgorithm = ExponentiatedGradient(),
                                  set::TD{<:AbstractAllocationSet} = BoundedAllocationSet(),
                                  w0::Option{<:AbstractVector} = nothing,
                                  fees::TD_Option{<:FeesE_Fees} = nothing,
                                  fb::TDO_Option{<:OptE_Opt} = nothing,
                                  x_src::Option{<:AbstractReturnsSource} = nothing,
                                  strict::Bool = false,
                                  cache::Option{<:OnlinePortfolioSelectionState} = nothing)::OnlinePortfolioSelection
    return OnlinePortfolioSelection(alg, set, w0, fees, fb, x_src, strict, cache)
end
"""
    group_frames(opt::OnlinePortfolioSelection) -> ReturnsFrames
    reader_frames(alg::AbstractOnlinePortfolioSelectionAlgorithm, path::AbstractString,
                  refit::Bool) -> ReturnsFrames

Walk the group of an [`OnlinePortfolioSelection`](@ref) head, for the warning of a forgotten raw reader and the benchmark names that the state pins.

The group is the source of the step, the readers of each [`RiskLoss`](@ref) in the rule tree, and the readers of a [`ProgrammeAllocationSet`](@ref) on `set`. The step is the outermost reader of the rows, so at `x_src = nothing` it reads the raw returns data, and the walk names it by the type of the head. [`field_frames`](@ref) walks every field of the head and of each rule. A rule over rules walks its sub-rules, and a vector of rules walks each one. The walk stops at a leader and at the fallback, because each is a head with its own group. It stops at a forecaster and at a sample selector too: these read the frame of the step through [`head_frame_returns`](@ref), so they are inner readers and state no source. A Risk Loss and a programme set walk their own fields under the refit rule of their own prior, see [`refits_regression`](@ref).

# Arguments

  - `opt`: The head.
  - `alg`: An Online Selection Rule.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of the group of `opt`, or the frames that `alg` adds.

# Related

  - [`field_frames`](@ref)
  - [`source_frames`](@ref)
  - [`warn_raw_readers`](@ref)
  - [`group_source`](@ref)
"""
function group_frames(opt::OnlinePortfolioSelection)
    return merge_frames(source_frames(opt.x_src, "OnlinePortfolioSelection", true),
                        field_frames(opt, "", false))
end
function reader_frames(alg::AbstractOnlinePortfolioSelectionAlgorithm, path::AbstractString,
                       refit::Bool)
    return field_frames(alg, path, refit)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`OnlinePortfolioSelection`](@ref) fields that may hold a [`TimeDependent`](@ref).

The constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref) share it, so the fold-less value of each field is declared once. `fees` and `fb` default to `nothing` and are left out. `set` has a default, so a fold-less solve runs the default Allocation Set and not an absent set.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function online_portfolio_selection_td_defaults()::NamedTuple
    return (; set = BoundedAllocationSet())
end
function time_dependent_field_defaults(::OnlinePortfolioSelection)::NamedTuple
    return online_portfolio_selection_td_defaults()
end
"""
    factory(opt::OnlinePortfolioSelection, w::VecNum)

Thread the previous fold's weights into the head's fee and fallback, and nowhere else.

The recursion never reads them, so the method carries `alg`, `set`, `w0`, `x_src`, `strict` and `cache` unchanged.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`factory`](@ref)
"""
function factory(opt::OnlinePortfolioSelection, w::VecNum)
    return OnlinePortfolioSelection(; alg = opt.alg, set = opt.set, w0 = opt.w0,
                                    fees = factory(opt.fees, w), fb = factory(opt.fb, w),
                                    x_src = opt.x_src, strict = opt.strict,
                                    cache = opt.cache)
end
"""
    port_opt_view(opt::OnlinePortfolioSelection, i, args...)

Slices the head to the selected assets.

The rule, the set and the fee take their own views. The Start Allocation is sliced and renormalised, and the state takes the view of [`port_opt_view`](@ref) on [`OnlinePortfolioSelectionState`](@ref).

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(opt::OnlinePortfolioSelection, i, args...)
    return OnlinePortfolioSelection(; alg = port_opt_view(opt.alg, i, args...),
                                    set = port_opt_view(opt.set, i, args...),
                                    w0 = renormalised_view(opt.w0, i),
                                    fees = port_opt_view(opt.fees, i, args...),
                                    fb = view_child(opt.fb, i, args...), x_src = opt.x_src,
                                    strict = opt.strict,
                                    cache = port_opt_view(opt.cache, i, args...))
end
function show_fields(opt::OnlinePortfolioSelection)
    return filter(!=(:cache), fieldnames(typeof(opt)))
end
function non_investable_universe(opt::OnlinePortfolioSelection, ::VecStr)
    return opt
end
"""
    rows_needed(td::TimeDependent)

Returns the number of rows that a per-fold schedule of Allocation Sets needs.

The count is the maximum over every entry of a vector schedule and over its explicit `default`. A callable schedule answers `nothing`, which is unbounded. The seed sizes the rows buffer once and never resizes it, so the cap must answer for every fold, and not only for the fold in which the seed runs. The entries of a callable do not exist before their fold, so the cap keeps every row folded so far, and no set of a fold can read a row that the cap dropped.

# Related

  - [`rows_needed`](@ref)
  - [`TimeDependent`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
"""
function rows_needed(td::TimeDependent)
    v = td.val
    if !isa(v, AbstractVector)
        return nothing
    end
    need = mapreduce(rows_needed, rows_needed_max, v)
    d = td.default
    return isa(d, AbstractAllocationSet) ? rows_needed_max(need, rows_needed(d)) : need
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the number of rows that the head's buffer keeps, the maximum over the rule tree and the Allocation Set.

A count of `nothing` is unbounded, and it wins the maximum. A schedule on `set` answers over every entry it holds, because the seed sizes the buffer once and every fold reads it.

# Related

  - [`rows_needed`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
"""
function rows_needed(opt::OnlinePortfolioSelection)
    return rows_needed_max(rows_needed(opt.alg), rows_needed(opt.set))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the head's Allocation Set outside every fold loop, which is its own set or the fold-less value of a schedule on `set`.

A schedule is defined over the folds of a cross-validation scheme. So a [`partial_fit!`](@ref) that a caller takes by hand steps inside the `default` of the schedule, or inside [`BoundedAllocationSet`](@ref) when the schedule has no `default`. The Causal Pass steps inside the same set, because its reset of the schedules has already run.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`fold_online_selection`](@ref)
  - [`time_dependent_reset_value`](@ref)
"""
function static_allocation_set(opt::OnlinePortfolioSelection)
    set = opt.set
    return if isa(set, TimeDependent)
        time_dependent_reset_value(set, online_portfolio_selection_td_defaults(), :set, opt)
    else
        set
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Seeds the head's state on its first row, before the first update.

The uniform start and a given `w0` meet the set in the same way. So the allocation held during the first period lies in the set wherever the set can be formed without rows. A set that reads the head's rows, such as a programme set with a fitted risk ceiling or a tracking error, has no constraints before the first row. The seed then holds the start as given, and the first Online Update projects it. That update holds every step until the head has two rows. A programme that fails at the start holds the start as given and warns, as the Held Step of a row does.

# Algorithm

 1. Form `start`, which is `1/N` at each of the `N` pinned assets when `w0` is `nothing`, and `w0` otherwise.
 2. Project `start` once onto `set` in the rule's geometry through [`project_start`](@ref), giving `w0`. A set that reads rows returns `start` unchanged.
 3. Seed the state of the rule on `w0` and `set` through [`rule_state_seed`](@ref), giving `st`. `st` is the state of the rule: what the rule keeps from one update to the next, such as a Gram matrix or a belief covariance, or `nothing` for a rule that keeps nothing. Steps 2 and 3 run inside one [`with_projection_step`](@ref) under the head's `strict`, which collects the holds in `held`.
 4. Warn on a hold through [`report_held_steps`](@ref).
 5. Size the rows buffer `X`, a [`ReturnsBufferState`](@ref) that keeps the rows with every other field of the Returns Result, from [`rows_needed`](@ref). A count of `nothing` gives an uncapped buffer, a count of zero gives no buffer, and any other count caps the buffer at that count.
 6. Keep the panel of `rd` as `pnl` when it is static, and `nothing` otherwise. Keep the benchmark names of `rd` as `nb` when a reader of the head's group states [`ExcessSource`](@ref), and `nothing` otherwise, through [`pinned_benchmark_names`](@ref).
 7. Return the state with no row folded, the allocation `w0`, the state `st` of the rule, the buffer `X`, the names, `nb` and `pnl`.

# Arguments

  - `opt`: The head.
  - `rd`: The returns data of the first block.
  - `set`: The Allocation Set, resolved over the pinned universe.

# Validation

  - `length(opt.w0) == size(rd.X, 2)`, when `w0` is given. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::OnlinePortfolioSelectionState`: The state before the first update.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`OnlinePortfolioSelectionState`](@ref)
"""
function online_selection_seed(opt::OnlinePortfolioSelection, rd::ReturnsResult,
                               set::AbstractAllocationSet)
    N = size(rd.X, 2)
    start = if isnothing(opt.w0)
        fill(one(eltype(rd.X)) / N, N)
    else
        @argcheck(length(opt.w0) == N,
                  DimensionMismatch("w0 ($(length(opt.w0))) must have one entry per pinned asset ($N)"))
        opt.w0
    end
    # A set that reads the head's rows cannot form its constraints before the first row, so
    # the start is held as given and the first Online Update projects it; every other set
    # meets the start once, uniform or given. The rule's seed projects its experts' starts
    # in the same step, and a hold at the start is warned as a row's is.
    (w0, st), held = with_projection_step(nothing, nothing; strict = opt.strict) do
        w = project_start(projection_geometry(opt.alg), set, start)
        return w, rule_state_seed(opt.alg, w, set)
    end
    report_held_steps(held, "the start")
    need = rows_needed(opt)
    X = if isnothing(need)
        ReturnsBufferState()
    elseif iszero(need)
        nothing
    else
        ReturnsBufferState(; max_history = need)
    end
    static = isnothing(rd.pnl) || panel_is_static(rd.pnl)
    return OnlinePortfolioSelectionState(; n = 0, w = w0, st = st, X = X, nx = rd.nx,
                                         nb = pinned_benchmark_names(opt, rd),
                                         pnl = static ? rd.pnl : nothing, amsk = nothing,
                                         ts = nothing)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every row of a Returns Result into the head's state, in order.

This is the Block Step, and a fold of the whole Returns Result from no state is the Causal Pass.

# Algorithm

 1. Read the returns of the block in the head's frame through [`benchmark_relative_returns`](@ref), giving `R`. `R` is `rd.X - rd.B` when the head's `x_src` is [`ExcessSource`](@ref), and `rd.X` otherwise.
 2. Read the active mask of `rd` through [`step_active_mask`](@ref), giving `amsk`.
 3. Resolve the Allocation Set over the pinned universe through [`resolve_allocation_set`](@ref), giving `set`.
 4. Seed the state on the first block, or pin and check the carried state, through [`online_selection_pin`](@ref), giving `state`.
 5. Fold each row in order through [`online_selection_row!`](@ref), under the timestamp that [`row_timestamp`](@ref) gives it. The row's returns are its row of `R`, and the row of the raw returns data is viewed as a Returns Result of one observation through [`port_opt_view`](@ref). Each row updates the state `st` of the rule, the allocation `w`, the buffer `X` and the hold record `hold`, and adds one to the row count `n`. `st` is the state of the rule: what the rule keeps from one update to the next, such as a Gram matrix or a belief covariance, or `nothing` for a rule that keeps nothing.
 6. Return the state with the new `n`, `w`, `st`, `X` and `hold`, the active mask of the last row, and the timestamps folded so far.

# Arguments

  - `opt`: The head.
  - `cache`: The state, or `nothing` before the first row.
  - `rd`: The returns data of the block, `observations × assets`.
  - `set`: The block's Allocation Set, before resolution over the pinned universe. It defaults to the head's own, and the fold loop's online arm passes the fold's entry of a schedule (see [`online_step_fold`](@ref)).

# Validation

  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - The head's `x_src` is not [`ExcessSource`](@ref), or `rd` carries a benchmark. An `ArgumentError` is thrown otherwise, through [`benchmark_relative_returns`](@ref).
  - Everything [`step_active_mask`](@ref) and [`assert_pinned_context`](@ref) refuse.
  - A non-finite cell at an active asset, under `strict`. An `ArgumentError` is thrown.

# Returns

  - `state::OnlinePortfolioSelectionState`: The state after the last row.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`online_selection_pin`](@ref)
  - [`online_selection_row!`](@ref)
  - [`online_update!`](@ref)
"""
function fold_online_selection(opt::OnlinePortfolioSelection,
                               cache::Option{<:OnlinePortfolioSelectionState},
                               rd::ReturnsResult,
                               set::AbstractAllocationSet = static_allocation_set(opt))
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    R = benchmark_relative_returns(rd, opt.x_src)
    amsk = step_active_mask(rd)
    set = resolve_allocation_set(set, size(rd.X, 2), opt.strict, eltype(rd.X))
    state = online_selection_pin(opt, cache, rd, set)
    w, st, X, n = state.w, state.st, state.X, state.n
    last_amsk = state.amsk
    hold = state.hold
    for t in axes(rd.X, 1)
        row = (; r = vec(R[t, :]), amsk = isnothing(amsk) ? nothing : vec(amsk[t, :]),
               ts = row_timestamp(rd.ts, t, n), i = n + 1, rd = port_opt_view(rd, t:t, :))
        st, w, X, hold = online_selection_row!(opt, st, w, X, row, rd.nx, set)
        n += 1
        last_amsk = row.amsk
    end
    last_amsk = isnothing(last_amsk) ? nothing : BitVector(last_amsk)
    return OnlinePortfolioSelectionState(; n = n, w = w, st = st, X = X, nx = state.nx,
                                         nb = state.nb, pnl = state.pnl, amsk = last_amsk,
                                         ts = fold_column(state.ts, rd.ts, nothing),
                                         hold = hold)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the state that a block folds into.

On the first block the method seeds the state through [`online_selection_seed`](@ref). On every later block it returns the state that the head carries, after the pin-and-check that every estimator runs at its online step. The check compares the names and the static panel with those of the first step, and the width with that of the state. When a reader of the head's group states [`ExcessSource`](@ref), it compares the benchmark names too, because that reader reads the excess over that benchmark. The timestamps must be present at every block or at none.

# Validation

  - `length(cache.w) == size(rd.X, 2)`. A `DimensionMismatch` is thrown otherwise.
  - Everything [`assert_pinned_context`](@ref) and [`assert_column_presence`](@ref) refuse.

# Related

  - [`fold_online_selection`](@ref)
  - [`online_selection_seed`](@ref)
"""
function online_selection_pin(opt::OnlinePortfolioSelection, ::Nothing, rd::ReturnsResult,
                              set::AbstractAllocationSet)
    return online_selection_seed(opt, rd, set)
end
function online_selection_pin(opt::OnlinePortfolioSelection,
                              cache::OnlinePortfolioSelectionState, rd::ReturnsResult,
                              ::AbstractAllocationSet)
    N = size(rd.X, 2)
    @argcheck(length(cache.w) == N,
              DimensionMismatch("the state holds $(length(cache.w)) assets and this block carries $N"))
    assert_pinned_context(cache.nx, rd.nx, :nx)
    assert_pinned_context(cache.nb, pinned_benchmark_names(opt, rd), :nb)
    static = isnothing(rd.pnl) || panel_is_static(rd.pnl)
    assert_pinned_context(cache.pnl, static ? rd.pnl : nothing, :pnl)
    if !iszero(cache.n)
        assert_column_presence(cache.ts, rd.ts, :ts)
    end
    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the benchmark names that the state of the head pins: the names of `rd` when a reader of the head's group states [`ExcessSource`](@ref), and `nothing` otherwise.

A reader of the Excess Frame reads the excess over the benchmark of the rows, so the names of that benchmark must stay the same from one block to the next. The reader can be the step, the prior of a [`RiskLoss`](@ref), or a reader of a [`ProgrammeAllocationSet`](@ref). A group that reads no benchmark pins no names, so a change of benchmark between two blocks passes.

# Related

  - [`online_selection_seed`](@ref)
  - [`online_selection_pin`](@ref)
  - [`group_source`](@ref)
"""
function pinned_benchmark_names(opt::OnlinePortfolioSelection, rd::ReturnsResult)
    return isnothing(group_source(opt)) ? nothing : rd.nb
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Takes one row of the recursion, inside a [`ProjectionStep`](@ref) that gives every projection of the row the rows it reads and a log for its holds.

The row's returns `r` are in the head's frame, in excess of the benchmark under `x_src = ExcessSource()`. The row's Returns Result `rd` is raw. The row reads a gap in two ways, and each reader gets the reading that the rest of the library gives it. The Online Update and every kernel over the price relatives read one at a gap, as if the leg held cash, which is the reading of a Held Gap. So the recursion stays over the full pinned universe and never carries a forced zero. Every statistic over the rows reads them as a batch verb reads a Returns Result, with `NaN` where there was no return and the active mask beside it. Such a statistic is the mean of a forecaster, the prior of a Risk Loss or of a programme set, or the re-solve of a leader. It reduces to its own Coverage Universe. So a window over an unlisted span never meets a constant column, and returns that never happened never dilute the statistic of a relisted asset.

A row with a held projection warns once, with the row's timestamp and each hold, and returns the first [`HeldStep`](@ref) record. A row whose projections all solved returns `nothing`.

# Algorithm

 1. Name the Held Gaps of the returns `r` at the row's index `i` through [`report_row_gaps`](@ref).
 2. Fold the row's Returns Result `rd` into the rows buffer `X` through [`partial_fit!`](@ref), when the tree keeps one. The buffer keeps the raw returns of `rd` verbatim with its active mask, and every other field of `rd` beside it.
 3. Form the price relative `x` through [`price_relative`](@ref), which is one at a gap.
 4. Read the buffer out as `rows`, a [`ReturnsResult`](@ref) of the rows that the head holds, through [`buffer_returns_result`](@ref).
 5. Take the Online Update through [`online_update!`](@ref), inside a projection step over `rows`, the timestamp `ts`, the head's `strict` and the source `x_src` of its step, which [`with_projection_step`](@ref) opens. This gives the new state `st` of the rule, the allocation `w` and the log of holds `held`.
 6. Warn on a hold through [`report_held_steps`](@ref), which gives the row's hold record.

# Arguments

  - `opt`: The head.
  - `st`: The state of the rule: what the rule keeps from one update to the next, such as a Gram matrix or a belief covariance, or `nothing` for a rule that keeps nothing.
  - `w`: The allocation held during the row's period.
  - `X`: The rows buffer, or `nothing`.
  - `row`: The row, as a named tuple of five fields. `r` holds its returns in the head's frame, `amsk` holds its active mask or `nothing`, `ts` holds its timestamp or its index in the fold, `i` holds its index in the fold, and `rd` holds the row as a [`ReturnsResult`](@ref) of one observation.
  - `nx`: The asset names, for the Held Gap message.
  - `set`: The Allocation Set, resolved.

# Returns

  - `(st', w', X', hold)::Tuple`: The state of the rule, the allocation for the next period, the buffer after the row, and the row's hold record or `nothing`.

# Related

  - [`fold_online_selection`](@ref)
  - [`report_row_gaps`](@ref)
  - [`buffer_returns_result`](@ref)
  - [`price_relative`](@ref)
  - [`online_update!`](@ref)
  - [`with_projection_step`](@ref)
  - [`HeldStep`](@ref)
"""
function online_selection_row!(opt::OnlinePortfolioSelection, st, w::AbstractVector,
                               X::Option{<:ReturnsBufferState}, row::NamedTuple,
                               nx::Option{<:VecStr}, set::AbstractAllocationSet)
    (; r, amsk, ts, i) = row
    report_row_gaps(r, w, amsk, nx, opt.strict, i)
    if !isnothing(X)
        X = partial_fit!(X, row.rd; own_returns = true)
    end
    x = price_relative.(r)
    rows = buffer_returns_result(X)
    (st, w), held = with_projection_step(() -> online_update!(opt.alg, st, w, x, rows, set),
                                         rows, ts; strict = opt.strict, x_src = opt.x_src)
    return st, w, X, report_held_steps(held, ts)
end
"""
    buffer_returns_result(X::Nothing)
    buffer_returns_result(X::ReturnsBufferState)

Reads the rows buffer out as the Returns Result that the Online Update gets.

The result is a [`ReturnsResult`](@ref) of the buffer's rows verbatim, under the pinned names, with every other field that the folded rows carried: the factor returns, the benchmark, the Exogenous Series, the timestamps and the implied volatilities of the rows, and the pinned names and `ivpa`. So a leader whose prior states `x_src = ExcessSource()` subtracts the benchmark that the rows carry, as it does in batch. When the buffer records active masks, [`buffer_panel`](@ref) makes them a time-varying Asset Panel on that Returns Result. Otherwise the result carries the static panel that the buffer pinned. The method returns `nothing` when the tree keeps no rows.

Every statistic over the rows reads this Returns Result, and the batch verbs read the same type. So `prior(pe, rd)`, `mean(me, rd.X, rd.pnl)` and `optimise(opt, rd)` reduce to the Coverage Universe of the window exactly as they do on a Returns Result that the ingestion layer built. The method takes no copy, because every array of the Returns Result views the valid region of its buffer. A Returns Result that carries rows carries their names by its own contract, so the refusal of a buffer with no pinned names never meets the head.

# Validation

  - The buffer pinned asset names. An `IsNothingError` is thrown otherwise.

# Related

  - [`online_selection_row!`](@ref)
  - [`buffer_panel`](@ref)
  - [`buffer_column_view`](@ref)
  - [`ReturnsBufferState`](@ref)
  - [`ReturnsResult`](@ref)
"""
function buffer_returns_result(::Nothing)
    return nothing
end
function buffer_returns_result(X::ReturnsBufferState)
    @argcheck(!isnothing(X.nx),
              IsNothingError("the head keeps rows and pins no asset names: a Returns Result that carries rows carries their names, so the rows that the rules read cannot be formed. Hand the head a Returns Result whose `nx` is set."))
    rows = X.X
    vols = implied_volatility_buffer_kwargs(rows)
    return ReturnsResult(; nx = X.nx, X = sample_buffer(rows), nf = X.nf,
                         F = buffer_column_view(X.F), nb = X.nb,
                         B = buffer_column_view(X.B), ne = X.ne,
                         E = buffer_column_view(X.E), ts = X.ts,
                         iv = get(vols, :iv, nothing), ivpa = get(vols, :ivpa, nothing),
                         pnl = something(buffer_panel(rows), Some(X.pnl)))
end
"""
    buffer_column_view(buffer::Nothing)
    buffer_column_view(buffer::SampleBufferState)
    buffer_column_view(buffer::AbstractVector)

Views one column of a [`ReturnsBufferState`](@ref) for the Returns Result of the rows that the head holds. A buffer gives a view of its valid rows, a plain vector gives itself, and `nothing` gives `nothing`. It takes no copy, as [`column_matrix`](@ref) does, because the Online Update reads the rows within the step.

# Related

  - [`buffer_returns_result`](@ref)
  - [`column_matrix`](@ref)
"""
function buffer_column_view(::Nothing)
    return nothing
end
function buffer_column_view(buffer::SampleBufferState)
    return sample_buffer(buffer)
end
function buffer_column_view(buffer::AbstractVector)
    return buffer
end
"""
    buffer_panel(X::SampleBufferState)

Returns the active masks of a rows buffer as the [`AssetPanel`](@ref) of a Returns Result, or `nothing` when the buffer records none, which is the static panel.

The panel uses the active mask as its estimation mask too, because [`step_active_mask`](@ref) admits only that panel.

# Related

  - [`buffer_returns_result`](@ref)
  - [`sample_buffer_kwargs`](@ref)
"""
function buffer_panel(X::SampleBufferState)
    A = X.A
    if isnothing(A)
        return nothing
    end
    M = buffer_rows_view(A, (X.off + 1):(X.off + X.n))
    return AssetPanel(; amsk = M, emsk = M)
end
"""
    row_timestamp(ts::Nothing, t::Integer, n::Integer)
    row_timestamp(ts::AbstractVector, t::Integer, n::Integer)

Returns the timestamp under which a row of a block folds.

It is the timestamp of the row in the returns data. When the returns data holds no timestamps, it is the row's index in the whole fold, `n + 1` after `n` rows.

# Related

  - [`fold_online_selection`](@ref)
  - [`HeldStep`](@ref)
"""
function row_timestamp(::Nothing, ::Integer, n::Integer)
    return n + 1
end
function row_timestamp(ts::AbstractVector, t::Integer, ::Integer)
    return ts[t]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Warns once for a row with a held projection, and names the row's timestamp and every hold.

It returns the first [`HeldStep`](@ref) record, or `nothing` when every projection of the row solved.

# Related

  - [`online_selection_row!`](@ref)
  - [`HeldStep`](@ref)
"""
function report_held_steps(held::AbstractVector{<:HeldStep}, ts)
    if isempty(held)
        return nothing
    end
    @warn("Held Step at $(ts): $(join([h.reason for h in held], "; ")).",
          maxlog = log_limit(:warn), _id = @log_id())
    return first(held)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Names the Held Gaps of one row of returns, and writes nothing.

The function passes over a non-finite cell that the row's active mask marks inactive, because an inactive asset has no return by definition. It also passes over a non-finite cell at which the recursion's weight is zero. Every other non-finite cell is a Held Gap, and [`strict_diagnostic`](@ref) reports them all in one [`held_gap_msg`](@ref). The report is a warning by default and a refusal under `strict`. The function does not fill the row. The buffer keeps the cell as it is, and the Online Update reads it as one through [`price_relative`](@ref).

# Arguments

  - `r`: The row.
  - `w`: The allocation held during the row's period.
  - `amsk`: The row's active mask, or `nothing`.
  - `nx`: The asset names, for the message, or `nothing`.
  - `strict`: Whether a Held Gap is an error.
  - `obs`: The row's index in the fold, which the message names as the observation of each Held Gap.

# Validation

  - Under `strict`, the row holds no Held Gap. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`fold_online_selection`](@ref)
  - [`price_relative`](@ref)
  - [`held_gap_msg`](@ref)
"""
function report_row_gaps(r::AbstractVector, w::AbstractVector,
                         amsk::Option{<:AbstractVector{<:Bool}}, nx::Option{<:VecStr},
                         strict::Bool, obs::Integer)::Nothing
    held = Tuple{Int, Int}[]
    for i in eachindex(r)
        if isfinite(r[i]) || (!isnothing(amsk) && !amsk[i]) || iszero(w[i])
            continue
        end
        push!(held, (obs, i))
    end
    if !isempty(held)
        strict_diagnostic(held_gap_msg(held, nx), strict; id = @log_id())
    end
    return nothing
end
"""
    partial_fit!(opt::OnlinePortfolioSelection, rd::ReturnsResult)

Folds the rows of a Returns Result into the head's recursion as the Block Step, and reads nothing out.

Each row of `rd` is one Online Update, in order, and the state never learns the cadence of the loop. A fold of `k` rows is `k` updates and builds no Recursion Result. After a fold of rows `1:t`, `optimise(opt)` equals `optimise(opt, rd[1:t])` exactly, because both take the same `t` single-row updates. The step also folds `rd` into the fallback through [`step_fallback`](@ref), because the head holds no returns window that the fallback could read when `optimise(opt)` runs with no data.

# Arguments

  - `opt`: The head.
  - `rd`: The returns data, which holds one row or a block of rows, `observations × assets`.

# Validation

  - Everything [`fold_online_selection`](@ref) refuses.
  - Everything [`step_fallback`](@ref) refuses.

# Returns

  - `opt`: The head, with its `cache` rebound to the state after the last row and its fallback stepped.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`fold_online_selection`](@ref)
  - [`step_fallback`](@ref)
  - [`optimise`](@ref)
"""
function partial_fit!(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any, <:Any,
                                                    <:Option{<:OnlinePortfolioSelectionState}},
                      rd::ReturnsResult)
    return rebuild_estimator(opt,
                             (; cache = fold_online_selection(opt, opt.cache, rd),
                              fb = step_fallback(opt.fb, rd)))
end
"""
    online_step_fold(opt::OnlinePortfolioSelection, ctx::TimeDependentContext, rd::ReturnsResult)

Folds the fold's rows into the head inside the fold's own Allocation Set.

The head is the one family whose online step is the optimisation. The Constrained Update projects the raw step of every row onto `set`, so the step reads a schedule on `set`, and not only `optimise(opt)` with no data. So the method resolves the entry here, in the same fold and before `optimise(opt)` with no data resolves the other schedules. The head that the loop threads on keeps the schedule, so fold `i + 1` resolves from the schedule and not from entry `i`. The step writes no field but the state and the fallback, so the method returns the head with its `cache` rebound and its fallback stepped through [`step_fallback`](@ref).

# Arguments

  - `opt`: The head the loop threads, with its schedules unresolved.
  - `ctx`: The fold's context.
  - `rd`: The returns data of the rows that the fold has gained.

# Returns

  - `opt`: The head, with its `cache` rebound and its schedules unresolved.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`online_step_fold`](@ref)
  - [`fold_online_selection`](@ref)
  - [`thread_online_folds!`](@ref)
"""
function online_step_fold(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                        <:Any, <:Any,
                                                        <:Option{<:OnlinePortfolioSelectionState}},
                          ctx::TimeDependentContext, rd::ReturnsResult)
    set = opt.set
    set = isa(set, TimeDependent) ? time_dependent_value(set, ctx) : set
    return rebuild_estimator(opt,
                             (; cache = fold_online_selection(opt, opt.cache, rd, set),
                              fb = step_fallback(opt.fb, rd)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the Recursion Result, the allocation of the state as a [`NaiveOptimisationResult`](@ref) over the assets active at the last folded row.

This function reads the state and writes nothing, so every call on one state returns the same Result. Its retcode is an [`OptimisationSuccess`](@ref). Its `res` is the [`HeldStep`](@ref) record of the last folded row when that row was held, and `nothing` otherwise. A reduced allocation with no mass is the one failure from which a rule cannot recover. It happens when every held asset is inactive at the last row. This function then returns an [`OptimisationFailure`](@ref) with `NaN` weights, so that a fallback chain continues.

# Mathematical definition

```math
\\begin{align}
\\bar{w}_i &= \\frac{w_{T+1,i}}{\\sum_{j \\in \\mathcal{I}} w_{T+1,j}}\\,, \\quad i \\in \\mathcal{I}\\,.
\\end{align}
```

Where:

  - ``\\bar{w}_i``: Weight of asset ``i`` in the Result.
  - ``w_{T+1,i}``: Weight of asset ``i`` in the allocation of the state, the Next-Period Allocation after ``T`` rows.
  - ``\\mathcal{I}``: Investable Mask of the Result, the assets active at the last folded row, or every asset under a static panel.
  - $(math_dict[:T])

The Result expands ``\\bar{\\boldsymbol{w}}`` back over the pinned universe with a zero at every asset outside ``\\mathcal{I}``. The weights are defined when the denominator is positive and every ``w_{T+1,i}`` is finite.

# Algorithm

 1. Read the active mask `imsk` of the last folded row, and form `idx`, the indices of the active assets, or every index when `imsk` is `nothing`.
 2. View the head's set at `idx`, giving `set`, and slice the allocation to `w`.
 3. Resolve the head's fee against the full set, and view it at the Investable Mask through [`investable_fees_view`](@ref), giving `fees`.
 4. Sum `w` to `s`. When no asset is active, `s` is not positive, or an entry of `w` is not finite, return an [`OptimisationFailure`](@ref) with `NaN` weights and no bounds.
 5. Resolve the weight bounds of `set` over the active assets, giving `wb`.
 6. Return `w ./ s` with `wb`, `fees` and an [`OptimisationSuccess`](@ref) whose `res` is the hold record of the state.

# Arguments

  - `opt`: The head.
  - `state`: The state to read out.

# Returns

  - `res::NaiveOptimisationResult`: The Next-Period Allocation, `pr = nothing`.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`OnlinePortfolioSelectionState`](@ref)
  - [`NaiveOptimisationResult`](@ref)
"""
function online_selection_result(opt::OnlinePortfolioSelection,
                                 state::OnlinePortfolioSelectionState)
    N = length(state.w)
    imsk = state.amsk
    idx = isnothing(imsk) ? Colon() : findall(imsk)
    set = isnothing(imsk) ? opt.set : port_opt_view(opt.set, idx)
    w = state.w[idx]
    Nr = length(w)
    fees = investable_fees_view(fees_constraints(opt.fees, opt.set.sets;
                                                 strict = opt.strict, datatype = eltype(w)),
                                imsk, N)
    s = sum(w)
    if iszero(Nr) || !(s > zero(s)) || !all(isfinite, w)
        # No asset is active at the last row, or the recursion's mass sits on none of
        # them: there is no allocation to read out, and no universe to bound.
        return NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                       retcode = OptimisationFailure(;
                                                                     res = "the recursion's allocation has no mass on the assets active at the last folded row, so no Next-Period Allocation can be read out."),
                                       w = fill(convert(eltype(w), NaN), Nr), imsk = imsk,
                                       fb = nothing, fees = fees)
    end
    wb = weight_bounds_constraints(set.wb, set.sets; N = Nr, strict = opt.strict,
                                   datatype = eltype(w))
    retcode, w = OptimisationSuccess(; res = state.hold), w ./ s
    return NaiveOptimisationResult(; pr = nothing, wb = wb, retcode = retcode, w = w,
                                   imsk = imsk, fb = nothing, fees = fees)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs the Causal Pass.

[`optimise`](@ref) calls this internal dispatch. It reads and writes no state that the head carries.

# Algorithm

 1. Reset every schedule of the head to its fold-less value through [`reset_time_dependent_estimator`](@ref).
 2. Fold every row of `rd` from the Start Allocation, with no carried state, through [`fold_online_selection`](@ref), giving `state`.
 3. Read the Next-Period Allocation out of `state` through [`online_selection_result`](@ref).

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`fold_online_selection`](@ref)
  - [`online_selection_result`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(opt::OnlinePortfolioSelection, rd::ReturnsResult; kwargs...)
    opt = reset_time_dependent_estimator(opt)
    state = fold_online_selection(opt, nothing, rd)
    return online_selection_result(opt, state)
end
"""
    optimise(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; kwargs...) -> NaiveOptimisationResult

Runs the Causal Pass over `rd`, and returns the Next-Period Allocation.

# Arguments

  - `opt`: The head.
  - $(arg_dict[:rd]) Every row is one Online Update, in order.
  - `kwargs`: Additional keyword arguments, ignored.

# Validation

  - No field in the tree of `opt` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise, through [`assert_batch_entry`](@ref).

# Returns

  - `res::NaiveOptimisationResult`: The Next-Period Allocation, with `pr = nothing`.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`partial_fit!`](@ref)
  - [`online_selection_result`](@ref)
"""
function optimise(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; kwargs...)::NaiveOptimisationResult
    assert_batch_entry(opt, "`optimise`")
    return _optimise(opt, rd; kwargs...)
end
"""
    optimise(opt::OnlinePortfolioSelection; kwargs...) -> OptimisationResult

Returns the Recursion Result, the allocation of the state as a Result, and runs no batch path.

The method refuses a head that has taken no step with an `ArgumentError`. The Recursion Result fails in one way, when the allocation has no mass on the assets active at the last row. The fallback chain then runs as the batch verb runs it. The fallback reads out through its own `optimise(fb)`, from the rows that [`step_fallback`](@ref) folded into it at every step of the head. So a fallback that needs rows answers from the same rows as in batch. The chain of the Result starts with the failed pair of the head, and the chain that the fallback walked follows it.

# Algorithm

 1. Refuse a head whose `cache` is `nothing`.
 2. Reset every schedule of the head to its fold-less value through [`reset_time_dependent_estimator`](@ref).
 3. Read the state out through [`online_selection_result`](@ref), giving `res`.
 4. Walk the fallback chain after `res` with [`read_out_fallback`](@ref). It returns `res` when `res` succeeded or when the head has no fallback.

# Arguments

  - `opt`: The stepped head.
  - `kwargs`: The keyword arguments of `optimise` with no data on the fallback, passed on unchanged.

# Validation

  - `opt.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `res::OptimisationResult`: The Next-Period Allocation, or the Result of the fallback chain when the Recursion Result fails.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`online_selection_result`](@ref)
  - [`partial_fit!`](@ref)
"""
function optimise(opt::OnlinePortfolioSelection; kwargs...)
    @argcheck(!isnothing(opt.cache),
              ArgumentError("`optimise(opt)` with no returns reads the state the online step wrote, and this `OnlinePortfolioSelection` has taken no step: its `cache` is `nothing`. Fold observations with `partial_fit!(opt, rd)` first, or pass the returns to `optimise(opt, rd)`."))
    opt = reset_time_dependent_estimator(opt)
    return read_out_fallback(opt, online_selection_result(opt, opt.cache); kwargs...)
end
"""
    batch_from_state(opt::OnlinePortfolioSelection)

Refuses the head with an `ArgumentError`, because the head has no batch estimator to return.

The head gives a Recursion Result. `optimise(opt)` rebuilds no returns data, runs no batch path and reads the state directly.

# Related

  - [`batch_from_state`](@ref)
  - [`optimise`](@ref)
"""
function batch_from_state(::OnlinePortfolioSelection)
    return throw(ArgumentError("an `OnlinePortfolioSelection` head has a Recursion Result, not a reconstitution: its state holds the next allocation and no returns data to rebuild, so there is no batch estimator to hand back. Read it out with `optimise(opt)`."))
end
"""
    held_timestamps(opt::OnlinePortfolioSelection)

Returns every timestamp that the head's state holds, or `nothing` before the first step.

# Related

  - [`held_timestamps`](@ref)
  - [`Resume`](@ref)
"""
function held_timestamps(opt::OnlinePortfolioSelection)
    return isnothing(opt.cache) ? nothing : opt.cache.ts
end
"""
    update_online_estimator(opt::Union{OnlinePortfolioSelection, PreviousWeights})

Resolves the [`Online`](@ref) wrappers in the fallback of the head, at the warm-up.

The two heads step their fallback through [`step_fallback`](@ref), so a wrapper in the prior of a fallback estimator declares the window that the fallback folds. The method resolves the fallback as the warm-up resolves an optimiser. A [`PreviousWeights`](@ref) that is the fallback of the head resolves its own fallback in turn. Any other fallback holds no wrapper to resolve, so the head passes unchanged. The recursion of the head holds no wrapper.

# Related

  - [`update_online_estimator`](@ref)
  - [`step_fallback`](@ref)
  - [`online_unreached_path`](@ref)
"""
function update_online_estimator(opt::Union{OnlinePortfolioSelection, PreviousWeights})
    return opt
end
function update_online_estimator(opt::Union{OnlinePortfolioSelection{<:Any, <:Any, <:Any,
                                                                     <:Any,
                                                                     <:OptimisationEstimator},
                                            PreviousWeights{<:Any, <:Any,
                                                            <:OptimisationEstimator}})
    return rebuild_estimator(opt, (; fb = update_online_estimator(opt.fb)))
end
"""
    online_unreached_path(opt::Union{OnlinePortfolioSelection, PreviousWeights})

Names the first [`Online`](@ref) wrapper in the head that the warm-up leaves unresolved, or answers `nothing`.

The method runs the warm-up of [`update_online_estimator`](@ref) on the head, and names the wrapper that is left, such as one in an inner optimiser of a meta-optimiser that is the fallback. It serves an [`OnlinePortfolioSelection`](@ref) head and a [`PreviousWeights`](@ref), the two heads that step their fallback.

# Related

  - [`online_unreached_path`](@ref)
  - [`assert_online_entry`](@ref)
"""
function online_unreached_path(opt::Union{OnlinePortfolioSelection, PreviousWeights})
    return online_wrapper_path(update_online_estimator(opt))
end
"""
    Online(::OnlinePortfolioSelection, args...; kwargs...)

Refuses to construct `Online(head)`, and names the three routes to the two answers that the wrapper stands for.

[`Online`](@ref) declares a refit from a buffer, and for a head each of its two settings is a batch walk-forward that the library already runs. Uncapped, the refit is the expanding walk-forward. Its allocation equals the allocation of the online arm exactly, because both take the same single-row updates from `w0`. But it costs a pass over every row folded so far, where the online arm costs one row. Capped, the refit is the rolling walk-forward, whose allocation is the recursion restarted from `w0` inside each window. So the wrapper adds no answer, and the refusal costs the caller nothing.

# Related

  - [`Online`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
  - [`IndexWalkForward`](@ref)
"""
function Online(::OnlinePortfolioSelection, args...; kwargs...)
    return throw(ArgumentError("`Online` does not wrap an `OnlinePortfolioSelection` head: the wrapper declares a refit from a buffer, and for a head both of its settings are a batch walk-forward that runs today. Uncapped, the refit is `IndexWalkForward(train_size, test_size; expand_train = true)`, whose allocation is the online arm's to the last bit and which pays a pass over every row folded so far. Capped at `max_history = w`, it is `IndexWalkForward(w, test_size)`, the recursion restarted from `w0` inside each window. Step the head with `OnlineIndexWalkForward`, or refit it with whichever of those two you meant."))
end
"""
    fees_carry_turnover(fees::Nothing)
    fees_carry_turnover(fees::FeesE_Fees)
    fees_carry_turnover(v::AbstractVector)
    fees_carry_turnover(td::TimeDependent)
    fees_carry_turnover(::Any)

Returns whether a fee, or any fee that a per-fold schedule visibly holds, carries a turnover term.

A static fee answers for its own `tn`. A [`TimeDependent`](@ref) schedule answers for every entry of a vector value, including the entries of a nested vector, and for its explicit `default`. The output of a callable does not exist before its fold, so a callable answers `false`. Every other value answers `false` too, as it does for [`needs_previous_weights`](@ref).

# Related

  - [`assert_online_fee_source`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function fees_carry_turnover(::Nothing)::Bool
    return false
end
function fees_carry_turnover(fees::FeesE_Fees)::Bool
    return !isnothing(fees.tn)
end
function fees_carry_turnover(v::AbstractVector)::Bool
    return any(fees_carry_turnover, v)
end
function fees_carry_turnover(td::TimeDependent)::Bool
    return fees_carry_turnover(td.val) || fees_carry_turnover(td.default)
end
function fees_carry_turnover(::Any)::Bool
    return false
end
"""
    assert_online_fee_source(opt::OnlinePortfolioSelection, pws)

Refuse, at the entry of the fold loop's online arm, an [`OnlinePortfolioSelection`](@ref) head whose `fees` carry a turnover term while the walk-forward threads no Previous-Weights Source.

The recursion reads its own allocation, so the fund's held book is the one base against which a turnover fee on this family can measure a trade. Without a Previous-Weights Source, the fee prices the distance between two consecutive targets. That distance counts buy-and-hold as trading and constant rebalancing as free. The head and the walk-forward first meet at this check, so the check runs as soon as the pair exists. A head with no `tn` fee passes, and so does every other estimator. [`fees_carry_turnover`](@ref) reads a per-fold schedule on `fees`. It checks every entry that the schedule visibly holds, and it does not check the output of a callable, which does not exist before its fold.

# Arguments

  - `opt`: The estimator handed to the loop.
  - `pws`: The scheme's Previous-Weights Source as [`resolve_previous_weights_source`](@ref) gives it, where `nothing` threads the target weights. A scheme that sets `wd` and no `pws` passes a [`DriftedWeights`](@ref), and an explicit [`TargetWeights`](@ref) passes `nothing`.

# Validation

  - `pws` is not `nothing` when `opt.fees` carries a `tn` term, in any entry that a schedule visibly holds. An `ArgumentError` is thrown otherwise.

# Related

  - [`online_folds`](@ref)
  - [`assert_online_entry`](@ref)
  - [`fees_carry_turnover`](@ref)
  - [`DriftedWeights`](@ref)
  - [`resolve_previous_weights_source`](@ref)
"""
function assert_online_fee_source(opt::OnlinePortfolioSelection, pws)::Nothing
    if isnothing(pws) && fees_carry_turnover(opt.fees)
        throw(ArgumentError("an `OnlinePortfolioSelection` head whose `fees` carry a turnover term needs a Previous-Weights Source on the walk-forward: the recursion reads its own allocation, so without one the fee would price the distance between two targets rather than the trade the fund makes. Set a Weight Drift `wd` on the scheme, whose unset `pws` then threads the drifted weights, or set `pws = DriftedWeights()`, or drop `tn` from the head's fees."))
    end
    return nothing
end
export OnlinePortfolioSelection
