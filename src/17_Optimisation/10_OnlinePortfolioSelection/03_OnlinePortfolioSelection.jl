"""
$(DocStringExtensions.TYPEDEF)

Updates its allocation from each realised price relative through the Online Selection Rule on `alg`.

The head is a naive optimiser. It fits no moment and solves no programme of its own. Every rule of the family is one recursion, which rebalances the allocation to its target at the start of every period, and the head does all the work that the rules share. `# Mathematical definition` states the recursion and the wealth it earns.

**The batch verb is the Causal Pass.** `optimise(opt, rd)` starts from the Start Allocation `w0`, applies the Online Update to every row of `rd` in order, and returns the **Next-Period Allocation**. After rows `1:T` that is the portfolio for period `T + 1`, so the pass takes one more update than a backtest driver that stops at the last row it holds. The pass rebalances at every row and pays no fee. `predict(res, rd)` on its Result charges the head's `fees`, with no drift.

**The online verbs run the same recursion.** `partial_fit!(opt, rd)` folds each row into the state as the Block Step. A fold of `k` rows is `k` single-row updates, so a walk-forward with `test_size = k` holds the Next-Period Allocation of the end of the block for `k` periods. `optimise(opt)` with no data is the **Recursion Read-out**. It wraps the allocation of the state in a Result and runs no batch path. So `optimise(opt)` after a fold of rows `1:t` equals `optimise(opt, rd[1:t])` exactly, at every `test_size`. The held path of a walk-forward differs from the batch pass in two ways. The target set at the start of a block drifts over the block, and the Block Step sets the cadence of the trades.

**The Start Allocation** is `w0`, `nothing` by default. When it is absent, the recursion starts at `1/N` over the whole pinned universe, and the unlisted names are included. If `k` of `N` assets are unlisted at the first row, the recursion puts `k/N` of its weight on legs that see `x = 1`, as cash does, until the rule moves that weight. The read-out renormalises over the listed legs, so the fund holds the listed legs alone. But the fund is not always the recursion that starts over the listed names alone, because the parked legs enter the gross return `⟨w, x⟩` that a rule reads. Buy-and-hold scales each leg by its own price relative, so its fund is the same. The exponentiated-gradient step divides by the gross return, so its fund is different. A caller who wants the recursion to be the fund from row one gives `w0` over the listed assets, with zeros elsewhere. The rule's step then decides what a zero does. The exponentiated-gradient step keeps a zero at zero, so that asset never gets weight. A Euclidean step that moves no weight onto the asset keeps the zero exact too, because [`project_simplex`](@ref) only renormalises a raw step that is on the simplex up to rounding. Buy-and-hold is such a step. A Euclidean step that moves weight onto the asset, as the passive-aggressive step does, gives it weight, and a gap at that asset is then a Held Gap.

A given `w0` is over the pinned names, and the head pins and views it with them. The first step projects it once onto the Allocation Set in the rule's geometry, as it projects the uniform start, so a start outside the set becomes feasible and is never refused. A set that reads the head's rows has no constraints before the first row, so it holds the start as given until the first update. The start reaches the first Online Update as `w`. A rule that steps from `w` continues from it. A rule whose next allocation is not a step from `w` replaces it at that update. The constant rebalanced portfolio returns its own `w`, an expert mixture returns the mix of its experts, and the Newton step solves its Gram. Such a rule can still carry `w0` forward. The gradient of the Newton step reads `w`, and an expert that steps from `w` starts at `w0`. Under every rule, the fund holds `w0` for exactly one period.

**The Online Update reads the recursion's own allocation, and no flag changes that.** The previous weights of the loop reach the head through [`factory`](@ref) alone. They go into `fees` and `fb`, and never into the recursion. The fund's held book is the one base against which a turnover fee on this family measures a trade. So the online arm of the fold loop refuses a head whose `fees` carry a `tn` term when the Previous-Weights Source of the walk-forward is `nothing`. It refuses by name and before any fold, and `pws = DriftedWeights()` is the whole configuration. A head with no `tn` fee is not checked.

**A non-finite return is kept, and read in two ways.** The rule's step reads `x = 1` there, as if the leg held cash. At an asset that the panel marks inactive, or at one to which the recursion gives no weight, this is silent. At an active asset with a non-zero weight it is a Held Gap, which warns by default and refuses under `strict`. The rows buffer keeps the cell as `NaN`, with the row's active mask beside it. Every statistic over the rows reads them as the batch verb reads a carrier, and reduces to its own Coverage Universe. Such a statistic is the mean of a forecaster, the prior of a Risk Loss or of a programme set, or the re-solve of a leader. A plain estimator drops an asset with a gap anywhere in its window. A mask-aware estimator answers the asset from the rows it has. A kernel over price relatives reads the gap as one.

The head never forces a zero into the recursion's allocation, and a relisted asset comes back at the recursion's own weight. A programme set that fits on the rows writes a zero at a leg that its prior cannot price, as a batch head does. It admits the leg again when the prior can price it. Under a time-varying panel, the Investable Mask of the read-out is the active mask of the last folded row. The read-out slices the full allocation to that mask and renormalises it, and the Result expands it back with a zero at every non-investable asset.

**The Allocation Set on `set` is the one the Constrained Update projects onto.** [`project`](@ref) projects the raw step of every rule onto it, in the rule's own Projection Geometry. The default [`BoundedAllocationSet`](@ref) is the simplex, and every projection onto it is closed form, so the default configuration solves nothing. A [`ProgrammeAllocationSet`](@ref) admits the full constraint vocabulary, and its projection is a programme. When a programme fails, the step is a **Held Step**. The projection returns the Price-Adjusted Allocation it was given, so the fund trades nothing that period, and the rule's carrier still absorbs the row. The head warns once, with the row's timestamp. The retcode of the Recursion Read-out is an [`OptimisationSuccess`](@ref) that carries the [`HeldStep`](@ref) record of the last folded row, so a fallback chain never runs on a hold. The constructor refuses a negative lower bound under an entropic, Tsallis or log-barrier rule.

`merge_states` on the state and `Online(head)` are refused by name. An update depends on the order of the rows, and the family never refits from a buffer.

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

The Next-Period Allocation after ``T`` rows is ``\\boldsymbol{w}_{T+1}``. Both projections are one call of [`project`](@ref), so a [`ProgrammeAllocationSet`](@ref) adds its Objective Penalty to both objectives.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OnlinePortfolioSelection(;
        alg::AbstractOnlinePortfolioSelectionAlgorithm = ExponentiatedGradient(),
        set::TD{<:AbstractAllocationSet} = BoundedAllocationSet(),
        w0::Option{<:AbstractVector} = nothing,
        fees::TD_Option{<:FeesE_Fees} = nothing,
        fb::TDO_Option{<:OptE_Opt} = nothing,
        strict::Bool = false,
        cache::Option{<:OnlinePortfolioSelectionState} = nothing
    ) -> OnlinePortfolioSelection

Keywords correspond to the struct's fields. `set`, `fees` and `fb` can hold a [`TimeDependent`](@ref) per-fold schedule. `alg`, `w0` and `strict` are static.

## Validation

  - `w0`: non-empty and finite, when given. It need not lie in the set, because the first step projects it onto the set.
  - If `fees` is a [`FeesEstimator`](@ref): `!isnothing(set.sets)`. A schedule on `set` is held to it per entry.
  - Everything [`assert_geometry_admits_set`](@ref) refuses: a negative lower bound under an entropic, Tsallis or log-barrier rule. A schedule on `set` is held to it per entry.
  - Everything [`assert_rule_admits_set`](@ref) refuses: a solver-free leader under a programme set. A schedule on `set` is held to it per entry.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following fields are propagated:

  - `fees`: Recursively updated via [`factory`](@ref), which threads the previous weights into a turnover fee.
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the fields with a per-asset axis are subset to the selected indices: `alg` and `set` recursively, `w0` sliced and renormalised, `fees` viewed at the indices, and `cache` through the state's own view.

# Examples

```jldoctest
julia> OnlinePortfolioSelection()
OnlinePortfolioSelection
     alg ┼ MirrorDescent
         │     eta ┼ Float64: 0.05
         │    proj ┼ EntropicProjection()
         │   alpha ┼ Int64: 0
         │     obj ┼ LogWealth()
         │    grad ┴ PlainGradient()
     set ┼ BoundedAllocationSet
         │     wb ┼ WeightBounds
         │        │   lb ┼ Float64: 0.0
         │        │   ub ┴ Float64: 1.0
         │   sets ┴ nothing
      w0 ┼ nothing
    fees ┼ nothing
      fb ┼ nothing
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

  - $(ref_dict[:lihoi2014])
"""
@concrete struct OnlinePortfolioSelection <: NaiveOptimisationEstimator
    """
    The Online Selection Rule: the update, its parameters and its Projection Geometry.
    """
    alg
    """
    The Allocation Set every allocation of the recursion lies in, or a [`TimeDependent`](@ref) schedule of one set per fold. Entry `i` is the complete set of fold `i`. The fold loop puts it in place before it folds the rows of fold `i`, so those rows step inside entry `i`, and the read-out reports it.
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
    $(field_dict[:strict_opt])
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
                                      fb::TDO_Option{<:OptE_Opt}, strict::Bool,
                                      cache::Option{<:OnlinePortfolioSelectionState})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :OnlinePortfolioSelection)
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
                                           (; alg, set, w0, fees, fb, strict),
                                           online_portfolio_selection_td_defaults())
        return new{typeof(alg), typeof(set), typeof(w0), typeof(fees), typeof(fb),
                   typeof(strict), typeof(cache)}(alg, set, w0, fees, fb, strict, cache)
    end
end
function OnlinePortfolioSelection(;
                                  alg::AbstractOnlinePortfolioSelectionAlgorithm = ExponentiatedGradient(),
                                  set::TD{<:AbstractAllocationSet} = BoundedAllocationSet(),
                                  w0::Option{<:AbstractVector} = nothing,
                                  fees::TD_Option{<:FeesE_Fees} = nothing,
                                  fb::TDO_Option{<:OptE_Opt} = nothing,
                                  strict::Bool = false,
                                  cache::Option{<:OnlinePortfolioSelectionState} = nothing)::OnlinePortfolioSelection
    return OnlinePortfolioSelection(alg, set, w0, fees, fb, strict, cache)
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

The recursion never reads them, so the method carries `alg`, `set`, `w0` and `cache` unchanged.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`factory`](@ref)
"""
function factory(opt::OnlinePortfolioSelection, w::VecNum)
    return OnlinePortfolioSelection(; alg = opt.alg, set = opt.set, w0 = opt.w0,
                                    fees = factory(opt.fees, w), fb = factory(opt.fb, w),
                                    strict = opt.strict, cache = opt.cache)
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
                                    fb = view_child(opt.fb, i, args...),
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
 3. Seed the rule's carrier on `w0` and `set` through [`rule_state_seed`](@ref), giving `st`. Steps 2 and 3 run inside one [`with_projection_step`](@ref), which collects the holds in `held`.
 4. Warn on a hold through [`report_held_steps`](@ref).
 5. Size the rows buffer `X` from [`rows_needed`](@ref). A count of `nothing` gives an uncapped buffer, a count of zero gives no buffer, and any other count caps the buffer at that count.
 6. Keep the carrier's panel as `pnl` when it is static, and `nothing` otherwise.
 7. Return the state with no row folded, the allocation `w0`, the carrier `st`, the buffer `X`, the names and `pnl`.

# Arguments

  - `opt`: The head.
  - `rd`: The carrier of the first block.
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
    (w0, st), held = with_projection_step(nothing, nothing) do
        w = project_start(projection_geometry(opt.alg), set, start)
        return w, rule_state_seed(opt.alg, w, set)
    end
    report_held_steps(held, "the start")
    need = rows_needed(opt)
    X = if isnothing(need)
        SampleBufferState()
    elseif iszero(need)
        nothing
    else
        SampleBufferState(; max_history = need)
    end
    static = isnothing(rd.pnl) || panel_is_static(rd.pnl)
    return OnlinePortfolioSelectionState(; n = 0, w = w0, st = st, X = X, nx = rd.nx,
                                         pnl = static ? rd.pnl : nothing, amsk = nothing,
                                         ts = nothing)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every row of a carrier into the head's state, in order.

This is the Block Step, and a fold of the whole carrier from no state is the Causal Pass.

# Algorithm

 1. Read the active mask of the carrier through [`step_active_mask`](@ref), giving `amsk`.
 2. Resolve the Allocation Set over the pinned universe through [`resolve_allocation_set`](@ref), giving `set`.
 3. Seed the state on the first block, or pin and check the carried state, through [`online_selection_pin`](@ref), giving `state`.
 4. Fold each row in order through [`online_selection_row!`](@ref), under the timestamp that [`row_timestamp`](@ref) gives it. Each row updates the carrier `st`, the allocation `w`, the buffer `X` and the hold record `hold`, and adds one to the row count `n`.
 5. Return the state with the new `n`, `w`, `st`, `X` and `hold`, the active mask of the last row, and the timestamps folded so far.

# Arguments

  - `opt`: The head.
  - `cache`: The state, or `nothing` before the first row.
  - `rd`: The carrier of the block, `observations × assets`.
  - `set`: The block's Allocation Set, before resolution over the pinned universe. It defaults to the head's own, and the fold loop's online arm passes the fold's entry of a schedule (see [`online_step_fold`](@ref)).

# Validation

  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
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
    amsk = step_active_mask(rd)
    set = resolve_allocation_set(set, size(rd.X, 2), opt.strict, eltype(rd.X))
    state = online_selection_pin(opt, cache, rd, set)
    w, st, X, n = state.w, state.st, state.X, state.n
    last_amsk = state.amsk
    hold = state.hold
    for t in axes(rd.X, 1)
        row = (; r = vec(rd.X[t, :]), amsk = isnothing(amsk) ? nothing : vec(amsk[t, :]),
               ts = row_timestamp(rd.ts, t, n), i = n + 1)
        st, w, X, hold = online_selection_row!(opt, st, w, X, row, rd.nx, set)
        n += 1
        last_amsk = row.amsk
    end
    last_amsk = isnothing(last_amsk) ? nothing : BitVector(last_amsk)
    return OnlinePortfolioSelectionState(; n = n, w = w, st = st, X = X, nx = state.nx,
                                         pnl = state.pnl, amsk = last_amsk,
                                         ts = fold_column(state.ts, rd.ts, nothing),
                                         hold = hold)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the state that a block folds into.

On the first block the method seeds the state through [`online_selection_seed`](@ref). On every later block it returns the state that the head carries, after the pin-and-check that every host runs. The check compares the names and the static panel with those of the first step, and the width with that of the state. The timestamps must be present at every block or at none.

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
function online_selection_pin(::OnlinePortfolioSelection,
                              cache::OnlinePortfolioSelectionState, rd::ReturnsResult,
                              ::AbstractAllocationSet)
    N = size(rd.X, 2)
    @argcheck(length(cache.w) == N,
              DimensionMismatch("the state holds $(length(cache.w)) assets and this block carries $N"))
    assert_pinned_context(cache.nx, rd.nx, :nx)
    static = isnothing(rd.pnl) || panel_is_static(rd.pnl)
    assert_pinned_context(cache.pnl, static ? rd.pnl : nothing, :pnl)
    if !iszero(cache.n)
        assert_column_presence(cache.ts, rd.ts, :ts)
    end
    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Takes one row of the recursion, inside a [`ProjectionStep`](@ref) that gives every projection of the row the rows it reads and a log for its holds.

The row reads a gap in two ways, and each reader gets the reading that the rest of the library gives it. The Online Update and every kernel over the price relatives read one at a gap, as if the leg held cash, which is the reading of a Held Gap. So the recursion stays over the full pinned universe and never carries a forced zero. Every statistic over the rows reads the carrier as a batch verb reads one, with `NaN` where there was no return and the active mask beside it. Such a statistic is the mean of a forecaster, the prior of a Risk Loss or of a programme set, or the re-solve of a leader. It reduces to its own Coverage Universe. So a window over an unlisted span never meets a constant column, and returns that never happened never dilute the statistic of a relisted asset.

A row with a held projection warns once, with the row's timestamp and each hold, and returns the first [`HeldStep`](@ref) record. A row whose projections all solved returns `nothing`.

# Algorithm

 1. Name the Held Gaps of the returns `r` at the row's index `i` through [`report_row_gaps`](@ref).
 2. Push `r` verbatim, with its active mask `amsk`, into the rows buffer `X`, when the tree keeps one.
 3. Form the price relative `x` through [`price_relative`](@ref), which is one at a gap.
 4. Read the buffer out as the carrier `rows` through [`rows_carrier`](@ref).
 5. Take the Online Update through [`online_update!`](@ref), inside a projection step over `rows` and the timestamp `ts` that [`with_projection_step`](@ref) opens. This gives the carrier `st`, the allocation `w` and the log of holds `held`.
 6. Warn on a hold through [`report_held_steps`](@ref), which gives the row's hold record.

# Arguments

  - `opt`: The head.
  - `st`: The rule's carrier.
  - `w`: The allocation held during the row's period.
  - `X`: The rows buffer, or `nothing`.
  - `row`: The row, as a named tuple of four fields. `r` holds its returns verbatim, `amsk` holds its active mask or `nothing`, `ts` holds its timestamp or its index in the fold, and `i` holds its index in the fold.
  - `nx`: The asset names, for the Held Gap message.
  - `set`: The Allocation Set, resolved.

# Returns

  - `(st', w', X', hold)::Tuple`: The carrier, the allocation for the next period, the buffer after the row, and the row's hold record or `nothing`.

# Related

  - [`fold_online_selection`](@ref)
  - [`report_row_gaps`](@ref)
  - [`rows_carrier`](@ref)
  - [`price_relative`](@ref)
  - [`online_update!`](@ref)
  - [`with_projection_step`](@ref)
  - [`HeldStep`](@ref)
"""
function online_selection_row!(opt::OnlinePortfolioSelection, st, w::AbstractVector,
                               X::Option{<:SampleBufferState}, row::NamedTuple,
                               nx::Option{<:VecStr}, set::AbstractAllocationSet)
    (; r, amsk, ts, i) = row
    report_row_gaps(r, w, amsk, nx, opt.strict, i)
    if !isnothing(X)
        X = partial_fit!(X, r; active_mask = amsk)
    end
    x = price_relative.(r)
    rows = rows_carrier(X, nx)
    (st, w), held = with_projection_step(() -> online_update!(opt.alg, st, w, x, rows, set),
                                         rows, ts; strict = opt.strict)
    return st, w, X, report_held_steps(held, ts)
end
"""
    rows_carrier(X::Nothing, nx)
    rows_carrier(X::SampleBufferState, nx::VecStr)
    rows_carrier(X::SampleBufferState, nx::Nothing)

Reads the rows buffer out as the carrier that the Online Update gets.

The carrier is a [`ReturnsResult`](@ref) of the buffer's rows verbatim, under the pinned names. When the buffer records active masks, [`buffer_panel`](@ref) makes them a time-varying Asset Panel on the carrier. The method returns `nothing` when the tree keeps no rows.

Every statistic over the rows reads this carrier, and it is the carrier that the batch verbs read. So `prior(pe, rd)`, `mean(me, rd.X, rd.pnl)` and `optimise(opt, rd)` reduce to the Coverage Universe of the window exactly as they do on a carrier that the ingestion layer built. The method takes no copy, because the carrier views the valid region of the buffer. A buffer with no pinned names is refused by name. A Returns Result that carries rows carries their names by its own contract, so the head never meets that pair.

# Validation

  - `nx` is not `nothing` when the head keeps rows. An `IsNothingError` is thrown otherwise.

# Related

  - [`online_selection_row!`](@ref)
  - [`buffer_panel`](@ref)
  - [`SampleBufferState`](@ref)
  - [`ReturnsResult`](@ref)
"""
function rows_carrier(::Nothing, ::Any)
    return nothing
end
function rows_carrier(X::SampleBufferState, nx::VecStr)
    return ReturnsResult(; nx = nx, X = sample_buffer(X), pnl = buffer_panel(X))
end
function rows_carrier(::SampleBufferState, ::Nothing)
    return throw(IsNothingError("the head keeps rows and pins no asset names: a Returns Result that carries rows carries their names, so the carrier the rules read cannot be formed. Hand the head a carrier whose `nx` is set."))
end
"""
    buffer_panel(X::SampleBufferState)

Returns the active masks of a rows buffer as the [`AssetPanel`](@ref) of a carrier, or `nothing` when the buffer records none, which is the static panel.

The panel uses the active mask as its estimation mask too, because [`step_active_mask`](@ref) admits only that panel.

# Related

  - [`rows_carrier`](@ref)
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

It is the carrier's timestamp. When the carrier holds none, it is the row's index in the whole fold, `n + 1` after `n` rows.

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
    @warn("Held Step at $(ts): $(join([h.reason for h in held], "; ")).")
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
        strict_diagnostic(held_gap_msg(held, nx), strict)
    end
    return nothing
end
"""
    partial_fit!(opt::OnlinePortfolioSelection, rd::ReturnsResult)

Folds the rows of a carrier into the head's recursion as the Block Step, and reads nothing out.

Each row of `rd` is one Online Update, in order, and the state never learns the cadence of the loop. A fold of `k` rows is `k` updates and no read-out. After a fold of rows `1:t`, `optimise(opt)` equals `optimise(opt, rd[1:t])` exactly, because both take the same `t` single-row updates.

# Arguments

  - `opt`: The head.
  - `rd`: The carrier holding one row or a block of them, `observations × assets`.

# Validation

  - Everything [`fold_online_selection`](@ref) refuses.

# Returns

  - `opt`: The head, with its `cache` rebound to the state after the last row.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`fold_online_selection`](@ref)
  - [`optimise`](@ref)
"""
function partial_fit!(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any,
                                                    <:Option{<:OnlinePortfolioSelectionState}},
                      rd::ReturnsResult)
    return rebuild_estimator(opt, (; cache = fold_online_selection(opt, opt.cache, rd)))
end
"""
    online_step_fold(opt::OnlinePortfolioSelection, ctx::TimeDependentContext, rd::ReturnsResult)

Folds the fold's rows into the head inside the fold's own Allocation Set.

The head is the one family whose online step is the optimisation. The Constrained Update projects the raw step of every row onto `set`, so the step reads a schedule on `set`, and not only the read-out. So the method resolves the entry here, in the same fold and before the read-out resolves the other schedules. The head that the loop threads on keeps the schedule, so fold `i + 1` resolves from the schedule and not from entry `i`. The step writes no field but the state, so the method returns the head with its `cache` rebound.

# Arguments

  - `opt`: The head the loop threads, with its schedules unresolved.
  - `ctx`: The fold's context.
  - `rd`: The carrier of the rows the fold has gained.

# Returns

  - `opt`: The head, with its `cache` rebound and its schedules unresolved.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`online_step_fold`](@ref)
  - [`fold_online_selection`](@ref)
  - [`thread_online_folds!`](@ref)
"""
function online_step_fold(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                        <:Any,
                                                        <:Option{<:OnlinePortfolioSelectionState}},
                          ctx::TimeDependentContext, rd::ReturnsResult)
    set = opt.set
    set = isa(set, TimeDependent) ? time_dependent_value(set, ctx) : set
    return rebuild_estimator(opt,
                             (; cache = fold_online_selection(opt, opt.cache, rd, set)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the Recursion Read-out, the allocation of the state as a [`NaiveOptimisationResult`](@ref) over the assets active at the last folded row.

The read-out reads the state and writes nothing, so every call on one state returns the same Result. Its retcode is an [`OptimisationSuccess`](@ref). Its `res` is the [`HeldStep`](@ref) record of the last folded row when that row was held, and `nothing` otherwise. A reduced allocation with no mass is the one failure from which a rule cannot recover. It happens when every held asset is inactive at the last row. The read-out then returns an [`OptimisationFailure`](@ref) with `NaN` weights, so that a fallback chain continues.

# Mathematical definition

```math
\\begin{align}
\\bar{w}_i &= \\frac{w_{T+1,i}}{\\sum_{j \\in \\mathcal{I}} w_{T+1,j}}\\,, \\quad i \\in \\mathcal{I}\\,.
\\end{align}
```

Where:

  - ``\\bar{w}_i``: Weight of asset ``i`` in the Result.
  - ``w_{T+1,i}``: Weight of asset ``i`` in the allocation of the state, the Next-Period Allocation after ``T`` rows.
  - ``\\mathcal{I}``: Investable Mask of the read-out, the assets active at the last folded row, or every asset under a static panel.
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
function online_selection_readout(opt::OnlinePortfolioSelection,
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

 1. Refuse `dims != 1` through [`assert_returns_result_dims`](@ref).
 2. Reset every schedule of the head to its fold-less value through [`reset_time_dependent_estimator`](@ref).
 3. Fold every row of `rd` from the Start Allocation, with no carried state, through [`fold_online_selection`](@ref), giving `state`.
 4. Read the Next-Period Allocation out of `state` through [`online_selection_readout`](@ref).

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`fold_online_selection`](@ref)
  - [`online_selection_readout`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(opt::OnlinePortfolioSelection, rd::ReturnsResult; dims::Int = 1,
                   kwargs...)
    assert_returns_result_dims(dims)
    opt = reset_time_dependent_estimator(opt)
    state = fold_online_selection(opt, nothing, rd)
    return online_selection_readout(opt, state)
end
"""
    optimise(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...) -> NaiveOptimisationResult

Runs the Causal Pass over `rd`, and returns the Next-Period Allocation.

# Arguments

  - `opt`: The head.
  - $(arg_dict[:rd]) Every row is one Online Update, in order.
  - `dims`: Must be `1`. A `ReturnsResult` is always observations × assets, so `dims == 2` throws `ConflictingArgumentError`.
  - `kwargs`: Additional keyword arguments, ignored.

# Validation

  - No field in the tree of `opt` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise, through [`assert_batch_entry`](@ref).

# Returns

  - `res::NaiveOptimisationResult`: The Next-Period Allocation, with `pr = nothing`.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`partial_fit!`](@ref)
  - [`online_selection_readout`](@ref)
"""
function optimise(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)::NaiveOptimisationResult
    assert_batch_entry(opt, "`optimise`")
    return _optimise(opt, rd; dims = dims, kwargs...)
end
"""
    optimise(opt::OnlinePortfolioSelection; kwargs...) -> OptimisationResult

Returns the Recursion Read-out, the allocation of the state as a Result, and runs no batch path.

The method refuses by name a head that has taken no step. The read-out fails in one way, when the allocation has no mass on the assets active at the last row. The fallback chain then runs as the batch verb runs it, and each fallback reads out through its own `optimise(fb)`. A fallback that needs rows has folded none and refuses. So the fallback that serves here is a head that needs no rows, such as [`PreviousWeights`](@ref).

# Algorithm

 1. Refuse a head whose `cache` is `nothing`.
 2. Reset every schedule of the head to its fold-less value through [`reset_time_dependent_estimator`](@ref).
 3. Read the state out through [`online_selection_readout`](@ref), giving `res`.
 4. Return `res` when it succeeded or when the head has no fallback.
 5. Otherwise warn, read the fallback out with `optimise(opt.fb)`, and return its Result through [`factory`](@ref), which records the failed pair of `opt` and `res`.

# Arguments

  - `opt`: The stepped head.
  - `kwargs`: Ignored.

# Validation

  - `opt.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `res::OptimisationResult`: The Next-Period Allocation, or the Result of the fallback chain when the read-out fails.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`online_selection_readout`](@ref)
  - [`partial_fit!`](@ref)
"""
function optimise(opt::OnlinePortfolioSelection; kwargs...)
    @argcheck(!isnothing(opt.cache),
              ArgumentError("`optimise(opt)` with no returns reads the state the online step wrote, and this `OnlinePortfolioSelection` has taken no step: its `cache` is `nothing`. Fold observations with `partial_fit!(opt, rd)` first, or pass the returns to `optimise(opt, rd)`."))
    opt = reset_time_dependent_estimator(opt)
    res = online_selection_readout(opt, opt.cache)
    if isa(res.retcode, OptimisationSuccess) || isnothing(opt.fb)
        return res
    end
    fb = Tuple{OptimisationEstimator, OptimisationResult}[(opt, res)]
    @warn("Using fallback method. Please ignore previous optimisation failure warnings.")
    return factory(optimise(opt.fb), fb)
end
"""
    online_readout(opt::OnlinePortfolioSelection)

Refuses by name, because the head has no batch estimator to return.

A Recursion Read-out rebuilds no carrier and runs no batch path. `optimise(opt)` reads the state directly.

# Related

  - [`online_readout`](@ref)
  - [`optimise`](@ref)
"""
function online_readout(::OnlinePortfolioSelection)
    return throw(ArgumentError("an `OnlinePortfolioSelection` head has a Recursion Read-out, not a reconstitution: its state holds the next allocation and no carrier to rebuild, so there is no batch estimator to hand back. Read it out with `optimise(opt)`."))
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
  - `pws`: The scheme's Previous-Weights Source, or `nothing`.

# Validation

  - `pws` is not `nothing` when `opt.fees` carries a `tn` term, in any entry that a schedule visibly holds. An `ArgumentError` is thrown otherwise.

# Related

  - [`online_folds`](@ref)
  - [`assert_online_entry`](@ref)
  - [`fees_carry_turnover`](@ref)
  - [`DriftedWeights`](@ref)
"""
function assert_online_fee_source(opt::OnlinePortfolioSelection, pws)::Nothing
    if isnothing(pws) && fees_carry_turnover(opt.fees)
        throw(ArgumentError("an `OnlinePortfolioSelection` head whose `fees` carry a turnover term needs a Previous-Weights Source on the walk-forward: the recursion reads its own allocation, so without one the fee would price the distance between two targets rather than the trade the fund makes. Set `pws = DriftedWeights()` on the scheme, or drop `tn` from the head's fees."))
    end
    return nothing
end
export OnlinePortfolioSelection
