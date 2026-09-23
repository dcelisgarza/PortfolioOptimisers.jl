"""
$(DocStringExtensions.TYPEDEF)

The online portfolio selection head: a naive optimiser that updates its allocation from each realised price relative through the Online Selection Rule on `alg`, fitting no moment and solving no programme of its own.

Every member of the family is one recursion, `w_{t+1} = f(w_t, x_t)`, and one wealth, `S_T = Π_t ⟨w_t, x_t⟩`: the allocation is rebalanced to its target at the start of every period. The head writes everything the rules share. **The batch verb is the Causal Pass**: `optimise(opt, rd)` starts from the Start Allocation `w0`, applies the Online Update to every row of `rd` in order, and answers the **Next-Period Allocation** — after rows `1:T`, the portfolio for period `T + 1`, one more update than a backtest driver that stops at the last row held takes. The pass is rebalanced every row and free; `predict(res, rd)` on its Result charges the head's `fees` with no drift. **The online verbs are the same recursion**: `partial_fit!(opt, rd)` folds each row into the state as the Block Step — a fold of `k` rows is `k` single-row updates, so a walk-forward with `test_size = k` holds the Next-Period Allocation as of the block's end for `k` periods — and `optimise(opt)` with no data is the **Recursion Read-out**, the state's allocation wrapped in a Result with no batch path run. The identity `optimise(opt)` after folding rows `1:t` equals `optimise(opt, rd[1:t])` therefore holds exactly, at every `test_size`. A walk-forward's held path differs from the batch pass by the drift of the block-start target over the block and by the Block Step's cadence; the difference is documented, not tested.

**The Start Allocation** is `w0`, `nothing` by default. Absent, the recursion starts at `1/N` over the whole pinned universe, unlisted names included: on a universe where `k` of `N` assets are unlisted at the first row, the recursion parks `k/N` of its weight in cash-like legs that see `x = 1` until the rule moves it, and the fund's tilts are scaled by `(N − k)/N` for as long as that lasts; a caller who wants the recursion to be the fund from row one gives `w0` over the listed assets with zeros elsewhere and accepts what the rule's geometry does with a zero. A given `w0` is over the pinned names, pinned and viewed with them, and projected once onto the Allocation Set in the rule's geometry at the first step, as the uniform start is, so a start outside the set is made feasible and never refused; a set that reads the head's rows has nothing to project onto before the first row, and holds the start as given until the first update. It reaches the first Online Update as `w`: a rule that reads `w` continues from it, and a rule that does not — the constant rebalanced portfolio, the mixture, the Newton step — replaces it after one period, so the fund holds `w0` for exactly one period.

**The Online Update reads the recursion's own allocation and no flag changes that**: the loop's previous weights reach the head through [`factory`](@ref) alone, into `fees` and `fb`, never into the recursion. A turnover fee on this family is measured against the fund's held book or against nothing the family does, so the fold loop's online arm refuses, by name and before any fold, a head whose `fees` carry a `tn` term when the walk-forward's Previous-Weights Source is `nothing`; `pws = DriftedWeights()` is the whole configuration. A head with no `tn` fee is not checked.

**A non-finite return is kept, and read two ways.** The rule's step sees `x = 1` there — the leg sat in cash — silently at an asset the panel marks inactive, and as a Held Gap at an active one, a warning by default and a refusal under `strict`. The rows buffer keeps the cell as `NaN` with the row's active mask beside it, and every statistic over the rows — a forecaster's mean, a Risk Loss's or a programme set's prior, a leader's re-solve — reads them as the batch verb reads a carrier and reduces to its own Coverage Universe: a plain estimator drops an asset with a gap anywhere in its window, a mask-aware one answers it from the rows it has, and a kernel over price relatives reads the gap as one. The recursion's allocation never carries a forced zero from the head, and a relisting asset re-enters at the recursion's own weight; a programme set that fits on the rows writes a zero at a leg its prior cannot price, as a batch head does, and re-admits the leg once it can. Under a time-varying panel the read-out's Investable Mask is the last folded row's active mask: the full allocation is sliced to it and renormalised, and the Result expands it back with a zero at every non-investable asset.

**The Allocation Set on `set` is the Constrained Update's**: every rule's raw step is projected onto it in the rule's own Projection Geometry through [`project`](@ref). The default [`BoundedAllocationSet`](@ref) is the simplex and every projection onto it is closed form, so the default configuration solves nothing; a [`ProgrammeAllocationSet`](@ref) admits the full constraint vocabulary and its projection is a programme. A programme can fail, and the step is then a **Held Step**: the projection answers the Price-Adjusted Allocation it was handed, so the fund trades nothing that period, the rule's carrier still absorbs the row, the head warns once with the row's timestamp, and the Recursion Read-out's retcode is an [`OptimisationSuccess`](@ref) carrying the last folded row's [`HeldStep`](@ref) record, so a fallback chain never runs on a hold. A negative lower bound under an entropic rule is refused at construction.

`merge_states` on the state and `Online(head)` are refused by name: an update is order-dependent, and the family never refits from a buffer.

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

Keywords correspond to the struct's fields. `set`, `fees` and `fb` may hold a [`TimeDependent`](@ref) per-fold schedule; `alg`, `w0` and `strict` are static.

## Validation

  - `w0`: non-empty and finite, when given; it is projected onto the set at the first step, so it need not lie in it.
  - If `fees` is a [`FeesEstimator`](@ref): `!isnothing(set.sets)`. A schedule on `set` is held to it per entry.
  - Everything [`assert_geometry_admits_set`](@ref) refuses: a negative lower bound under an entropic rule. A schedule on `set` is held to it per entry.
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
    The Allocation Set every allocation of the recursion lies in, or a [`TimeDependent`](@ref) schedule of one per fold. Entry `i` is fold `i`'s complete set: the fold loop swaps it in before the fold's rows are folded, so the rows of fold `i` step inside entry `i`, and the read-out reports it.
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

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. `fees` and `fb` default to `nothing` and are omitted; `set` does not, so a fold-less solve runs the default Allocation Set rather than no set at all.

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

The recursion never reads them: `alg`, `set`, `w0` and `cache` are carried unchanged.

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

Slice the head to the selected assets: the rule, the set and the fee by their own views, the Start Allocation sliced and renormalised, and the state through [`port_opt_view`](@ref) on [`OnlinePortfolioSelectionState`](@ref).

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(opt::OnlinePortfolioSelection, i, args...)
    return OnlinePortfolioSelection(; alg = port_opt_view(opt.alg, i, args...),
                                    set = port_opt_view(opt.set, i, args...),
                                    w0 = renormalised_view(opt.w0, i),
                                    fees = port_opt_view(opt.fees, i, args...), fb = opt.fb,
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

The rows a per-fold schedule of Allocation Sets needs: the maximum over every entry a vector schedule holds and over its explicit `default`, and `nothing` — unbounded — for a callable.

The buffer is sized once, when the state is seeded, and it is never resized, so the cap must answer for every fold and not for the one the seed runs in. A callable's entries do not exist before the fold does, so the cap keeps every row folded so far and no fold's set can read a row the cap dropped.

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

The rows the head's buffer keeps: the maximum over the rule tree and the Allocation Set, `nothing` being unbounded.

A schedule on `set` answers over every entry it holds, because the buffer is sized once and read by every fold.

# Related

  - [`rows_needed`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
"""
function rows_needed(opt::OnlinePortfolioSelection)
    return rows_needed_max(rows_needed(opt.alg), rows_needed(opt.set))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The head's Allocation Set outside every fold loop: its own, or the fold-less value of a schedule on `set`.

A schedule is defined over the folds of a cross-validation scheme, so a [`partial_fit!`](@ref) a caller takes by hand — and the Causal Pass, whose reset has already run — steps inside the schedule's `default`, else inside [`BoundedAllocationSet`](@ref).

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

Seeds the head's state on its first row: pins the names and a static panel, forms the Start Allocation, projects it once onto the set in the rule's geometry, and seeds the rule's carrier on the set through [`rule_state_seed`](@ref) and the rows buffer.

The uniform start and a given `w0` meet the set alike, so the allocation held during the first period lies in the set wherever the set can be formed without rows. A set that reads the head's rows — a programme set with a fitted risk ceiling or a tracking error — has no constraints to form before the first row, so the start is held as given and the first Online Update projects it, as that update holds every step until the head has two rows. A programme that fails at the start holds the start as given and warns, as a row's Held Step does.

# Arguments

  - `opt`: The head.
  - `rd`: The carrier of the first block.
  - `set`: The Allocation Set, resolved over the pinned universe.

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

Folds every row of a carrier into the head's state, in order: the Block Step, and the whole of the Causal Pass.

# Algorithm

 1. Refuse what [`step_active_mask`](@ref) refuses, and resolve the Allocation Set over the pinned universe.
 2. Seed the state on the first block, or pin-and-check the carried one, through [`online_selection_pin`](@ref).
 3. Per row, through [`online_selection_row!`](@ref): name every non-finite cell at an active asset the recursion holds as a Held Gap through the head's `strict`, silently where the row's active mask is `false`; push the row verbatim, with its active mask, into the rows buffer where the tree keeps one; form `x` through [`price_relative`](@ref), one at a gap; call [`online_update!`](@ref) with the buffer read out as a carrier ([`rows_carrier`](@ref)) inside a [`ProjectionStep`](@ref); warn on a Held Step; record the row's active mask, timestamp and hold.

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
               ts = row_timestamp(rd.ts, t, n))
        st, w, X, hold = online_selection_row!(opt, st, w, X, row, rd.nx, set)
        n += 1
        last_amsk = isnothing(row.amsk) ? nothing : BitVector(row.amsk)
    end
    return OnlinePortfolioSelectionState(; n = n, w = w, st = st, X = X, nx = state.nx,
                                         pnl = state.pnl, amsk = last_amsk,
                                         ts = fold_column(state.ts, rd.ts, nothing),
                                         hold = hold)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The state a block folds into: seeded on the first block through [`online_selection_seed`](@ref), and on every later one the state the head carries, after the pin-and-check every host runs — the names and the static panel against the first step's, the width against the state's, and the timestamps present at every block or at none.

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
    assert_column_presence(cache.ts, rd.ts, cache.n, :ts)
    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

One row of the recursion: name the row's Held Gaps, push the row verbatim with its active mask into the rows buffer where the tree keeps one, form the price relative `x` through [`price_relative`](@ref), and take the Online Update over the buffer read out as a carrier inside a [`ProjectionStep`](@ref), so every projection of the row reads the rows and reports a hold to the head.

The gap is read in two ways, and each reader gets the one the rest of the library gives it. The Online Update and every kernel over the price relatives read one at a gap — the leg sat in cash, the Held Gap's own reading — so the recursion stays over the full pinned universe and never carries a forced zero. Every statistic over the rows — a forecaster's mean, a Risk Loss's prior, a programme set's prior, a leader's re-solve — reads the carrier as a batch verb reads one: `NaN` where there was no return, the active mask beside it, and reduces to its own Coverage Universe, so a window over an unlisted span never meets a constant column and a relisted asset's statistic is never diluted by returns that never happened.

A row one of whose projections was held is warned once, naming the row's timestamp and each hold, and answers the first [`HeldStep`](@ref) record; a row whose projections all solved answers `nothing`.

# Arguments

  - `opt`: The head.
  - `st`: The rule's carrier.
  - `w`: The allocation held during the row's period.
  - `X`: The rows buffer, or `nothing`.
  - `row`: The row: `r`, its returns verbatim; `amsk`, its active mask or `nothing`; `ts`, its timestamp or its index in the fold.
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
    (; r, amsk, ts) = row
    report_row_gaps(r, w, amsk, nx, opt.strict)
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

The rows buffer read out as the carrier the Online Update is handed: a [`ReturnsResult`](@ref) of the buffer's rows verbatim under the pinned names, with the buffer's active masks as a time-varying Asset Panel where it records them ([`buffer_panel`](@ref)), or `nothing` when the tree keeps no rows.

It is the door every statistic over the rows enters by, and it is the batch door: `prior(pe, rd)`, `mean(me, rd.X, rd.pnl)` and `optimise(opt, rd)` reduce to the Coverage Universe of the window exactly as they do on a carrier the ingestion layer built. No copy is taken; the carrier views the buffer's valid region. A buffer with no pinned names is refused by name: a Returns Result that carries rows carries their names by its own contract, so the head never meets the pair.

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

The active masks a rows buffer holds, as the [`AssetPanel`](@ref) a carrier states them on — the same mask as the estimation mask, which is what [`step_active_mask`](@ref) admits — or `nothing` when the buffer records none, which is the static panel.

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

The timestamp a row of a block folds under: the carrier's, or the row's index in the whole fold, `n + 1` after `n` rows, when the carrier holds none.

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

Warns once for a row one of whose projections was held, naming the row's timestamp and every hold, and answers the first [`HeldStep`](@ref) record; answers `nothing` for a row whose projections all solved.

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

A non-finite cell the row's active mask marks inactive is passed in silence, because an inactive asset has no return by definition. One at an active asset, or at any asset when there is no mask, is a Held Gap where the recursion's weight is non-zero, reported through [`strict_diagnostic`](@ref) with [`held_gap_msg`](@ref): a warning by default, a refusal under `strict`. The row is not filled: the buffer keeps the cell as it is, and the Online Update reads it as one through [`price_relative`](@ref).

# Arguments

  - `r`: The row.
  - `w`: The allocation held during the row's period.
  - `amsk`: The row's active mask, or `nothing`.
  - `nx`: The asset names, for the message, or `nothing`.
  - `strict`: Whether a Held Gap is an error.

# Returns

  - `nothing`.

# Related

  - [`fold_online_selection`](@ref)
  - [`price_relative`](@ref)
  - [`held_gap_msg`](@ref)
"""
function report_row_gaps(r::AbstractVector, w::AbstractVector,
                         amsk::Option{<:AbstractVector{<:Bool}}, nx::Option{<:VecStr},
                         strict::Bool)::Nothing
    held = Tuple{Int, Int}[]
    for i in eachindex(r)
        if isfinite(r[i]) || (!isnothing(amsk) && !amsk[i]) || iszero(w[i])
            continue
        end
        push!(held, (1, i))
    end
    if !isempty(held)
        strict_diagnostic(held_gap_msg(held, nx), strict)
    end
    return nothing
end
"""
    partial_fit!(opt::OnlinePortfolioSelection, rd::ReturnsResult)

Folds the rows of a carrier into the head's recursion, without reading out: the Block Step.

Each row of `rd` is one Online Update, in order, and the state never learns the loop's cadence; a fold of `k` rows is `k` updates and no read-out. The identity with the batch verb is exact: after folding rows `1:t`, `optimise(opt)` equals `optimise(opt, rd[1:t])`, because both take the same `t` single-row updates.

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

The head is the one family whose online step *is* the optimisation: every row's raw step is projected onto `set`, so a schedule there is read by the step and not by the read-out alone. The entry is resolved here, one fold before the read-out resolves the rest, and the head the loop threads on keeps the schedule, so fold `i + 1` resolves from the schedule and not from entry `i`. The state is the only field the step writes, so carrying it back is a rebind of `cache`.

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

The Recursion Read-out: wraps the state's allocation in a [`NaiveOptimisationResult`](@ref), reduced to the last folded row's active mask and renormalised, with the resolved bounds and the head's fee.

The read-out is pure: it reads the state and writes nothing, so it is callable any number of times for the same answer. Its retcode is the last step's: a plain [`OptimisationSuccess`](@ref), or one whose `res` is the [`HeldStep`](@ref) record when the last folded row was held. A reduced allocation with no mass — every held asset inactive at the last row — is the one failure a rule cannot recover from, and answers an [`OptimisationFailure`](@ref) with `NaN` weights so a fallback chain walks on.

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

Run the Causal Pass.

Internal dispatch called by [`optimise`](@ref). Folds every row of `rd` through [`fold_online_selection`](@ref) from the Start Allocation, reading and writing no state the head may carry, and reads the Next-Period Allocation out through [`online_selection_readout`](@ref).

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

Run the Causal Pass over `rd` and answer the Next-Period Allocation.

# Arguments

  - `opt`: The head.
  - $(arg_dict[:rd]) Every row is one Online Update, in order.
  - `dims`: Must be `1`. A `ReturnsResult` is always observations × assets, so `dims == 2` throws `ConflictingArgumentError`.
  - `kwargs`: Additional keyword arguments, ignored.

# Validation

  - No field in the tree of `opt` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise, through [`assert_batch_entry`](@ref).
"""
function optimise(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)::NaiveOptimisationResult
    assert_batch_entry(opt, "`optimise`")
    return _optimise(opt, rd; dims = dims, kwargs...)
end
"""
    optimise(opt::OnlinePortfolioSelection; kwargs...) -> OptimisationResult

The Recursion Read-out: the state's allocation as a Result, with no batch path run.

A head that has taken no step is refused by name. On the one failure the read-out can answer — an allocation with no mass on the assets active at the last row — the fallback chain walks as the batch verb walks it, each fallback read out through its own `optimise(fb)`; a fallback that needs rows has folded none and refuses, so the one that serves here is a row-free head such as [`PreviousWeights`](@ref).

# Arguments

  - `opt`: The stepped head.
  - `kwargs`: Ignored.

# Validation

  - `opt.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

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

Refused by name: a Recursion Read-out reconstitutes no carrier and runs no batch path, so there is no batch estimator to hand back. `optimise(opt)` reads the state directly.

# Related

  - [`online_readout`](@ref)
  - [`optimise`](@ref)
"""
function online_readout(::OnlinePortfolioSelection)
    return throw(ArgumentError("an `OnlinePortfolioSelection` head has a Recursion Read-out, not a reconstitution: its state holds the next allocation and no carrier to rebuild, so there is no batch estimator to hand back. Read it out with `optimise(opt)`."))
end
"""
    held_timestamps(opt::OnlinePortfolioSelection)

The timestamps the head's state holds, uncapped, or `nothing` before the first step.

# Related

  - [`held_timestamps`](@ref)
  - [`Resume`](@ref)
"""
function held_timestamps(opt::OnlinePortfolioSelection)
    return isnothing(opt.cache) ? nothing : opt.cache.ts
end
"""
    Online(::OnlinePortfolioSelection, args...; kwargs...)

Refuses `Online(head)` at the construction door, naming the three routes that give the two answers the wrapper would give.

[`Online`](@ref) declares a refit from a buffer, and for a head both of its settings are a batch walk-forward the library already runs. Uncapped, the refit is the expanding walk-forward, whose allocation equals the online arm's exactly, because both take the same sequence of single-row updates from `w0`; it pays a pass over every row folded so far where the online arm pays one row. Capped, the refit is the rolling walk-forward, whose allocation is the recursion restarted from `w0` inside each window. So the wrapper adds no answer, and the refusal costs the caller nothing.

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

Whether a fee, or any fee a per-fold schedule can be seen to hold, carries a turnover term.

A static fee answers for its own `tn`. A [`TimeDependent`](@ref) schedule answers for every entry of a vector value, descending into per-fold vector entries, and for its explicit `default`; a callable's output cannot be inspected before the fold exists and contributes `false`, as every other value does for [`needs_previous_weights`](@ref).

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

A turnover fee on this family is measured against the fund's held book or against nothing the family does: the recursion reads its own allocation, so without a source the fee would price the distance between two targets, which reads buy-and-hold as trading and constant rebalancing as free. The head and the walk-forward meet first at this door, so it is the earliest point the pairing exists. A head with no `tn` fee, and any other estimator, passes. A per-fold schedule on `fees` is read through [`fees_carry_turnover`](@ref): every entry it can be seen to hold is checked, and a callable's output, which does not exist before the fold does, is not.

# Arguments

  - `opt`: The estimator handed to the loop.
  - `pws`: The scheme's Previous-Weights Source, or `nothing`.

# Validation

  - `pws` is not `nothing` when `opt.fees` carries a `tn` term, in any entry a schedule can be seen to hold. An `ArgumentError` is thrown otherwise.

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
