"""
    pipe_writes(o::Online) = pipe_writes(o.est)
    pipe_reads(o::Online) = pipe_reads(o.est)

An [`Online`](@ref) step writes and reads the slots of the estimator that it wraps.

`Online(EmpiricalPrior(); max_history = w)` is a prior step whose window is capped. The warm-up of the online arm resolves it to a plain prior that carries a buffer.

# Related

  - [`Online`](@ref)
  - [`pipe_writes`](@ref)
  - [`update_online_estimator(p::Pipeline)`](@ref)
"""
pipe_writes(o::Online) = pipe_writes(o.est)
pipe_reads(o::Online) = pipe_reads(o.est)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an [`Online`](@ref) step that reaches a fit with no folds, by name.

The warm-up of the online arm resolves a wrapper, and `fit(pipe, data)` runs no warm-up. A wrapper that reaches it has no buffer to seed and no step to fold. A fit of the wrapped estimator as a plain one would return a batch answer under an online declaration, so the method refuses the step instead.

# Validation

  - The method always throws an `ArgumentError`. The message names the wrapped estimator and the two ways out: an Online Scheme, or the plain estimator.

# Related

  - [`run_step`](@ref)
  - [`Online`](@ref)
  - [`update_online_estimator`](@ref)
"""
function run_step(o::Online, ::PipelineContext)
    return throw(ArgumentError("an `Online($(typeof(o.est).name.name))` step is a declaration the online arm of the fold loop resolves at its warm-up, and `fit(pipe, data)` has none: run the pipeline through an Online Scheme (`OnlineIndexWalkForward` or `OnlineDateWalkForward`), or hand the step the plain estimator."))
end
"""
$(DocStringExtensions.TYPEDEF)

Holds every block of observations that `Online(pipe)` folds, concatenated, so the read-out is a batch fit over them.

This state is the declared refit of a [`Pipeline`](@ref), and no step folds under it. The buffer holds the input of the pipeline as the caller gave it, at the price level or at the returns level, and `fit(pipe)` runs `fit(pipe, data)` over the buffer. So the read-out is exact for every configuration, at the cost of a batch fit. With a cap the buffer keeps the last `max_history` rows, and the run equals the rolling batch walk-forward.

The type of the state in `pipe.cache` selects the route. A [`ReturnsBufferState`](@ref) is the Fold Context of the host route, and a `PipelineBufferState` is the refit route.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PipelineBufferState(; data::Option{<:Prices_RR} = nothing, max_history::Option{<:Integer} = nothing)

## Validation

  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`Online`](@ref)
  - [`vcat_carrier_rows`](@ref)
  - [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref)
"""
@concrete struct PipelineBufferState <: AbstractPartialFitState
    """
    The observations folded so far, as one carrier of the pipeline's input level, or `nothing` before the first block.
    """
    data
    """
    $(field_dict[:pf_max_history])
    """
    max_history
end
function PipelineBufferState(; data::Option{<:Prices_RR} = nothing,
                             max_history::Option{<:Integer} = nothing)::PipelineBufferState
    if !isnothing(max_history)
        @argcheck(max_history > 0,
                  DomainError(max_history,
                              "max_history is the number of observations the buffer keeps, so it must be positive. Pass `nothing`, the default, to keep every observation folded."))
    end
    return PipelineBufferState(data, max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Appends a block of observations to a [`PipelineBufferState`](@ref), and drops the oldest rows past the cap.

The state is immutable, so the method returns a new state and leaves `state` as it was.

# Algorithm

 1. Concatenate the held carrier and `data` by rows through [`vcat_carrier_rows`](@ref), giving `held`. When the state holds no carrier yet, `held` is `data`.
 2. Count the rows of `held`, giving `n`.
 3. When `max_history` is set and `n > max_history`, view the last `max_history` rows of `held`.
 4. Return a new state that holds `held` and carries the same cap.

# Validation

  - Everything [`vcat_carrier_rows`](@ref) refuses.

# Related

  - [`PipelineBufferState`](@ref)
  - [`vcat_carrier_rows`](@ref)
"""
function partial_fit!(state::PipelineBufferState, data::Prices_RR)
    held = isnothing(state.data) ? data : vcat_carrier_rows(state.data, data)
    n = carrier_rows(held)
    if !isnothing(state.max_history) && n > state.max_history
        held = pipeline_data_view(held, (n - state.max_history + 1):n)
    end
    return PipelineBufferState(; data = held, max_history = state.max_history)
end
function merge_states(a::PipelineBufferState, b::PipelineBufferState)
    @argcheck(isequal(a.max_history, b.max_history),
              ArgumentError("two PipelineBufferStates merge under one cap; got $(a.max_history) and $(b.max_history)"))
    if isnothing(b.data)
        return a
    end
    return partial_fit!(a, b.data)
end
function Base.copy(x::PipelineBufferState)
    return PipelineBufferState(x.data, x.max_history)
end
function port_opt_view(x::PipelineBufferState, i, args...)
    data = isnothing(x.data) ? nothing : pipeline_asset_view(x.data, i)
    return PipelineBufferState(; data = data, max_history = x.max_history)
end
function online_state_seed(::Pipeline, max_history::Option{<:Integer})
    return PipelineBufferState(; max_history = max_history)
end
"""
    rebuild_estimator(p::Pipeline, repl::NamedTuple)

Rebuilds a [`Pipeline`](@ref) with the fields in `repl` replaced, through the positional constructor.

The keyword constructor derives the names from the steps. A rebuild that seeds a state must keep the names as they are, and only the positional constructor takes them as given.

# Related

  - [`rebuild_estimator`](@ref)
  - [`Online`](@ref)
"""
function rebuild_estimator(p::Pipeline, repl::NamedTuple)
    return Pipeline(get(repl, :names, p.names), get(repl, :steps, p.steps),
                    get(repl, :cache, p.cache))
end
"""
    step_estimator(step)

Returns the estimator that a [`Pipeline`](@ref) step stands for, unwrapped from a [`PipelineStep`](@ref) and from an [`Online`](@ref) declaration.

The online step classifies each step by this estimator. A callable step comes back as it is.

# Related

  - [`rewrap_step`](@ref)
  - [`pipeline_row_owner`](@ref)
"""
step_estimator(step) = step
step_estimator(ps::PipelineStep) = step_estimator(ps.est)
step_estimator(o::Online) = step_estimator(o.est)
"""
    rewrap_step(step, est)

Puts a folded estimator back in the wrapper that its step came in.

A [`PipelineStep`](@ref) keeps its reads, its writes and its target, and a bare step is the estimator itself. An [`Online`](@ref) wrapper never comes back, because the warm-up resolved it before the first fold.

# Related

  - [`step_estimator`](@ref)
"""
rewrap_step(::Any, est) = est
rewrap_step(ps::PipelineStep, est) = PipelineStep(est, ps.reads, ps.writes, ps.target)
"""
    is_data_step(step) -> Bool

Answers whether a [`Pipeline`](@ref) step writes a data slot, `:prices` or `:returns`.

A data step changes the rows that the row owner folds.

# Related

  - [`pipeline_row_owner`](@ref)
  - [`assert_online_entry(p::Pipeline)`](@ref)
"""
is_data_step(step) = pipe_writes(step) in PIPELINE_DATA_SLOTS
"""
    is_universe_step(est) -> Bool

Answers whether a data step is universe-only.

A universe-only step folds nothing. At the read-out its batch verb runs over the rows of the row owner, and the read-out applies its universe as a view. The asset selectors are universe-only, and every other step answers `false`.

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref)
"""
is_universe_step(::Any) = false
is_universe_step(::AbstractAssetSelector) = true
"""
    is_row_owner(est) -> Bool

Answers whether a step can own the rows of the online step of a [`Pipeline`](@ref).

A prior estimator can, and so can an optimisation step of any form, a schedule included. A schedule answers `true` so that the walk names the step that it meets, and [`assert_online_owner`](@ref) then states why a schedule cannot fold.

# Related

  - [`pipeline_row_owner`](@ref)
"""
is_row_owner(::Any) = false
is_row_owner(::AbstractPriorEstimator) = true
is_row_owner(::Union{<:OptE_Opt, <:TD_OptE_Opt}) = true
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finds the index of the row owner of a [`Pipeline`](@ref), or answers `0` when the pipeline has none.

A prior step owns the rows when one exists, because [`inject_context`](@ref) replaces the `pe` of the optimiser with it, and the prior of the optimiser is never fitted. Without a prior step, the optimisation step owns the rows. It is then a host itself, and it keeps its own Fold Context. A pipeline with neither has nothing to fold the rows into.

# Algorithm

 1. Find the first step whose estimator ([`step_estimator`](@ref)) is a prior, giving `k`. Return `k` when it exists.
 2. Otherwise find the first step whose estimator is a row owner ([`is_row_owner`](@ref)), giving `k`. Return `k`, or `0` when no step is one.

# Related

  - [`is_row_owner`](@ref)
  - [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref)
"""
function pipeline_row_owner(p::Pipeline)
    k = findfirst(step -> isa(step_estimator(step), AbstractPriorEstimator), p.steps)
    if !isnothing(k)
        return k
    end
    k = findfirst(step -> is_row_owner(step_estimator(step)), p.steps)
    return isnothing(k) ? 0 : k
end
"""
    online_entry_state(p::Pipeline)

Names the first state that a [`Pipeline`](@ref) carries at the entry of the online arm of the fold loop, or answers `nothing`.

The answer is `"cache"` when the pipeline holds a state of its own. Otherwise the method walks the steps in order, and answers the path of the first state under a step, prefixed with the name of the step.

# Related

  - [`online_entry_state`](@ref)
  - [`assert_online_entry(p::Pipeline)`](@ref)
"""
function online_entry_state(p::Pipeline)
    if !isnothing(p.cache)
        return "cache"
    end
    for (name, step) in zip(p.names, p.steps)
        path = online_entry_state(step_estimator(step))
        if !isnothing(path)
            return string(name, ".", path)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Names the first [`Online`](@ref) declaration under the steps of a [`Pipeline`](@ref), prefixed with the name of the step, or answers `nothing`.

Two refusals read this walk. Under `Online(pipe)` no member folds, so a wrapper below the root has no fold to seed and its cap would have no effect, and [`assert_online_entry`](@ref) refuses it by name. On the host route the warm-up resolves every wrapper, so the fold refuses a wrapper that it still meets.

# Related

  - [`assert_online_entry(o::Online{<:Pipeline})`](@ref)
  - [`step_online_member`](@ref)
"""
function pipeline_online_member(p::Pipeline)
    for (name, step) in zip(p.names, p.steps)
        path = step_online_member(step)
        if !isnothing(path)
            return isempty(path) ? string(name) : string(name, ".", path)
        end
    end
    return nothing
end
"""
    step_online_member(step)

Names the first [`Online`](@ref) declaration that one [`Pipeline`](@ref) step carries, relative to the step.

The answer is `""` when the step is itself a wrapper, and the dotted path of the wrapper when the estimator of the step holds one at any depth ([`online_wrapper_path`](@ref)). Otherwise it is `nothing`. A [`PipelineStep`](@ref) answers for its estimator, and a nested `Pipeline` answers through [`pipeline_online_member`](@ref). The walk goes to every depth because the warm-up does too. [`update_online_estimator`](@ref) resolves a wrapper under the prior of an optimiser, and under the prior of a prior host.

# Related

  - [`pipeline_online_member`](@ref)
  - [`online_wrapper_path`](@ref)
"""
step_online_member(step) = online_wrapper_path(step)
step_online_member(ps::PipelineStep) = step_online_member(ps.est)
step_online_member(p::Pipeline) = pipeline_online_member(p)
"""
    step_online_cap(step)

Reads the first `max_history` that an [`Online`](@ref) declaration of one [`Pipeline`](@ref) step carries, or answers `nothing` when no wrapper of the step carries a cap.

The refusal of a capped owner on the host route reads this cap. A cap on the owner is a window counted in the rows of the owner, and a row-local step before the owner folds a state across the front edge of that window. The walk goes to every depth, as [`step_online_member`](@ref) does, so a cap on the prior of a prior host under an optimiser counts.

# Algorithm

 1. A wrapper answers its own `max_history`. An `Online` never wraps another, so the walk does not enter it.
 2. A [`PipelineStep`](@ref) answers for its estimator.
 3. An estimator walks its estimator-valued fields ([`estimator_fields`](@ref)) in order, and answers the first cap that a field answers, or `nothing`.
 4. Any other value answers `nothing`.

# Related

  - [`step_online_member`](@ref)
  - [`assert_online_entry(p::Pipeline)`](@ref)
"""
step_online_cap(::Any) = nothing
step_online_cap(o::Online) = o.max_history
step_online_cap(ps::PipelineStep) = step_online_cap(ps.est)
function step_online_cap(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    for f in estimator_fields(est)
        cap = step_online_cap(getfield(est, f))
        if !isnothing(cap)
            return cap
        end
    end
    return nothing
end
"""
    assert_online_entry(p::Pipeline)
    assert_online_entry(o::Online{<:Pipeline})

Refuses a [`Pipeline`](@ref) that cannot take the online step, by name.

The fold loop calls it at the entry of its online arm, before any fit. The checks depend on the route. The host route folds the rows through the steps into a row owner, and it has the most checks. The refit route, `Online(pipe)`, folds no member, so it has two.

The rolling window through a Pipeline is `Online(pipe; max_history = w)`. A capped owner stays on the host route only when every data step before it is universe-only, because the read-out refits a universe-only step over the capped rows of the owner.

# Validation

On the host route, in this order:

  - The pipeline carries no state, in its own `cache` or under a step ([`online_entry_state`](@ref)), because the loop starts cold.
  - The pipeline has a row owner ([`pipeline_row_owner`](@ref)), a prior step or an optimisation step.
  - The row owner passes [`assert_online_owner`](@ref). It is not a [`TimeDependent`](@ref) schedule and not an [`OnlinePortfolioSelection`](@ref) head, and an optimisation step that owns the rows passes the checks of [`assert_online_entry`](@ref) for an optimiser. A schedule owns the rows only when no prior step precedes it. With a prior step before it, the schedule swaps a stateless step and composes.
  - Every data step before the owner can fold or defer. The check refuses a nested `Pipeline`, a data step that is not a preprocessing estimator, and a preprocessing estimator with no online form ([`supports_partial_fit`](@ref)). A callable [`PipelineStep`](@ref) that writes `:prices` or `:returns` is not a preprocessing estimator. [`PriceGapFill`](@ref) with a statistic fill and [`MissingDataFilter`](@ref) with `row_thr < 1` are window-valued, and have no online form. The message names both ways out, a [`partial_fit_transform`](@ref) for the step or a refit with `Online(pipe)`.
  - When a wrapper on the owner carries a cap ([`step_online_cap`](@ref)), every data step before the owner is universe-only ([`is_universe_step`](@ref)). The owner counts its window in its own rows, and the carry of a row-local step reads rows that the window dropped, so the run equals no batch scheme. This holds on returns input too.

On the refit route:

  - The pipeline carries no state, as on the host route.
  - No step carries an [`Online`](@ref) declaration of its own ([`pipeline_online_member`](@ref)), because no member folds under a refit and its cap would have no effect.

Each refusal throws an `ArgumentError` that names the step.

# Related

  - [`online_folds`](@ref)
  - [`online_entry_state`](@ref)
  - [`pipeline_row_owner`](@ref)
  - [`step_online_cap`](@ref)
  - [`supports_partial_fit`](@ref)
  - [`is_universe_step`](@ref)
"""
function assert_online_entry(p::Pipeline)
    path = online_entry_state(p)
    @argcheck(isnothing(path),
              ArgumentError("`Pipeline` enters the online arm of the fold loop carrying a partial-fit state at `$(path)`, and the loop starts cold: its argument is the configuration alone, and the warm-up folds the first training window into a pipeline that has folded nothing. Hand the loop the pipeline with every `cache` at `nothing`, or read the stepped pipeline out by hand with `fit_and_predict(pipe, data; test_idx)`."))
    k = pipeline_row_owner(p)
    @argcheck(k > 0,
              ArgumentError("a `Pipeline` with neither a prior step nor an optimisation step has no row owner, so the online step has nothing to fold the observations into. Add a prior or an optimiser, or refit every fold with `ff = nothing`."))
    owner = step_estimator(p.steps[k])
    assert_online_owner(owner, p.names[k])
    cap = step_online_cap(p.steps[k])
    for i in 1:(k - 1)
        step = p.steps[i]
        if !is_data_step(step)
            continue
        end
        est = step_estimator(step)
        @argcheck(!isa(est, Pipeline),
                  ArgumentError("the `$(p.names[i])` step is a nested `Pipeline` writing the `:$(pipe_writes(step))` slot before the row owner, and the online step does not recurse into a nested pipeline's data steps yet. Flatten its steps into this pipeline, or declare a refit with `Online(pipe)`."))
        @argcheck(isa(est, AbstractPreprocessingEstimator),
                  ArgumentError("the `$(p.names[i])` step writes the `:$(pipe_writes(step))` slot before the row owner and is not a preprocessing estimator, so the online step cannot fold its rows: an unknown transform applied to one more row may change every earlier row. Give the step's estimator a `partial_fit_transform` and a data-less `fit_preprocessing`, or declare a refit with `Online(pipe)`."))
        @argcheck(supports_partial_fit(est) || is_universe_step(est),
                  ArgumentError("the `$(p.names[i])` step, a `$(typeof(est).name.name)`, has no online form in this configuration: refitting it over one more observation changes what it wrote at earlier ones, so no state folded at one step can be corrected at the next. Give it a `partial_fit_transform` and a data-less `fit_preprocessing` where an exact form exists, or declare a refit with `Online(pipe)`, whose buffer refits every step from the observations seen so far."))
        @argcheck(isnothing(cap) || is_universe_step(est),
                  ArgumentError("the `$(p.names[i])` step, a `$(typeof(est).name.name)`, folds a state that reaches across the front edge of the window the `$(p.names[k])` step declares with `max_history = $(cap)`: the owner's window is counted in its own rows, and the step's carry — the last price row, the carried price, the missing counts — reads observations the window has dropped, so the run equals no batch scheme. Cap the pipeline's window with `Online(pipe; max_history = $(cap))`, which refits every step over the window and equals the rolling batch walk-forward, or drop the cap, under which the host route equals the expanding one."))
    end
    return nothing
end
function assert_online_entry(o::Online{<:Pipeline})
    path = online_entry_state(o.est)
    @argcheck(isnothing(path),
              ArgumentError("`Online(pipe)` enters the online arm of the fold loop with the pipeline carrying a partial-fit state at `$(path)`, and the loop starts cold. Hand the loop the pipeline with every `cache` at `nothing`."))
    member = pipeline_online_member(o.est)
    @argcheck(isnothing(member),
              ArgumentError("`Online(pipe)` declares a refit of the whole pipeline from its input-carrier buffer, under which no member folds, and the `$(member)` step declares an `Online` of its own: it would have no fold to seed, and its `max_history` would be silently ignored. Cap the pipeline's window with `Online(pipe; max_history = …)`, or drop the outer wrapper and let the member's own buffer fold on the host route."))
    return nothing
end
"""
    assert_online_owner(owner, name::AbstractString)

Refuses a row owner that cannot fold, by name.

A prior passes. An optimisation step passes when it passes the checks of [`assert_online_entry`](@ref) for an optimiser. A precomputed result never reaches this check. A result is not a step, so it enters a pipeline only as an entry of a schedule, and the check refuses the schedule first.

The check refuses an [`OnlinePortfolioSelection`](@ref) head because of its read-out, not because of its fold. The read-out of the head is its own recursion, so it rebuilds no carrier ([`online_readout`](@ref)). The read-out of the Pipeline rebuilds a carrier from the row owner, and then refits the universe steps over it. Two things are missing, and a carrier would supply only the first. A rule whose tree reads no rows holds no rows, so nothing exists to rebuild the carrier from. And the read-out expresses a selection as a view of the state of the owner. For a recursion that view is not the run over those columns, because the allocation is a path, the projection couples the columns, and the wealth factor reads every column.

# Validation

  - The owner is not a [`TimeDependent`](@ref) schedule. A schedule swaps the estimator that carries the state at every fold, so the optimiser of a fold never saw the rows folded before it.
  - The owner is not an [`OnlinePortfolioSelection`](@ref) head.
  - An optimisation owner passes [`assert_online_entry`](@ref).

Each refusal throws an `ArgumentError` that names the step.

# Related

  - [`assert_online_entry(p::Pipeline)`](@ref)
  - [`fold_pipeline_owner`](@ref)
  - [`online_readout`](@ref)
"""
function assert_online_owner(::AbstractPriorEstimator, ::AbstractString)
    return nothing
end
function assert_online_owner(::OnlinePortfolioSelection, name::AbstractString)
    return throw(ArgumentError("the `$(name)` step is an `OnlinePortfolioSelection` head and owns the rows of the online step, because no prior step precedes it: the head's read-out is its own recursion, so it holds no carrier for `fit(pipe)` to reconstitute the universe steps over, and a rule whose tree reads no rows holds none at all. Nor would a carrier be enough: a universe step's selection is read out as a view of the owner's state, and a view of a recursion is not the recursion over those columns. Put a prior step before the head, whose rows the read-out reads, or declare a refit with `Online(pipe)`, which fits every step over the observations folded so far."))
end
function assert_online_owner(::TimeDependent, name::AbstractString)
    return throw(ArgumentError("the `$(name)` step is a `TimeDependent` schedule of optimisers and owns the rows of the online step, because no prior step precedes it: a schedule swaps the estimator that carries the state every fold, so the optimiser a fold is handed never saw the rows folded before it. Put a prior step before it, whose state the loop threads while the schedule swaps a stateless step, or schedule a field that carries no state, or refit every fold with `ff = nothing`."))
end
function assert_online_owner(opt::OptimisationEstimator, ::AbstractString)
    return assert_online_entry(opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs the entry checks of the cross-validation doors of a Pipeline.

# Validation

  - The pipeline holds no holdout step ([`assert_no_holdout`](@ref)).
  - An `Online(pipe)` root runs only under an Online Scheme ([`folds_are_stepped`](@ref)). The wrapper resolves at the warm-up of the online arm, and a batch scheme has none.
  - A [`Resume`](@ref) runs only under a scheme that [`assert_resume_scheme`](@ref) accepts.

Each refusal throws an `ArgumentError`.

# Related

  - [`assert_no_holdout`](@ref)
  - [`folds_are_stepped`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
"""
function assert_pipeline_door(pipe::Pipeline, ::Any)
    assert_no_holdout(pipe)
    return nothing
end
function assert_pipeline_door(o::Online{<:Pipeline}, cv)
    assert_no_holdout(o.est)
    @argcheck(folds_are_stepped(cv),
              ArgumentError("`Online(pipe)` declares a refit from a buffer the online arm of the fold loop seeds at its warm-up, and this scheme is not an Online Scheme, so every fold refits from its training window already. Build the scheme with `OnlineIndexWalkForward` or `OnlineDateWalkForward`, or hand the door the plain pipeline."))
    return nothing
end
"""
    cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::CVER; ex = FLoops.ThreadedEx(), id = nothing)
    cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::MultipleRandomised; ex = FLoops.ThreadedEx(), kwargs...)

Runs a walk-forward over `Online(pipe)`, the declared refit of a [`Pipeline`](@ref) from a buffer of its input carrier.

These are the doors of the pipeline, and the fold loop threads the wrapper through them. The online arm resolves the wrapper at its warm-up into a pipeline that carries a [`PipelineBufferState`](@ref). Every fold appends its new rows to the buffer, and the read-out is `fit(pipe, buffer)`. So the run is exact for every configuration, at the cost of a batch fit per fold. This includes the window-valued steps that the host route refuses.

With `max_history = w` the buffer keeps the last `w` input rows, and the run equals the rolling batch walk-forward whose purged training window holds `w` rows. Build both schemes with the training size `w + purged_size` and the same `purged_size`, such as `IndexWalkForward(w + p, t; purged_size = p)` and `OnlineIndexWalkForward(w + p, t; purged_size = p)`.

# Validation

  - Everything [`assert_pipeline_door`](@ref) refuses, a scheme that is not an Online Scheme included.
  - Everything [`assert_online_entry`](@ref) refuses on the refit route.

# Related

  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
  - [`PipelineBufferState`](@ref)
  - [`Online`](@ref)
  - [`assert_pipeline_door`](@ref)
"""
function cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::CVER;
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                           id = nothing)
    return pipeline_cross_val_predict(o, data, cv; ex = ex, id = id)
end
function cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::MultipleRandomised;
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(), kwargs...)
    return pipeline_cross_val_predict(o, data, cv; ex = ex)
end
"""
    is_time_dependent(o::Online{<:Pipeline})
    needs_previous_weights(o::Online{<:Pipeline})
    assert_time_dependent_fold_count(o::Online{<:Pipeline}, n::Integer, all_binds::Bool = true)

Answers the three traits of the fold loop for an `Online(pipe)` root, as the pipeline that it wraps.

The fold loop reads the traits off the root that it receives. The wrapper resolves at the warm-up of the online arm, and the loop makes its per-fold copy of the pipeline, so the pipeline answers.

# Related

  - [`fold_loop`](@ref)
  - [`is_time_dependent(p::Pipeline)`](@ref)
  - [`needs_previous_weights(p::Pipeline)`](@ref)
"""
is_time_dependent(o::Online{<:Pipeline}) = is_time_dependent(o.est)
needs_previous_weights(o::Online{<:Pipeline}) = needs_previous_weights(o.est)
function assert_time_dependent_fold_count(o::Online{<:Pipeline}, n::Integer,
                                          all_binds::Bool = true)::Nothing
    return assert_time_dependent_fold_count(o.est, n, all_binds)
end
"""
    update_online_step(step)

Resolves the [`Online`](@ref) declarations of one [`Pipeline`](@ref) step at warm-up.

A wrapper step becomes the estimator that it wraps, with a seeded buffer. The method rebuilds a [`PipelineStep`](@ref) around its resolved estimator. An estimator resolves the wrappers in its own fields through [`update_online_estimator`](@ref). A callable and a result pass through unchanged.

# Related

  - [`update_online_estimator(p::Pipeline)`](@ref)
  - [`update_online_estimator`](@ref)
"""
update_online_step(step) = step
update_online_step(o::Online) = update_online_estimator(o)
function update_online_step(est::Union{<:AbstractEstimator,
                                       <:StatsBase.CovarianceEstimator})
    return update_online_estimator(est)
end
function update_online_step(ps::PipelineStep)
    if isa(ps.est, Function)
        return ps
    end
    return PipelineStep(update_online_step(ps.est), ps.reads, ps.writes, ps.target)
end
"""
    update_online_estimator(p::Pipeline)

Resolves the [`Online`](@ref) declarations in the steps of a [`Pipeline`](@ref), at warm-up.

On the host route [`update_online_step`](@ref) resolves every step. So an `Online(EmpiricalPrior(); max_history = w)` step becomes the prior with its buffer, and an optimisation step resolves the wrappers under its own prior. On the refit route the pipeline carries a [`PipelineBufferState`](@ref), which `Online(pipe)` seeded, and the method returns the pipeline unchanged. No member folds under a refit, and [`assert_online_entry`](@ref) already refused any wrapper below the root by name.

# Related

  - [`update_online_estimator`](@ref)
  - [`update_online_step`](@ref)
  - [`Online`](@ref)
"""
function update_online_estimator(p::Pipeline)
    if isa(p.cache, PipelineBufferState)
        return p
    end
    return Pipeline(p.names, map(update_online_step, p.steps), p.cache)
end
"""
    copy_states(p::Pipeline)

Copies every partial-fit state that a [`Pipeline`](@ref) carries, and rebuilds the pipeline around the copies, for [`Resume`](@ref).

The states are the own `cache` of the pipeline and the states under every step, which [`copy_step_states`](@ref) copies. A fold of the copy leaves the held rows of the original unchanged.

# Related

  - [`copy_states`](@ref)
  - [`copy_step_states`](@ref)
  - [`Resume`](@ref)
"""
function copy_states(p::Pipeline)
    return Pipeline(p.names, map(copy_step_states, p.steps), copy_state(p.cache))
end
"""
    copy_step_states(step)

Copies the partial-fit states under one [`Pipeline`](@ref) step.

The method rebuilds a [`PipelineStep`](@ref) around its copied estimator, and copies an estimator step through [`copy_states`](@ref). A callable and a result pass through unchanged.

# Related

  - [`copy_states(p::Pipeline)`](@ref)
  - [`update_online_step`](@ref)
"""
copy_step_states(step) = step
function copy_step_states(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    return copy_states(est)
end
function copy_step_states(ps::PipelineStep)
    if isa(ps.est, Function)
        return ps
    end
    return PipelineStep(copy_step_states(ps.est), ps.reads, ps.writes, ps.target)
end
"""
    held_timestamps(p::Pipeline)

Returns the timestamps that a stepped [`Pipeline`](@ref) holds, through the state of its row owner, or `nothing`.

The state that the pipeline carries selects the answer.

  - Under `Online(pipe)` the [`PipelineBufferState`](@ref) holds the input carrier, and the timestamps of that carrier are the answer.
  - A prior owner leaves the Pipeline its own [`ReturnsBufferState`](@ref), which holds them.
  - An optimisation owner keeps its own Fold Context, and the pipeline holds none, so the owner answers ([`held_timestamps`](@ref)).
  - A pipeline whose owner carries no state, or that has no owner, has taken no step, and answers `nothing`.

At the price level the answer of the host route starts one row after the answer of the refit route. A `PricesToReturns` step drops the first price row, so the returns carry the timestamps of the prices from the second row on. Both answers are a run of the timestamps of the price carrier that ends at the last row folded, which is what the alignment check of [`Resume`](@ref) reads.

# Related

  - [`held_timestamps`](@ref)
  - [`pipeline_held_timestamps`](@ref)
  - [`Resume`](@ref)
"""
function held_timestamps(p::Pipeline)
    return pipeline_held_timestamps(p, p.cache)
end
"""
    pipeline_held_timestamps(p::Pipeline, cache::PipelineBufferState)
    pipeline_held_timestamps(p::Pipeline, cache::ReturnsBufferState)
    pipeline_held_timestamps(p::Pipeline, ::Nothing)

The three arms of [`held_timestamps(p::Pipeline)`](@ref), selected by dispatch on the state that the pipeline carries.

A buffer of the input carrier answers with the timestamps of that carrier, and a Fold Context answers with its own. With no state, the row owner ([`pipeline_row_owner`](@ref)) answers when it carries a state ([`online_entry_state`](@ref)), and the method answers `nothing` otherwise.

# Related

  - [`held_timestamps(p::Pipeline)`](@ref)
  - [`pipeline_row_owner`](@ref)
"""
function pipeline_held_timestamps(::Pipeline, cache::PipelineBufferState)
    return isnothing(cache.data) ? nothing : carrier_timestamps(cache.data)
end
function pipeline_held_timestamps(::Pipeline, cache::ReturnsBufferState)
    return cache.ts
end
function pipeline_held_timestamps(p::Pipeline, ::Nothing)
    k = pipeline_row_owner(p)
    owner = iszero(k) ? nothing : step_estimator(p.steps[k])
    return isnothing(online_entry_state(owner)) ? nothing : held_timestamps(owner)
end
"""
    PipelineResume = Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any, <:Pipeline}}

Alias for a [`Resume`](@ref) whose Result carries a [`Pipeline`](@ref).

The pipeline doors dispatch on it, so a resumed pipeline takes the pipeline's own entry checks and fold loop, and not those of an optimiser.

# Related

  - [`Resume`](@ref)
  - [`cross_val_predict(r::PipelineResume, data::Prices_RR, cv::CVER)`](@ref)
"""
const PipelineResume = Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any,
                                                            <:Pipeline}}
"""
    cross_val_predict(r::PipelineResume, data::Prices_RR, cv::CVER; ex = FLoops.ThreadedEx(), id = nothing)

Continues an online walk-forward over a [`Pipeline`](@ref) from its Result, over the full history extended.

This is the pipeline door of [`Resume`](@ref). The door checks the scheme through [`assert_resume_scheme`](@ref), and it refuses a holdout as the one-shot door does. The fold loop then takes its resumed arm through [`pipeline_cross_val_predict`](@ref). The host route and the refit route of `Online(pipe)` resume alike, because the entry reads the state off the pipeline through walks that take any estimator.

# Validation

  - Everything [`assert_pipeline_door`](@ref) refuses for a `Resume`.
  - Everything the resumed arm of the fold loop refuses ([`Resume`](@ref)).

# Related

  - [`Resume`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
  - [`assert_pipeline_door`](@ref)
"""
function cross_val_predict(r::PipelineResume, data::Prices_RR, cv::CVER;
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                           id = nothing)
    return pipeline_cross_val_predict(r, data, cv; ex = ex, id = id)
end
function assert_pipeline_door(r::PipelineResume, cv)
    assert_no_holdout(r.res.opt)
    assert_resume_scheme(cv)
    return nothing
end
"""
    partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)

Folds a block of observations into a [`Pipeline`](@ref), without a fit.

This is the online step of the Pipeline, and the pipeline is a host. The verb hands each step before the **row owner** the block that the step before it emitted, and folds the rows into the owner, which is the prior step, else the optimisation step ([`pipeline_row_owner`](@ref)). The verb leaves every step after the owner untouched, and the read-out fits it as batch fits it. Under `Online(pipe)` the pipeline carries a [`PipelineBufferState`](@ref) instead. The verb then appends the block to the buffer, no step folds, and the read-out is a batch fit over the buffer.

The arity mirrors the batch verb. `fit(pipe, data)` takes a carrier at the input level of the pipeline, so the online step takes one too, with one observation or a block of them.

# Algorithm

On the host route:

 1. Find the row owner, giving `k`.
 2. Walk the steps before `k` in order. Skip each step that writes no data slot ([`is_data_step`](@ref)) and each universe-only step ([`is_universe_step`](@ref)). A universe-only step passes the rows through untouched, and the read-out applies its universe.
 3. Fold each remaining step through [`partial_fit_transform`](@ref), giving the folded step and the next `block`.
 4. Fold `block` into the owner through [`fold_pipeline_owner`](@ref), giving the folded owner and `cache`. A prior owner folds through [`fold_prior`](@ref), and `cache` is a [`ReturnsBufferState`](@ref) of the Pipeline that records the rest of the carrier, so that `fit(pipe)` can rebuild the carrier that the batch path reads. An optimisation owner folds through its own [`partial_fit!`](@ref) and keeps its own Fold Context, so `cache` is `nothing`.
 5. Return the pipeline rebuilt from the folded steps and `cache`.

On the refit route, append `data` to the buffer through [`partial_fit!`](@ref), and return the pipeline with the new state.

# Arguments

  - `pipe`: The pipeline to fold into, with its wrappers resolved by [`update_online_estimator`](@ref).
  - `data`: The block of observations, at the price level or at the returns level, as the input of the pipeline.

# Validation

On the host route:

  - The pipeline has a row owner. An `ArgumentError` is thrown otherwise.
  - No step holds an [`Online`](@ref) declaration ([`pipeline_online_member`](@ref)), because the warm-up resolves each one. A wrapper that no warm-up resolved would fold the estimator that it wraps with no buffer, and its cap would have no effect. An `ArgumentError` that names the step is thrown otherwise.
  - The rows reach the owner as a [`ReturnsResult`](@ref). An `ArgumentError` is thrown otherwise.
  - Everything the folds of the steps refuse.

# Returns

  - `pipe`: The pipeline, with the owner and every step before it folded, and its own state recorded.

# Related

  - [`fit(pipe::Pipeline)`](@ref)
  - [`partial_fit_transform`](@ref)
  - [`fold_prior`](@ref)
  - [`ReturnsBufferState`](@ref)
  - [`PipelineBufferState`](@ref)
  - [`update_online_estimator(p::Pipeline)`](@ref)
"""
function partial_fit!(pipe::Pipeline{<:Any, <:Any,
                                     <:Option{<:Union{<:PipelineBufferState,
                                                      <:ReturnsBufferState}}},
                      data::Prices_RR)
    return fold_pipeline(pipe, pipe.cache, data)
end
"""
    fold_pipeline(pipe::Pipeline, cache::PipelineBufferState, data::Prices_RR)
    fold_pipeline(pipe::Pipeline, cache::Option{<:ReturnsBufferState}, data::Prices_RR)

The two routes of [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref), selected by dispatch on the state that the Pipeline carries.

A buffer of the input carrier appends the block. A Fold Context, or no state, walks the steps. The steps and the refusals are those of the verb.

# Related

  - [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref)
  - [`PipelineBufferState`](@ref)
  - [`ReturnsBufferState`](@ref)
"""
function fold_pipeline(pipe::Pipeline, cache::PipelineBufferState, data::Prices_RR)
    return Pipeline(pipe.names, pipe.steps, partial_fit!(cache, data))
end
function fold_pipeline(pipe::Pipeline, cache::Option{<:ReturnsBufferState}, data::Prices_RR)
    k = pipeline_row_owner(pipe)
    @argcheck(k > 0,
              ArgumentError("a `Pipeline` with neither a prior step nor an optimisation step has no row owner, so the online step has nothing to fold the observations into."))
    member = pipeline_online_member(pipe)
    @argcheck(isnothing(member),
              ArgumentError("the `$(member)` step holds an `Online` declaration that no warm-up resolved, and `partial_fit!(pipe, data)` would fold the estimator it wraps with no buffer, so its `max_history` would be ignored. Run the pipeline through an Online Scheme (`OnlineIndexWalkForward` or `OnlineDateWalkForward`), whose warm-up resolves the declaration, or hand the step the plain estimator."))
    steps = collect(Any, pipe.steps)
    block = data
    for i in 1:(k - 1)
        step = steps[i]
        if !is_data_step(step)
            continue
        end
        est = step_estimator(step)
        if is_universe_step(est)
            continue
        end
        est, block = partial_fit_transform(est, block)
        steps[i] = rewrap_step(step, est)
    end
    @argcheck(isa(block, ReturnsResult),
              ArgumentError("the rows reach the `$(pipe.names[k])` step, the row owner, as a `$(typeof(block).name.name)`, and a prior or an optimiser folds a `ReturnsResult`: put a `PricesToReturns` step before it."))
    owner = step_estimator(steps[k])
    owner, cache = fold_pipeline_owner(owner, cache, block)
    steps[k] = rewrap_step(steps[k], owner)
    return Pipeline(pipe.names, Tuple(steps), cache)
end
"""
    fold_pipeline_owner(pe::AbstractPriorEstimator, cache, rd::ReturnsResult)
    fold_pipeline_owner(opt::OptimisationEstimator, cache, rd::ReturnsResult)
    fold_pipeline_owner(opt::OnlinePortfolioSelection, cache, rd::ReturnsResult)

Folds a block into the row owner of a [`Pipeline`](@ref), and records the own context of the Pipeline where the owner keeps none.

An optimisation owner is a host itself. It folds through its own [`partial_fit!`](@ref), and the Pipeline records nothing beside it. The method refuses an [`OnlinePortfolioSelection`](@ref) head. This is the door of a hand-driven run, as [`assert_online_owner`](@ref) is the door of the fold loop, and that docstring states the reason.

# Algorithm

For a prior owner:

 1. Fold `rd` into the prior through [`fold_prior`](@ref), giving `pe`. This goes first, because the context takes its cap from the buffer that the prior seeds, as the context of an optimiser does ([`fold_returns`](@ref)).
 2. Read the buffer of the prior through [`prior_returns_buffer`](@ref), giving `rows`.
 3. Set `own_factors` when [`needs_factor_returns`](@ref) answers `false` for `pe`. The context keeps the factor columns only then, so the carrier holds each factor column once.
 4. Fold `rd` into the context of the Pipeline through [`fold_context`](@ref) under the cap `rows.max_history`, giving `cache`.
 5. Return `pe` and `cache`.

# Validation

  - The owner is not an [`OnlinePortfolioSelection`](@ref) head. An `ArgumentError` is thrown otherwise.

# Related

  - [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref)
  - [`assert_online_owner`](@ref)
  - [`fold_prior`](@ref)
  - [`fold_context`](@ref)
"""
function fold_pipeline_owner(pe::AbstractPriorEstimator,
                             cache::Option{<:ReturnsBufferState}, rd::ReturnsResult)
    pe = fold_prior(pe, rd)
    rows = prior_returns_buffer(pe)
    own_factors = needs_factor_returns(pe) === false
    return pe, fold_context(cache, rd, rows.max_history, false, own_factors)
end
function fold_pipeline_owner(opt::OptimisationEstimator, ::Nothing, rd::ReturnsResult)
    return partial_fit!(opt, rd), nothing
end
function fold_pipeline_owner(::OnlinePortfolioSelection, ::Nothing, ::ReturnsResult)
    return throw(ArgumentError("an `OnlinePortfolioSelection` head owns the rows of this `Pipeline`, because no prior step precedes it, and a head cannot own them: its read-out is its own recursion, so `fit(pipe)` is left with no carrier to reconstitute the universe steps over, and a view of a recursion is not the recursion over the columns a universe step keeps. Put a prior step before the head, whose rows the read-out reads, or declare a refit with `Online(pipe)`, which fits every step over the observations folded so far. The fold loop names the same two routes at its own door."))
end
"""
    fit(pipe::Pipeline)

Reads a folded [`Pipeline`](@ref) out, with the result that the batch fit over the observations folded so far gives.

This is the read-out verb of the online step of the Pipeline, as `optimise(opt)` is for an optimiser. It returns an ordinary [`PipelineResult`](@ref), so [`predict`](@ref), [`assert_universe_aligned`](@ref) and the search take it unchanged.

# Algorithm

 1. Rebuild the carrier of the observations folded so far from the row owner through [`pipeline_returns_result`](@ref), giving `rd₀`. A prior owner gives its rows through [`prior_returns_buffer`](@ref), and the Pipeline adds the rest from its own [`ReturnsBufferState`](@ref). An optimisation owner gives its carrier through [`returns_result`](@ref). The carrier spans the input universe of the pipeline at warm-up width.
 2. Walk the data steps before the owner in order, through [`readout_data_step`](@ref). A row-local step reads its fitted Result out of its state, restricted to the assets that survive so far. A universe-only step runs its batch verb over `rd₀` viewed to the surviving assets, and narrows the surviving set. The index `idx` of the surviving assets into `rd₀` is the column map, and `rd = port_opt_view(rd₀, :, idx)` is the context's returns.
 3. Run every other step before the owner over that context, as batch runs it. Such a step is a phylogeny step, an uncertainty-set step or a constraint step.
 4. Read the owner out over the surviving assets through [`readout_owner`](@ref). A prior owner reads out through `prior(port_opt_view(pe, idx))`, which views its state by asset and never slices its rows again. An optimisation owner reads out through `optimise(opt)` on the viewed estimator, with the context injected.
 5. Run every step after the owner as batch, the optimisation step with the context injected, and return the result.

So a selection that moves between two steps is a view of a state fitted over the whole universe, which is the batch fit over those columns. The selector ranks the assets again at every step, as the batch loop refits it at every fold.

Under `Online(pipe)` the read-out is `fit(pipe, buffer)`, the batch fit over the observations that the [`PipelineBufferState`](@ref) holds.

# Validation

  - The pipeline has a row owner. An `ArgumentError` is thrown otherwise.
  - The pipeline has taken a step. An `ArgumentError` is thrown otherwise.
  - A prior owner reads out over as many assets as survive the universe steps before it ([`readout_owner`](@ref)). An `ArgumentError` is thrown otherwise.

# Returns

  - `res::PipelineResult`: The result the batch fit over the observations folded so far gives.

# Related

  - [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref)
  - [`readout_data_step`](@ref)
  - [`returns_result`](@ref)
  - [`fit`](@ref)
"""
function StatsAPI.fit(pipe::Pipeline)::PipelineResult
    return readout_pipeline(pipe, pipe.cache)
end
"""
    readout_pipeline(pipe::Pipeline, cache::PipelineBufferState)
    readout_pipeline(pipe::Pipeline, cache::Option{<:ReturnsBufferState})

The two routes of [`fit(pipe::Pipeline)`](@ref), selected by dispatch on the state that the Pipeline carries.

A buffer of the input carrier takes a batch fit. A Fold Context, or no state, takes the walk of the read-out. The steps and the refusals are those of the verb.

# Related

  - [`fit(pipe::Pipeline)`](@ref)
  - [`PipelineBufferState`](@ref)
  - [`ReturnsBufferState`](@ref)
"""
function readout_pipeline(pipe::Pipeline, cache::PipelineBufferState)
    @argcheck(!isnothing(cache.data),
              ArgumentError("`fit(pipe)` with no data reads the buffer `Online(pipe)` seeded, and this pipeline has folded nothing into it. Fold observations with `partial_fit!(pipe, data)` first, or pass the data to `fit(pipe, data)`."))
    return StatsAPI.fit(Pipeline(pipe.names, pipe.steps), cache.data)
end
function readout_pipeline(pipe::Pipeline, cache::Option{<:ReturnsBufferState})
    k = pipeline_row_owner(pipe)
    @argcheck(k > 0,
              ArgumentError("a `Pipeline` with neither a prior step nor an optimisation step has no row owner, and so no state to read out."))
    owner = step_estimator(pipe.steps[k])
    rd0 = pipeline_returns_result(owner, cache, pipe.names[k])
    n = length(pipe.steps)
    fitted = Vector{Any}(undef, n)
    idx = collect(1:length(rd0.nx))
    for i in 1:(k - 1)
        step = pipe.steps[i]
        if !is_data_step(step)
            continue
        end
        fitted[i], idx = readout_data_step(step_estimator(step), rd0, idx)
    end
    rd = port_opt_view(rd0, :, idx)
    ctx = PipelineContext(; returns = rd)
    for i in 1:(k - 1)
        step = pipe.steps[i]
        if is_data_step(step)
            continue
        end
        fitted[i], ctx = run_step(maybe_inject_step(step, ctx), ctx)
    end
    fitted[k], ctx = readout_owner(owner, idx, length(rd0.nx), ctx)
    for i in (k + 1):n
        fitted[i], ctx = run_step(maybe_inject_step(pipe.steps[i], ctx), ctx)
    end
    return PipelineResult(pipe.names, Tuple(fitted), ctx)
end
"""
    pipeline_returns_result(pe::AbstractPriorEstimator, cache::ReturnsBufferState, name)
    pipeline_returns_result(opt::OptimisationEstimator, ::Nothing, name)
    pipeline_returns_result(owner, ::Nothing, name)

Rebuilds the [`ReturnsResult`](@ref) of the observations that a [`Pipeline`](@ref) folded, from its row owner.

A prior owner rebuilds it from the Fold Context of the Pipeline and the buffer of the prior. An optimisation owner rebuilds it from its own Fold Context ([`returns_result`](@ref)).

# Validation

  - The pipeline has taken a step, so it carries a Fold Context or its optimisation owner carries a state. An `ArgumentError` that names the owner is thrown otherwise.

# Related

  - [`fit(pipe::Pipeline)`](@ref)
  - [`returns_result`](@ref)
  - [`prior_returns_buffer`](@ref)
"""
function pipeline_returns_result(pe::AbstractPriorEstimator, cache::ReturnsBufferState,
                                 ::AbstractString)
    return returns_result(cache, prior_returns_buffer(pe))
end
function pipeline_returns_result(opt::OptimisationEstimator, ::Nothing,
                                 name::AbstractString)
    @argcheck(!isnothing(online_entry_state(opt)),
              ArgumentError("`fit(pipe)` with no data reads the state the online step wrote, and the `$(name)` step, the row owner, has taken no step. Fold observations with `partial_fit!(pipe, data)` first, or pass the data to `fit(pipe, data)`."))
    return returns_result(opt)
end
function pipeline_returns_result(::Any, ::Nothing, name::AbstractString)
    return throw(ArgumentError("`fit(pipe)` with no data reads the state the online step wrote, and this pipeline has taken no step: its `cache` is `nothing` and the `$(name)` step, the row owner, carries none. Fold observations with `partial_fit!(pipe, data)` first, or pass the data to `fit(pipe, data)`."))
end
"""
    readout_data_step(est, rd0::ReturnsResult, idx::AbstractVector{<:Integer}) -> (fitted, idx′)

Reads one data step of a folded [`Pipeline`](@ref) out, and narrows the surviving assets where the universe of the step does.

Dispatch on the estimator of the step selects the read-out.

  - A [`PriceGapFill`](@ref) reads the [`PriceGapFillResult`](@ref) of the whole history out of its state, restricted to the assets that survive so far, because the batch fit ran over the columns that an earlier filter left.
  - A [`MissingDataFilter`](@ref) reads its [`MissingDataFilterResult`](@ref) out the same way, and narrows the surviving assets to the ones that it keeps, in the order that they hold.
  - An [`AbstractAssetSelector`](@ref) runs its batch verb over the rows of the owner, viewed to the surviving assets. It narrows them to the fitted universe, in the fitted order, as [`apply_preprocessing`](@ref) orders it.
  - Any other row-local step, such as a [`PricesToReturns`](@ref), reads out its own state through [`fit_preprocessing`](@ref) and narrows nothing.

# Arguments

  - `est`: The step's estimator, carrying its state.
  - `rd0`: The carrier of every observation folded, over the pipeline's input universe.
  - `idx`: The indices into `rd0` of the assets surviving the steps before this one.

# Returns

  - `(fitted, idx′)`: The step's fitted Result, and the surviving indices after it.

# Related

  - [`fit(pipe::Pipeline)`](@ref)
  - [`fit_preprocessing`](@ref)
  - [`is_universe_step`](@ref)
"""
function readout_data_step(est::AbstractPreprocessingEstimator, ::ReturnsResult,
                           idx::AbstractVector{<:Integer})
    return fit_preprocessing(est), idx
end
function readout_data_step(est::PriceGapFill, rd0::ReturnsResult,
                           idx::AbstractVector{<:Integer})
    res = fit_preprocessing(est)
    cur = rd0.nx[idx]
    keep = findall(n -> string(n) in cur, res.nx)
    return PriceGapFillResult(res.nx[keep], res.v[keep], res.te, res.fill, res.strict), idx
end
function readout_data_step(est::MissingDataFilter, rd0::ReturnsResult,
                           idx::AbstractVector{<:Integer})
    res = fit_preprocessing(est)
    cur = rd0.nx[idx]
    keep = findall(n -> string(n) in cur, res.nx)
    names = string.(res.nx[keep])
    sel = findall(in(names), cur)
    return MissingDataFilterResult(res.nx[keep], res.row_thr), idx[sel]
end
function readout_data_step(sel::AbstractAssetSelector, rd0::ReturnsResult,
                           idx::AbstractVector{<:Integer})
    rd = port_opt_view(rd0, :, idx)
    res = fit_preprocessing(sel, rd)
    pos = [findfirst(==(name), rd.nx) for name in res.nx]
    return res, idx[pos]
end
"""
    readout_owner(pe::AbstractPriorEstimator, idx, n::Integer, ctx::PipelineContext)
    readout_owner(opt::OptimisationEstimator, idx, n::Integer, ctx::PipelineContext)

Reads the row owner of a folded [`Pipeline`](@ref) out over the surviving assets, and writes its slot.

The method views the state of the owner to the surviving assets through [`view_owner`](@ref) before the read-out. The view is a slice by asset. A state fitted over the full universe, viewed to a set of columns and read out, is the batch fit over that set. A prior owner reads out through `prior(pe)`. The method injects the context into an optimisation owner and reads it out through `optimise(opt)`, so the clustering, the constraints and every uncertainty set come from the rebuilt carrier, as batch fits them.

# Validation

  - A prior owner reads out over as many assets as survive. Otherwise its state has no asset view, and the method throws an `ArgumentError` that asks for the universe steps after the prior.

# Related

  - [`fit(pipe::Pipeline)`](@ref)
  - [`prior`](@ref)
  - [`optimise`](@ref)
  - [`inject_context`](@ref)
"""
function readout_owner(pe::AbstractPriorEstimator, idx, n::Integer, ctx::PipelineContext)
    pe = view_owner(pe, idx, n, ctx)
    pr = prior(pe)
    @argcheck(length(pr.mu) == length(idx),
              ArgumentError("the `$(typeof(pe).name.name)` owning the rows read out over $(length(pr.mu)) assets where $(length(idx)) survive the universe steps before it: its state has no asset view, so a selection that moves between two steps cannot be expressed as a view of it. Put the universe steps after the prior, or hold a prior whose state `port_opt_view` slices."))
    return pr, set_slot(ctx, :prior, pr)
end
function readout_owner(opt::OptimisationEstimator, idx, n::Integer, ctx::PipelineContext)
    opt = inject_context(view_owner(opt, idx, n, ctx), ctx)
    res = optimise(opt)
    return res, set_slot(ctx, :opt, res)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Views the row owner's state to the surviving assets, and leaves it alone when every one of the `n` assets survives in its own order.

# Related

  - [`readout_owner`](@ref)
  - [`port_opt_view`](@ref)
"""
function view_owner(est, idx, n::Integer, ctx::PipelineContext)
    if idx == 1:n
        return est
    end
    return port_opt_view(est, idx, ctx.returns.X)
end
"""
    pipeline_fold_fit(pipe::Pipeline, data::Prices_RR, train_idx::VecInt, cols = :)
    pipeline_fold_fit(pipe::Pipeline, data::Prices_RR, ::Nothing, cols = :)

The fit of one fold of a [`Pipeline`](@ref), by whether the fold carries a training window.

This is the pipeline form of [`fit_fold_result`](@ref). A window fits the workflow over it, `fit(pipe, data[train_idx])`, which is the refit that every fold of the batch arms runs. `nothing` means that the pipeline holds its window, because the online arm of [`fold_loop`](@ref) folded every row of it. The fold then reads the pipeline out through `fit(pipe)` with no data.

# Related

  - [`fit_fold_result`](@ref)
  - [`fit(pipe::Pipeline)`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
"""
function pipeline_fold_fit(pipe::Pipeline, data::Prices_RR, train_idx::VecInt, cols = :)
    return StatsAPI.fit(pipe, pipeline_data_view(data, train_idx, cols))
end
function pipeline_fold_fit(pipe::Pipeline, ::Prices_RR, ::Nothing, ::Any = :)
    return StatsAPI.fit(pipe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Lists the fields that `show` renders for a [`Pipeline`](@ref): the names, the steps, and the `cache` only when it holds a state.

A pipeline that took no step therefore renders no `cache` line.

# Related

  - [`show_fields`](@ref)
"""
function show_fields(p::Pipeline)
    return isnothing(p.cache) ? (:names, :steps) : (:names, :steps, :cache)
end
