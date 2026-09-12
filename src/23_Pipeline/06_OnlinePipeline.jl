"""
    pipe_writes(o::Online) = pipe_writes(o.est)
    pipe_reads(o::Online) = pipe_reads(o.est)

An [`Online`](@ref) step writes and reads the slots of the estimator it wraps: `Online(EmpiricalPrior(); max_history = w)` is a prior step whose window is capped, and it resolves to a plain prior at the warm-up of the online arm.

# Related

  - [`Online`](@ref)
  - [`pipe_writes`](@ref)
  - [`update_online_estimator(p::Pipeline)`](@ref)
"""
pipe_writes(o::Online) = pipe_writes(o.est)
pipe_reads(o::Online) = pipe_reads(o.est)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an [`Online`](@ref) step that reached a fold-less fit, by name.

A wrapper is a declaration the online arm's warm-up resolves; `fit(pipe, data)` has no warm-up, so a wrapper that reaches it has no buffer to seed and no step to fold, and is refused rather than fitted as the plain estimator it wraps, which would read a batch answer under an online declaration.

# Related

  - [`run_step`](@ref)
  - [`Online`](@ref)
  - [`update_online_estimator`](@ref)
"""
function run_step(o::Online, ::PipelineContext)
    return throw(ArgumentError("an `Online($(typeof(o.est).name.name))` step is a declaration the online arm of the fold loop resolves at its warm-up, and `fit(pipe, data)` has none: run the pipeline through a scheme with `ff = OnlineStep()`, or hand the step the plain estimator."))
end
"""
$(DocStringExtensions.TYPEDEF)

The input-carrier buffer `Online(pipe)` seeds: every block of observations a [`Pipeline`](@ref) is handed, concatenated, so the read-out is the batch fit over them.

The declared refit of ADR 0142. No step folds under it — the buffer holds the pipeline's input as it was given, price- or returns-level, and `fit(pipe)` runs `fit(pipe, data)` over the buffer — so it is exact for every configuration at batch cost, and with a cap it is a rolling Pipeline, equal to the rolling batch walk-forward. The state's type is the route: a [`ReturnsBufferState`](@ref) in `pipe.cache` is the host route's Fold Context, and this is the refit route.

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

Appends a block of observations to a [`PipelineBufferState`](@ref), and drops the oldest past the cap.

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

Rebuilds a [`Pipeline`](@ref) with the fields in `repl` replaced, through the positional constructor: the names were built once from the steps, and a rebuild that seeds a state keeps them as they are, which the keyword constructor cannot express.

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

The estimator a [`Pipeline`](@ref) step stands for, unwrapped from a [`PipelineStep`](@ref) and from an [`Online`](@ref) declaration, for the classification the online step makes. A callable step is returned as it is.

# Related

  - [`rewrap_step`](@ref)
  - [`pipeline_row_owner`](@ref)
"""
step_estimator(step) = step
step_estimator(ps::PipelineStep) = step_estimator(ps.est)
step_estimator(o::Online) = step_estimator(o.est)
"""
    rewrap_step(step, est)

Puts a folded estimator back in the wrapper its step came in: a [`PipelineStep`](@ref) keeps its reads, writes and target, and a bare step is the estimator itself.

# Related

  - [`step_estimator`](@ref)
"""
rewrap_step(::Any, est) = est
rewrap_step(ps::PipelineStep, est) = PipelineStep(est, ps.reads, ps.writes, ps.target)
"""
    is_data_step(step) -> Bool

Answers whether a [`Pipeline`](@ref) step writes a data slot, `:prices` or `:returns`, and so changes the rows the row owner folds.

# Related

  - [`pipeline_row_owner`](@ref)
  - [`assert_online_entry(p::Pipeline)`](@ref)
"""
is_data_step(step) = pipe_writes(step) in PIPELINE_DATA_SLOTS
"""
    is_universe_step(est) -> Bool

Answers whether a data step is universe-only: it folds nothing, and at the read-out its batch verb runs over the row owner's rows and its universe is applied as a view. The asset selectors are; every other step answers `false`.

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref)
"""
is_universe_step(::Any) = false
is_universe_step(::AbstractAssetSelector) = true
"""
    is_row_owner(est) -> Bool

Answers whether a step can own the rows of a [`Pipeline`](@ref)'s online step: a prior estimator, or an optimisation step of any form, a schedule included, so that the walk names the one it meets and the refusals below state why a schedule cannot fold.

# Related

  - [`pipeline_row_owner`](@ref)
"""
is_row_owner(::Any) = false
is_row_owner(::AbstractPriorEstimator) = true
is_row_owner(::Union{<:OptE_Opt, <:TD_OptE_Opt}) = true
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finds the row owner of a [`Pipeline`](@ref) by a walk: the first prior step, else the optimisation step, else `0`.

The prior step owns the rows when there is one, because [`inject_context`](@ref) overrides the optimiser's `pe` with it and the optimiser's own prior is never fitted; else the optimisation step, which is then a host of ADR 0137 and keeps its own Fold Context; and a pipeline with neither has nothing to fold into.

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

Names the first state a [`Pipeline`](@ref) carries at the entry of the fold loop's online arm — its own `cache`, or a state anywhere under a step, prefixed by the step's name — or answers `nothing`.

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

Names the first [`Online`](@ref) declaration a [`Pipeline`](@ref)'s steps carry, prefixed by the step's name, or answers `nothing`.

The walk the refit route runs: a wrapper below an `Online(pipe)` has no fold to seed, because no member folds under the refit, and its cap would be silently ignored, so it is refused by name.

# Related

  - [`assert_online_entry(o::Online{<:Pipeline})`](@ref)
  - [`online_fields`](@ref)
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

Names the first [`Online`](@ref) declaration one [`Pipeline`](@ref) step carries, relative to the step: `""` when the step is itself a wrapper, the field's dotted path when a field of its estimator holds one, `nothing` otherwise. A JuMP or hierarchical head is scanned through the bundle it holds, as [`update_online_estimator`](@ref) resolves it.

# Related

  - [`pipeline_online_member`](@ref)
  - [`online_fields`](@ref)
"""
step_online_member(::Any) = nothing
step_online_member(::Online) = ""
step_online_member(ps::PipelineStep) = step_online_member(ps.est)
function step_online_member(est::Union{<:AbstractEstimator,
                                       <:StatsBase.CovarianceEstimator})
    fns = online_fields(est)
    return isempty(fns) ? nothing : string(fns[1])
end
function step_online_member(est::Union{<:JuMPOptimisationEstimator,
                                       <:HierarchicalRiskParity,
                                       <:HierarchicalEqualRiskContribution,
                                       <:SchurComplementHierarchicalRiskParity})
    inner = step_online_member(est.opt)
    return isnothing(inner) ? nothing : string("opt.", inner)
end
"""
    assert_online_entry(p::Pipeline)
    assert_online_entry(o::Online{<:Pipeline})

Refuses a [`Pipeline`](@ref) that cannot take the online step, at the entry of the fold loop's online arm and before any fit, by name.

The walk over the steps ADR 0142 decides. On the host route, five refusals. A state anywhere — the Pipeline's own `cache` or a step's — because the loop starts cold (ADR 0140). No row owner: a pipeline with neither a prior nor an optimisation step has nothing to fold into. A [`TimeDependent`](@ref) schedule as the row owner, which is the case only when no prior step precedes the optimisation step: a schedule swaps the estimator that carries the state; with a prior step before it, the schedule swaps a stateless step and composes. An optimisation step that owns the rows is held to [`assert_online_entry`](@ref)'s own refusals. And every data step before the owner must fold or defer: a **window-valued** configuration ([`PriceGapFill`](@ref) with a statistic fill, [`MissingDataFilter`](@ref) with `row_thr < 1`), a callable [`PipelineStep`](@ref) writing `:prices` or `:returns`, a nested `Pipeline`, and a caller's preprocessing estimator with no online form are refused, and the message names both routes: give the step a [`partial_fit_transform`](@ref), or declare a refit with `Online(pipe)`.

On the refit route, two: a state anywhere, as above, and an [`Online`](@ref) member below the `Online(pipe)`, because no member folds under a refit and its cap would be silently ignored.

# Validation

  - Everything above. An `ArgumentError` naming the step is thrown otherwise.

# Related

  - [`online_folds`](@ref)
  - [`online_entry_state`](@ref)
  - [`pipeline_row_owner`](@ref)
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

Refuses a row owner that cannot fold, by name: a [`TimeDependent`](@ref) schedule, and an optimisation step [`assert_online_entry`](@ref) refuses. A precomputed result never reaches the walk, because a result is not a step: it enters a pipeline only as a schedule's entry, and the schedule is refused first.

# Related

  - [`assert_online_entry(p::Pipeline)`](@ref)
"""
function assert_online_owner(::AbstractPriorEstimator, ::AbstractString)
    return nothing
end
function assert_online_owner(::TimeDependent, name::AbstractString)
    return throw(ArgumentError("the `$(name)` step is a `TimeDependent` schedule of optimisers and owns the rows of the online step, because no prior step precedes it: a schedule swaps the estimator that carries the state every fold, so the optimiser a fold is handed never saw the rows folded before it. Put a prior step before it, whose state the loop threads while the schedule swaps a stateless step, or schedule a field that carries no state, or refit every fold with `ff = nothing`."))
end
function assert_online_owner(opt::OptimisationEstimator, ::AbstractString)
    return assert_online_entry(opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The entry checks of a Pipeline's cross-validation door: no holdout, and an `Online(pipe)` root only under a scheme that declares a Fold Fit, because the wrapper resolves at the online arm's warm-up and a batch scheme has none.

# Related

  - [`assert_no_holdout`](@ref)
  - [`fold_fit`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
"""
function assert_pipeline_door(pipe::Pipeline, ::Any)
    assert_no_holdout(pipe)
    return nothing
end
function assert_pipeline_door(o::Online{<:Pipeline}, cv)
    assert_no_holdout(o.est)
    @argcheck(!isnothing(fold_fit(cv)),
              ArgumentError("`Online(pipe)` declares a refit from a buffer the online arm of the fold loop seeds at its warm-up, and this scheme declares no Fold Fit, so every fold refits from its training window already. Set `ff = OnlineStep()` on the scheme, or hand the door the plain pipeline."))
    return nothing
end
"""
    cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::CVER; ex = FLoops.ThreadedEx(), id = nothing)
    cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::MultipleRandomised; ex = FLoops.ThreadedEx(), kwargs...)

Run a walk-forward over `Online(pipe)`, the declared refit of a [`Pipeline`](@ref) from an input-carrier buffer.

The same doors as the pipeline's, with the wrapper threaded through the fold loop: the online arm resolves it at warm-up into a pipeline carrying a [`PipelineBufferState`](@ref), every fold appends its new rows to the buffer, and the read-out is `fit(pipe, buffer)`. So the run is exact for every configuration at batch cost — the window-valued steps the host route refuses included — and with `max_history = w` it equals the rolling batch walk-forward with warm-up `w + purged_size`. A scheme with no Fold Fit is refused by name, because the wrapper resolves only at the online arm's warm-up.

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

The fold loop reads its three traits off the root it is handed, and an `Online(pipe)` root answers for the pipeline it wraps: the wrapper resolves at the online arm's warm-up, and the per-fold copy is made of the pipeline.

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

Resolves the [`Online`](@ref) declarations of one [`Pipeline`](@ref) step at warm-up: a wrapper step becomes the estimator it wraps carrying a seeded buffer, a [`PipelineStep`](@ref) is rebuilt around its resolved estimator, an estimator resolves the wrappers in its own fields, and a callable or a result passes through.

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

Resolves the [`Online`](@ref) declarations a [`Pipeline`](@ref) carries in its steps, at warm-up.

On the host route every step is resolved through [`update_online_step`](@ref), so an `Online(EmpiricalPrior(); max_history = w)` step becomes the prior carrying its buffer, and an optimisation step resolves the wrappers under its own prior. On the refit route — the pipeline carries a [`PipelineBufferState`](@ref), which `Online(pipe)` seeded — no member is seeded, because no member folds under a refit; [`assert_online_entry`](@ref) has refused any wrapper below the root by name already.

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

Copies every partial-fit state a [`Pipeline`](@ref) carries — its own `cache`, and the states under every step through [`copy_step_states`](@ref) — and rebuilds the pipeline around the copies, for [`Resume`](@ref).

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

Copies the partial-fit states under one [`Pipeline`](@ref) step: a [`PipelineStep`](@ref) is rebuilt around its copied estimator, an estimator step is copied through [`copy_states`](@ref), and a callable or a result passes through.

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

The timestamps a stepped [`Pipeline`](@ref) holds, through the state its row owner keeps, or `nothing`.

Three arms, by the state the pipeline carries. Under `Online(pipe)` the [`PipelineBufferState`](@ref) holds the input carrier itself, and its timestamps are the answer. A prior owner leaves the Pipeline its own [`ReturnsBufferState`](@ref), which holds them. An optimisation owner keeps its own Fold Context, and the pipeline holds none, so the owner answers ([`held_timestamps`](@ref)). At the price level a `PricesToReturns` step drops the first row, and the returns' timestamps are the prices' from the second row on, so the held span still equals its rows of the price carrier.

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

The three arms of [`held_timestamps(p::Pipeline)`](@ref), chosen by dispatch on the state the pipeline carries: the input-carrier buffer's timestamps, the Fold Context's, or the optimisation owner's through [`pipeline_row_owner`](@ref).

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
    return iszero(k) ? nothing : held_timestamps(step_estimator(p.steps[k]))
end
"""
    PipelineResume = Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any, <:Pipeline}}

Alias for a [`Resume`](@ref) whose Result carries a [`Pipeline`](@ref): the declaration the pipeline doors take.

# Related

  - [`Resume`](@ref)
  - [`cross_val_predict(r::PipelineResume, data::Prices_RR, cv::CVER)`](@ref)
"""
const PipelineResume = Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any,
                                                            <:Pipeline}}
"""
    cross_val_predict(r::PipelineResume, data::Prices_RR, cv::CVER; ex = FLoops.ThreadedEx(), id = nothing)

Continue an online walk-forward over a [`Pipeline`](@ref) from its Result, over the full history extended.

The pipeline door of [`Resume`](@ref): the scheme is checked ([`assert_resume_scheme`](@ref)), the holdout refused as the one-shot door refuses it, and the fold loop takes its resumed arm through [`pipeline_cross_val_predict`](@ref). A host route and an `Online(pipe)` refit route resume alike, because the entry reads the state off the pipeline generically (ADR 0142).

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

Folds a block of observations into a [`Pipeline`](@ref), without fitting.

The Pipeline's online step, decided by ADR 0142: the pipeline is a host, and the verb walks the steps in order, handing each the block the step before it emitted, until the rows reach the **row owner** — the prior step, else the optimisation step ([`pipeline_row_owner`](@ref)). A row-local step folds and emits through [`partial_fit_transform`](@ref); a universe-only step ([`is_universe_step`](@ref)) passes the rows through untouched, its universe deferred to the read-out; every step after the owner is untouched, and is fitted at the read-out exactly as batch fits it. A prior owner is folded through [`fold_prior`](@ref), and the Pipeline records the rest of the carrier — the benchmark, the timestamps, the pinned names and a static panel — in a [`ReturnsBufferState`](@ref) of its own, taking the owner's cap, so that `fit(pipe)` can rebuild the carrier the batch path reads. An optimisation owner is folded through its own [`partial_fit!`](@ref), and keeps its own Fold Context; the Pipeline then holds none.

Under `Online(pipe)` the pipeline carries a [`PipelineBufferState`](@ref) instead, and the block is appended to it: no step folds, and the read-out is a batch fit over the buffer.

The arity mirrors the batch verb: `fit(pipe, data)` takes a carrier of the pipeline's input level, so the step takes one, holding one observation or a block of them.

# Arguments

  - `pipe`: The pipeline to fold into, its wrappers resolved by [`update_online_estimator`](@ref).
  - `data`: The block of observations, price- or returns-level as the pipeline's input.

# Validation

  - The pipeline has a row owner, and the rows reach it at the returns level. An `ArgumentError` is thrown otherwise.
  - Everything the steps' own folds refuse.

# Returns

  - `pipe`: The pipeline, with every step before the owner and the owner itself folded, and its own context recorded.

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

The two routes of [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref), chosen by dispatch on the state the Pipeline carries: an input-carrier buffer appends the block, and a Fold Context, or none, walks the steps.

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

Folds a block into the row owner of a [`Pipeline`](@ref) and records the Pipeline's own context where the owner keeps none.

A prior owner is folded through [`fold_prior`](@ref), first, because the Pipeline's context takes its cap from the buffer the prior seeds, exactly as an optimiser's does ([`fold_returns`](@ref)). The factor column is owned once, on the same terms: the context keeps it only when the prior's tree never reads it. An optimisation owner is a host of ADR 0137 and folds through its own step, and the Pipeline records nothing beside it.

# Related

  - [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref)
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
"""
    fit(pipe::Pipeline)

Reads a folded [`Pipeline`](@ref) out: reconstitutes the carrier from the row owner's rows, refits every universe step over it, views the owner's state to the surviving assets, and runs the tail as batch.

The read-out verb of the Pipeline's online step, mirroring `optimise(opt)`. It returns an ordinary [`PipelineResult`](@ref), so [`predict`](@ref), [`assert_universe_aligned`](@ref) and the search consume it unchanged.

# Algorithm

 1. Rebuild the carrier of the observations folded so far, `rd₀`, from the row owner: a prior owner's rows through [`prior_returns_buffer`](@ref) and the Pipeline's own [`ReturnsBufferState`](@ref), an optimisation owner's through its [`returns_result`](@ref). The carrier is over the pipeline's input universe at warm-up width.
 2. Walk the data steps before the owner in order, through [`readout_data_step`](@ref). A row-local step reads its fitted Result out of its state, restricted to the assets that survive so far. A universe-only step runs its batch verb over `rd₀` viewed to the surviving assets, and narrows the surviving set. The index `idx` of the surviving assets into `rd₀` is the column map, and `rd = port_opt_view(rd₀, :, idx)` is the context's returns.
 3. Run every other step before the owner — a phylogeny, an uncertainty-set or a constraint step — over that context, as batch runs it.
 4. Read the owner out over the surviving assets: a prior owner through `prior(port_opt_view(pe, idx))`, whose state is viewed by asset (ADR 0107) and never re-sliced; an optimisation owner through `optimise(opt)` on the viewed and injected estimator.
 5. Run every step after the owner as batch, the optimisation step with the context injected, and return the result.

So a selection that moves between two steps is expressed as a view of a state fitted over the whole universe, which is the batch fit over those columns, and the selector re-ranks per step exactly as the batch loop refits it per fold.

Under `Online(pipe)` the read-out is `fit(pipe, buffer)`, the batch fit over the observations the [`PipelineBufferState`](@ref) holds.

# Validation

  - The pipeline has taken a step. An `ArgumentError` is thrown otherwise.

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

The two routes of [`fit(pipe::Pipeline)`](@ref), chosen by dispatch on the state the Pipeline carries: an input-carrier buffer is fitted in batch, and a Fold Context, or none, is read out through the walk.

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

Rebuilds the [`ReturnsResult`](@ref) of the observations a [`Pipeline`](@ref) has folded, from its row owner.

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

Reads one data step of a folded [`Pipeline`](@ref) out, and narrows the surviving assets where the step's universe does.

The method Julia selects is the step's class. A [`PricesToReturns`](@ref) reads itself out. A [`PriceGapFill`](@ref) reads the [`PriceGapFillResult`](@ref) of the whole history out of its state, restricted to the assets surviving so far, because the batch fit was made over the columns an earlier filter left. A [`MissingDataFilter`](@ref) reads its [`MissingDataFilterResult`](@ref) out the same way, and narrows the surviving set to the assets it keeps, in the order they hold. An [`AbstractAssetSelector`](@ref) runs its batch verb over the owner's rows viewed to the surviving assets, and narrows the set to the fitted universe in fitted order, as [`apply_preprocessing`](@ref) orders it. Any other row-local step reads its own read-out out.

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
    return PriceGapFillResult(res.nx[keep], res.v[keep], res.fill, res.strict), idx
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

The owner's state is viewed to the surviving assets through [`port_opt_view`](@ref) before it is read out, which is ADR 0107's slice by asset: a state fitted over the full universe, viewed to a column set and read out, is the batch fit over that column set. A prior owner reads out through `prior(pe)`; an optimisation owner is injected with the context and read out through `optimise(opt)`, so the clustering, the constraints and every uncertainty set are fitted from the reconstituted carrier exactly as batch fits them.

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

The pipeline's twin of [`fit_fold_result`](@ref). A window fits the workflow over it, `fit(pipe, data[train_idx])`, which is the refit every fold of the batch arms runs. `nothing` says *the pipeline holds its window* — the online arm of [`fold_loop`](@ref) has folded every row of it — so the fold reads the pipeline out through `fit(pipe)` with no data.

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

Renders the names and the steps of a [`Pipeline`](@ref), and its `cache` only where a state is set, so no rendering of a pipeline that took no step moves.

# Related

  - [`show_fields`](@ref)
"""
function show_fields(p::Pipeline)
    return isnothing(p.cache) ? (:names, :steps) : (:names, :steps, :cache)
end
