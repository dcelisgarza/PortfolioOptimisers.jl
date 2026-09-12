"""
    copy_states(est)
    copy_states(td::TimeDependent)

Copies every partial-fit state an estimator tree carries, and returns the tree rebuilt around the copies.

The walk of [`online_entry_state`](@ref) with a rebuild in place of a report: it copies `est`'s own `cache` when it holds a state, descends into every estimator-valued field ([`estimator_fields`](@ref)), and rebuilds each host whose fields moved through [`rebuild_estimator`](@ref). A host nothing under changed is returned as it is. A [`TimeDependent`](@ref) schedule is returned unchanged, because its entries are batch configuration resolved per fold and the loop threads no state through them, and so is anything that is not an estimator.

[`Resume`](@ref) calls it once at entry, so the Result it holds is never written: [`partial_fit!`](@ref) promises nothing about a kept estimator, a buffer appends into its backing array, and every state answers `Base.copy`.

# Arguments

  - `est`: The estimator, or any value a field holds.

# Returns

  - `est`: The tree, with every state copied.

# Related

  - [`Resume`](@ref)
  - [`online_entry_state`](@ref)
  - [`estimator_fields`](@ref)
  - [`rebuild_estimator`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function copy_states(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    T = typeof(est)
    # The field tuple is read by index, which names nothing, so a host without a `cache`
    # field reads as one carrying nothing.
    props = NamedTuple{fieldnames(T)}(ntuple(i -> getfield(est, i), Val(fieldcount(T))))
    cache = copy_state(get(props, :cache, nothing))
    fns = estimator_fields(est)
    repl = NamedTuple{fns}(map(f -> copy_states(getfield(est, f)), fns))
    moved = any(f -> getfield(repl, f) !== getfield(est, f), fns)
    if isnothing(cache) && !moved
        return est
    end
    return rebuild_estimator(est, isnothing(cache) ? repl : merge(repl, (; cache = cache)))
end
function copy_states(td::TimeDependent)
    return td
end
function copy_states(x)
    return x
end
"""
    copy_state(state::AbstractPartialFitState)
    copy_state(::Nothing)

The `cache` arm of [`copy_states`](@ref): a state is copied, and no state stays none.

# Related

  - [`copy_states`](@ref)
"""
function copy_state(state::AbstractPartialFitState)
    return copy(state)
end
function copy_state(::Nothing)
    return nothing
end
"""
    carrier_timestamps(rd::ReturnsResult)
    carrier_timestamps(pr::PricesResult)

The timestamps of a carrier, or `nothing` when it holds none.

# Related

  - [`held_timestamps`](@ref)
  - [`resume_fold_count`](@ref)
"""
function carrier_timestamps(rd::ReturnsResult)
    return rd.ts
end
function carrier_timestamps(pr::PricesResult)
    return TimeSeries.timestamp(pr.X)
end
"""
    context_timestamps(cache::ReturnsBufferState)
    context_timestamps(::Nothing)

The timestamps a Fold Context holds; none before the first step.

# Related

  - [`held_timestamps`](@ref)
"""
function context_timestamps(cache::ReturnsBufferState)
    return cache.ts
end
function context_timestamps(::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Continues an online walk-forward from the folds its Result holds, over the history extended.

A transient declaration in the estimator slot, in the idiom of [`Online`](@ref) and [`TimeDependent`](@ref), decided by [#1018](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1018) (ADR 0144). An online run's [`MultiPeriodPredictionResult`](@ref) carries the estimator the loop threaded, folded through the last training end. `Resume(res)` hands that Result back to [`cross_val_predict`](@ref) over the **full history extended** — the old carrier with the new rows appended — and the same scheme. The scheme enumerates every fold as the one-shot run does; the loop skips the folds the Result holds — located by the state's last held timestamp, so a resumed Result, which holds the new folds only, resumes again — folds the ordinary delta from the last training end into a **copy** of `res.opt`, and continues from the fold after them, threading `res.pred[end]`'s weights as the previous fold's. No warm-up runs: the warm-up was fold 1's window, and fold 1 is skipped. The resumed Result carries the new folds only and `opt` again, so the chain continues, and `vcat` on the two Results stacks them for scoring.

The oracle is the one-shot run. For an online scheme `cv`, a history `rd_T` and its extension `rd_{T+k}`,

```julia
res  = cross_val_predict(mr, rd_T, cv)
res2 = cross_val_predict(Resume(res), rd_{T+k}, cv)
vcat(res.pred, res2.pred) ≈ cross_val_predict(mr, rd_{T+k}, cv).pred   # fold for fold
```

to the tolerance of the moment layer and of the solver for the weights, and exactly for the carrier the read-out rebuilds, a capped run included. `res` is never written — `Resume` copies every state at entry — so one Result resumes any number of times, and a terminal view never spoils a later continuation.

The re-entry checks four things, each an `ArgumentError` by name. The scheme is a walk-forward with a Fold Fit, and it adds at least one fold. **Timestamps are required**: the carrier and the state must both hold them, and every held timestamp must equal its row of the carrier — `state.ts == rd.ts[(r - h + 1):r]` for `r` the last training end and `h` the held count — which pins the prefix exactly over the held span, under a cap or not; an index-only caller attaches a synthetic calendar, `ts = Date(1) .+ Day.(0:(T - 1))`. And **a partial last fold is terminal**: with `reduce_test = true` the old run's last fold is the first part of a full window of the longer run, so it can be neither skipped nor completed. Resume the same Result terminally with `reduce_test = true` for the live view of the leftover rows, and again with `reduce_test = false` when the rows arrive.

Every state is an immutable struct of arrays, names, integers and a timestamp vector, so a Result of an online run round-trips through the stdlib `Serialization` as it stands, and a deserialised Result resumes to the same weights. That format is bound to the Julia version that wrote it, and no other format is provided.

A batch Result, a [`PopulationPredictionResult`](@ref) (a [`MultipleRandomised`](@ref) run), a scheme that is not a walk-forward, a scheme with no Fold Fit, and a search are refused by name. A [`Pipeline`](@ref) host resumes on the same terms, through the state its row owner keeps.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Resume(res::MultiPeriodPredictionResult)
    Resume(; res::MultiPeriodPredictionResult)

## Validation

  - `res.opt` is not `nothing`. An `ArgumentError` is thrown otherwise, because a batch run threads no estimator.
  - `res` is not a [`PopulationPredictionResult`](@ref). An `ArgumentError` is thrown otherwise.

# Related

  - [`MultiPeriodPredictionResult`](@ref)
  - [`OnlineStep`](@ref)
  - [`online_folds`](@ref)
  - [`cross_val_predict`](@ref)
  - [`Base.vcat(a::MultiPeriodPredictionResult, b::MultiPeriodPredictionResult)`](@ref)
  - [`Online`](@ref)
  - [`TimeDependent`](@ref)
"""
struct Resume{T1}
    """
    The Result of the online run to continue, carrying the threaded estimator in `opt`.
    """
    res::T1
    function Resume(res::MultiPeriodPredictionResult)
        @argcheck(!isnothing(res.opt),
                  ArgumentError("`Resume` continues an online run, and this Result carries no estimator (`res.opt === nothing`), so it is a batch run and there is nothing to continue. Run the walk-forward with `ff = OnlineStep()`, and resume the Result it returns."))
        return new{typeof(res)}(res)
    end
end
function Resume(::PopulationPredictionResult)
    return throw(ArgumentError("`Resume` takes the `MultiPeriodPredictionResult` of one online walk-forward, and a `PopulationPredictionResult` is a population of paths from a `MultipleRandomised` run: whether a longer history reproduces each path's asset subset is not established, so a resumed population is refused."))
end
function Resume(; res::MultiPeriodPredictionResult)::Resume
    return Resume(res)
end
"""
    is_time_dependent(r::Resume)
    needs_previous_weights(r::Resume)
    assert_time_dependent_fold_count(r::Resume, n::Integer, all_binds::Bool = true)

The fold loop reads its three traits off the root it is handed, and a [`Resume`](@ref) root answers for the estimator its Result carries: the loop resolves the declaration at entry, and the per-fold copy is made of that estimator.

# Related

  - [`fold_loop`](@ref)
  - [`is_time_dependent`](@ref)
  - [`needs_previous_weights`](@ref)
"""
is_time_dependent(r::Resume) = is_time_dependent(r.res.opt)
needs_previous_weights(r::Resume) = needs_previous_weights(r.res.opt)
function assert_time_dependent_fold_count(r::Resume, n::Integer,
                                          all_binds::Bool = true)::Nothing
    return assert_time_dependent_fold_count(r.res.opt, n, all_binds)
end
"""
    assert_resume_scheme(cv::WFCVER)
    assert_resume_scheme(cv::CVER)

Refuses a scheme a [`Resume`](@ref) cannot re-enter, by name: one that is not a walk-forward, and a walk-forward with no Fold Fit.

# Related

  - [`Resume`](@ref)
  - [`fold_fit`](@ref)
  - [`OnlineStep`](@ref)
"""
function assert_resume_scheme(cv::WFCVER)::Nothing
    @argcheck(!isnothing(fold_fit(cv)),
              ArgumentError("`Resume` continues the online arm of the fold loop, and this walk-forward declares no Fold Fit, so every fold would refit from its training window and there is no state to continue. Set `ff = OnlineStep()` on the scheme, the same one the Result came from."))
    return nothing
end
function assert_resume_scheme(cv::CVER)::Nothing
    return throw(ArgumentError("`Resume` continues an online walk-forward, and a `$(typeof(cv).name.name)` is not one: only a walk-forward threads one estimator from fold to fold, so only a walk-forward has a fold to continue from. A `MultipleRandomised` run is a population of paths, and its resume is not established."))
end
"""
    resume_fold_count(opt, rd, train_idx) -> Int

Finds the fold a [`Resume`](@ref) continues from, by the timestamps, and refuses a carrier whose rows do not carry the state.

The state's last held timestamp names the row it folded through, and the fold whose training window ends at that row is the last fold the run holds: `n_old` is that fold's index in the scheme's enumeration over the extended carrier, and the loop continues from `n_old + 1`. The count is read off the state and not off `length(res.pred)`, because a resumed Result holds the new folds only, and the chain `Resume(res2)` must skip every fold before it. Both the carrier and the state must hold timestamps, and every timestamp the state holds must equal its row of the carrier: `held == carrier_timestamps(rd)[(r - h + 1):r]` for `r = last(train_idx[n_old])` and `h` the held count. That is exact over the held span at `O(h)`, and it is the one check that pins a prefix — a row dropped or inserted before `r` moves the carrier's row `r`, and a changed scheme moves every training end — under a cap, where the state holds no total, as much as without one.

# Arguments

  - `opt`: The stepped estimator the Result carries.
  - `rd`: The carrier of the full history extended.
  - `train_idx`: The training windows of every fold the scheme enumerates over `rd`.

# Validation

  - The carrier holds timestamps. An `ArgumentError` is thrown otherwise.
  - The state holds timestamps. An `ArgumentError` is thrown otherwise.
  - Some fold's training window ends at the state's last held timestamp. An `ArgumentError` is thrown otherwise.
  - The held timestamps equal the carrier's over the held span. An `ArgumentError` is thrown otherwise.

# Returns

  - `n_old::Int`: The index of the last fold the run holds.

# Related

  - [`Resume`](@ref)
  - [`held_timestamps`](@ref)
  - [`carrier_timestamps`](@ref)
"""
function resume_fold_count(opt, rd, train_idx)
    given = carrier_timestamps(rd)
    @argcheck(!isnothing(given),
              ArgumentError("`Resume` aligns the carrier with the state by their timestamps, and this carrier holds none. Give the carrier the timestamps the run was made over; an index-only caller attaches a synthetic calendar, `ts = Date(1) .+ Day.(0:(T - 1))`."))
    held = held_timestamps(opt)
    @argcheck(!isnothing(held),
              ArgumentError("`Resume` aligns the carrier with the state by their timestamps, and the `$(typeof(opt).name.name)` the Result carries holds none: either the run was made over a carrier without timestamps, or the estimator keeps no Fold Context. Run the walk-forward over a carrier with `ts` set; an index-only caller attaches a synthetic calendar, `ts = Date(1) .+ Day.(0:(T - 1))`."))
    stop = last(held)
    n_old = findfirst(i -> given[last(train_idx[i])] == stop, eachindex(train_idx))
    @argcheck(!isnothing(n_old),
              ArgumentError("`Resume` continues from the fold whose training window ends where the state stopped, $(repr(stop)), and no fold of the scheme over this carrier ends there: the training ends run $(repr(given[last(train_idx[1])])) to $(repr(given[last(train_idx[end])])). Hand `Resume` the full history the run was made over with the new rows appended, and the same scheme."))
    r = last(train_idx[n_old])
    h = length(held)
    @argcheck(h <= r && held == view(given, (r - h + 1):r),
              ArgumentError("`Resume` continues a run whose state folded through row $(r), holding the last $(h) timestamp(s), and the carrier's rows $(max(r - h + 1, 1)):$(r) do not carry them: the held span runs $(repr(first(held))) to $(repr(stop)), and the carrier's $(repr(given[max(r - h + 1, 1)])) to $(repr(given[r])). Hand `Resume` the full history the run was made over with the new rows appended, and the same scheme."))
    return n_old
end
"""
    assert_resume_folds(n_old::Integer, n::Integer)

Refuses a carrier that adds no fold to the run a [`Resume`](@ref) continues.

# Related

  - [`Resume`](@ref)
  - [`resume_fold_count`](@ref)
"""
function assert_resume_folds(n_old::Integer, n::Integer)::Nothing
    @argcheck(n > n_old,
              ArgumentError("`Resume` continues from fold $(n_old + 1), and the scheme enumerates $(n) fold(s) over this carrier, so the carrier adds no fold to the $(n_old) the run holds. Hand it the full history with the new rows appended."))
    return nothing
end
"""
    assert_resume_full_fold(test_idx::VecInt, pred::PredictionResult, ts)

Refuses a [`Resume`](@ref) whose last held fold was partial: its test window under the longer history runs past the last row the fold predicted.

With `reduce_test = true` the last fold of the old run predicted the rows left over at the end of the carrier, which under the longer history are the first part of a full window — the same training end, the same weights, too short a span. Skipping it loses rows and completing it needs a mid-window entry, so it is terminal. The check compares timestamps, not counts — the fold's last predicted timestamp against the carrier's at the window's last row — because a price-level window of `L` rows predicts `L - 1` returns, and the timestamp of the last one is the last price's either way. The message names the two-resume workaround.

# Arguments

  - `test_idx`: The test window of the last held fold, over the extended carrier.
  - `pred`: The last held fold's prediction.
  - `ts`: The extended carrier's timestamps.

# Related

  - [`Resume`](@ref)
  - [`IndexWalkForward`](@ref)
"""
function assert_resume_full_fold(test_idx::VecInt, pred::PredictionResult, ts)::Nothing
    predicted = last(pred.rd.ts)
    window = ts[last(test_idx)]
    @argcheck(predicted == window,
              ArgumentError("`Resume` continues from the fold after the last one the Result holds, and that fold was partial: it predicted through $(repr(predicted)) and its test window over the longer history runs through $(repr(window)), so the leftover rows of the old carrier were a reduced last window (`reduce_test = true`). A partial last fold is terminal, because the fold cannot be skipped without losing rows or completed without a mid-window entry. Resume the same Result terminally with `reduce_test = true` for the live view of the leftover rows, and resume it again with `reduce_test = false` to continue when the rows arrive."))
    return nothing
end
"""
    cv_resume_info(n_old::Integer, n::Integer)

Build the informational message emitted when a cross-validation run resumes from a Result, the counterpart of [`cv_online_info`](@ref) for the resumed arm of [`online_folds`](@ref).

# Returns

  - `msg::String`: The message.

# Related

  - [`Resume`](@ref)
  - [`cv_online_info`](@ref)
"""
function cv_resume_info(n_old::Integer, n::Integer)
    return "Resuming cross-validation online from fold $(n_old + 1) of $(n): the scheme enumerates $(n) fold(s) over the extended history, the first $(n_old) end where the Result's estimator stopped and are skipped, and the estimator the Result carries is copied and folded from its last training end through the rest, in order. The Result returned holds the new folds only; `vcat` it onto the old one to score the whole run."
end
"""
    online_folds(fit_fold, r::Resume, n::Integer, ::Type{ElT}; rd, train_idx, test_idx, fold_view, pws)

The resumed arm of [`fold_loop`](@ref): continue an online walk-forward from the folds its Result holds.

Taken when the estimator slot holds a [`Resume`](@ref). It resolves the declaration at entry — a copy of `res.opt` through [`copy_states`](@ref), the last threadable fold of `res.pred` as the previous prediction, advanced over the folds the Result holds by [`advance_previous_fold`](@ref) exactly as the one-shot loop advanced it, and the folds to skip, `n_old`, read off the state's last held timestamp by [`resume_fold_count`](@ref) — and the declaration is gone before any verb below the loop meets it. The count comes from the state and not from `length(res.pred)`, because a resumed Result holds the new folds only and the chain `Resume(res2)` skips every fold before it. The re-entry checks run in order: the held timestamps name a fold and equal their rows ([`resume_fold_count`](@ref)), the carrier adds a fold ([`assert_resume_folds`](@ref)), and the last held fold was not partial ([`assert_resume_full_fold`](@ref)). The folds `(n_old + 1):n` then run through [`thread_online_folds!`](@ref), the body the online arm runs, from the last training end `last(train_idx[n_old])`. No warm-up runs. The schedule's [`TimeDependentContext`](@ref) carries the combined enumeration's `i` and `n`, and the pinned context — names, static panel, column presence — runs on every delta step through the state's own step.

# Arguments

  - `fit_fold`: The per-fold resolution and callback [`fold_loop`](@ref) builds.
  - `r`: The declaration, holding the Result to continue.
  - `n`: The number of folds the scheme enumerates over the extended history.
  - `rd`: The carrier of the full history extended.
  - `train_idx`: The training windows of every fold, in split order.
  - `test_idx`: The test windows of every fold, in split order.
  - `fold_view`: Unread; a multiple-randomised path never reaches this arm.
  - `pws`: The scheme's Previous-Weights Source, or `nothing`.

# Validation

  - Everything [`resume_fold_count`](@ref), [`assert_resume_folds`](@ref) and [`assert_resume_full_fold`](@ref) refuse.

# Returns

  - `predictions::Vector{ElT}`: One prediction per new fold, in split order.
  - `est`: The threaded estimator, folded through `last(train_idx[n])`.

# Related

  - [`Resume`](@ref)
  - [`fold_loop`](@ref)
  - [`thread_online_folds!`](@ref)
  - [`resume_fold_count`](@ref)
  - [`copy_states`](@ref)
  - [`cv_resume_info`](@ref)
"""
function online_folds(fit_fold, r::Resume, n::Integer, ::Type{ElT}; rd, train_idx, test_idx,
                      fold_view = nothing, pws = nothing) where {ElT}
    res = r.res
    n_old = resume_fold_count(res.opt, rd, train_idx)
    @info(cv_resume_info(n_old, n))
    assert_resume_folds(n_old, n)
    last_end = last(train_idx[n_old])
    assert_resume_full_fold(test_idx[n_old], res.pred[end], carrier_timestamps(rd))
    prev = nothing
    for pred in res.pred
        prev = advance_previous_fold(pws, prev, pred)
    end
    predictions = Vector{ElT}(undef, n - n_old)
    est = thread_online_folds!(predictions, fit_fold, copy_states(res.opt), (n_old + 1):n,
                               prev; rd = rd, train_idx = train_idx, last_end = last_end,
                               pws = pws)
    return predictions, est
end
"""
    OptimiserResume = Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any, <:NonFiniteAllocationOptimisationEstimator}}

Alias for a [`Resume`](@ref) whose Result carries an optimiser: the declaration the optimiser doors take.

# Related

  - [`Resume`](@ref)
  - [`cross_val_predict`](@ref)
"""
const OptimiserResume = Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any,
                                                             <:NonFiniteAllocationOptimisationEstimator}}
"""
    cross_val_predict(r::OptimiserResume, rd::ReturnsResult, cv::CVER; cols = :, ex = FLoops.ThreadedEx())
    fit_and_predict(r::OptimiserResume, rd::ReturnsResult, cv::CVER; cols = :, ex = FLoops.ThreadedEx(), id = nothing)

Continue an online walk-forward from a Result, over the full history extended.

The doors that take an optimiser in the estimator slot take a [`Resume`](@ref) too. The carrier is viewed by `cols` as the one-shot door views it, and the estimator is not: the Result's estimator was threaded over the old run's view already, so the pinned context refuses a different one at the first delta step. The scheme is checked at the door ([`assert_resume_scheme`](@ref)) and the fold loop takes its resumed arm. The Result returned holds the new folds only, its `opt` folded through the last training end, and `id`.

# Arguments

  - `r`: The declaration, holding the Result to continue.
  - $(arg_dict[:rd])
  - `cv`: The scheme the Result came from, a walk-forward with `ff = OnlineStep()`.
  - `cols`: The asset view of the carrier.
  - `ex`: Unread; the resumed arm runs in order.
  - `id`: The identifier the Result carries.

# Validation

  - Everything [`assert_resume_scheme`](@ref) and the resumed arm of [`online_folds`](@ref) refuse.

# Returns

  - `res::MultiPeriodPredictionResult`: The new folds, carrying the threaded estimator.

# Related

  - [`Resume`](@ref)
  - [`online_folds`](@ref)
  - [`cross_val_predict`](@ref)
  - [`fit_and_predict`](@ref)
"""
function cross_val_predict(r::OptimiserResume, rd::ReturnsResult, cv::CVER; cols = :,
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    if !isa(cols, Colon)
        rd = port_opt_view(rd, cols)
    end
    return fit_and_predict(r, rd, cv; ex = ex)
end
function fit_and_predict(r::OptimiserResume, rd::ReturnsResult, cv::CVER; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         id = nothing)
    assert_resume_scheme(cv)
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    (; wd, pws, fa, store_weight_path, strict) = fold_evaluation(cv)
    hwd = held_weights_drift(wd, pws)
    predictions, est = fold_loop(r, length(train_idx), ex; rd = rd, train_idx = train_idx,
                                 test_idx = test_idx, cv = cv, pws = pws) do fold
        return fit_and_predict(fold.est, fold.rd; train_idx = fold.train,
                               test_idx = fold.test, cols = cols, wd = wd, hwd = hwd,
                               fa = fa, store_weight_path = store_weight_path,
                               strict = strict, w_prev = fold.w_prev)
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id, opt = est)
end
"""
    search_cross_validation(::Resume, ::AbstractSearchCrossValidationEstimator, ::Any)

A search refuses a [`Resume`](@ref) by name: a search scores every candidate through the one fold loop, so a resumed search is per-candidate Results the search Result does not carry.

# Related

  - [`Resume`](@ref)
  - [`search_cross_validation`](@ref)
"""
function search_cross_validation(::Resume, ::AbstractSearchCrossValidationEstimator, ::Any)
    return throw(ArgumentError("a search takes the configuration it searches over and scores every candidate through the one fold loop, so a `Resume` has no place in it: a resumed search would need one Result per candidate, which a search Result does not carry. Resume the walk-forward of the candidate the search picked, `Resume(cross_val_predict(res.opt, rd, cv))`."))
end
"""
    Base.vcat(a::MultiPeriodPredictionResult, b::MultiPeriodPredictionResult)

Stacks the folds of a run and of its resume into one Result, for scoring.

`pred` is concatenated and `mrd` re-stacked by the constructor; the Result carries `a`'s `id` and `b`'s `opt`, the estimator the later run threaded. The two must abut in time: the last row of `a` is before the first row of `b`, both read off the stacked carriers' timestamps, so two Results that overlap, or that come from carriers without timestamps, are refused by name.

# Arguments

  - `a`: The earlier Result.
  - `b`: The later Result, a resume of `a`.

# Validation

  - Both carry timestamps and `a` ends before `b` starts. An `ArgumentError` is thrown otherwise.

# Returns

  - `res::MultiPeriodPredictionResult`: The stacked Result.

# Related

  - [`Resume`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
"""
function Base.vcat(a::MultiPeriodPredictionResult, b::MultiPeriodPredictionResult)
    ta = a.mrd.ts
    tb = b.mrd.ts
    @argcheck(!isnothing(ta) && !isnothing(tb),
              ArgumentError("two `MultiPeriodPredictionResult`s stack by their timestamps, and $(isnothing(ta) ? "the earlier" : "the later") one holds none. Make the runs over a carrier with `ts` set; an index-only caller attaches a synthetic calendar, `ts = Date(1) .+ Day.(0:(T - 1))`."))
    @argcheck(ta[end] < tb[1],
              ArgumentError("two `MultiPeriodPredictionResult`s stack when the earlier ends before the later starts, and this pair does not: the earlier ends at $(repr(ta[end])) and the later starts at $(repr(tb[1])). Stack a run and its resume, in order."))
    return MultiPeriodPredictionResult(; pred = vcat(a.pred, b.pred), id = a.id,
                                       opt = b.opt)
end

export Resume
