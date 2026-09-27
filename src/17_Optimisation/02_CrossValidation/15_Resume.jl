"""
    copy_states(est)
    copy_states(td::TimeDependent)

Copies every partial-fit state that an estimator tree carries, and returns the tree rebuilt around the copies.

The walk is the one [`online_entry_state`](@ref) takes, with a rebuild where that verb makes a report. A host whose subtree holds no state comes back as it is. A [`TimeDependent`](@ref) schedule comes back unchanged, because its entries are batch configuration that the fold loop resolves per fold and threads no state through. Any value that is not an estimator comes back unchanged too.

Two callers use it. [`Resume`](@ref) calls it once at entry, so the Result it holds is never written. That copy is necessary because [`partial_fit!`](@ref) promises nothing about a kept estimator, and a buffer appends into its backing array. The generic method of [`partial_fit`](@ref) calls it before every fold. The value form therefore keeps its promise on a host that folds through the states of its members, such as a [`HighOrderPriorEstimator`](@ref) or a hierarchical optimiser, as well as on a leaf that carries its own `cache`. Every state type defines `Base.copy`, so one copy per state is enough.

# Algorithm

 1. Read the `cache` field of `est`, or `nothing` when `est` has no such field, and copy it through [`copy_state`](@ref), giving `cache`.
 2. Call `copy_states` on every estimator-valued field of `est` ([`estimator_fields`](@ref)), giving `repl`.
 3. Set `moved` when an entry of `repl` is not the object (`!==`) that `est` holds in that field.
 4. Return `est` when `cache` is `nothing` and `moved` is false.
 5. Otherwise rebuild `est` through [`rebuild_estimator`](@ref) from `repl`, and from `cache` too when it is not `nothing`.

# Arguments

  - `est`: The estimator, or any value that a field holds.

# Returns

  - `est`: The tree, with every state copied.

# Related

  - [`Resume`](@ref)
  - [`partial_fit`](@ref)
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

Copies the partial-fit state in a `cache` field, and returns `nothing` when the field holds none.

This is the `cache` arm of [`copy_states`](@ref).

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

Returns the timestamps of a carrier, or `nothing` when it holds none.

A prices carrier always holds them, because its `TimeArray` has a timestamp column.

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

Returns the timestamps that a Fold Context holds, or `nothing` before the first step.

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

Continues an online walk-forward from the folds that its Result holds, over a longer history.

`Resume` is a transient declaration in the estimator slot, like [`Online`](@ref) and [`TimeDependent`](@ref). The [`MultiPeriodPredictionResult`](@ref) of an online run carries in `opt` the estimator that the fold loop threaded, folded through the last training end. `Resume(res)` hands that Result back to [`cross_val_predict`](@ref), with the same scheme and the **full history extended**, which is the old carrier with the new rows appended. The loop skips the folds that the Result holds, folds the new rows into a copy of `res.opt`, and runs the folds after them. No warm-up runs. The resumed Result holds the new folds only and carries `opt` again, so it can be resumed in turn, and `vcat` stacks the two Results for scoring.

The oracle is the one-shot run. For an online scheme `cv`, a history `rd_T` and its extension `rd_{T+k}`,

```julia
res  = cross_val_predict(mr, rd_T, cv)
res2 = cross_val_predict(Resume(res), rd_{T+k}, cv)
vcat(res.pred, res2.pred) ≈ cross_val_predict(mr, rd_{T+k}, cv).pred   # fold for fold
```

The weights agree to the tolerance of the moment layer and of the solver. The carrier that the read-out rebuilds agrees exactly, under a cap too. `Resume` copies every state at entry, so `res` is never written. One Result can be resumed any number of times, and a terminal view does not change a later continuation.

`cross_val_predict` refuses a resume in five cases, each with an `ArgumentError` that names the cause.

  - The scheme is not an Online Scheme.
  - The carrier adds no fold.
  - The carrier or the state holds no timestamps, or the timestamps that the state holds differ from their rows of the carrier. An index-only caller attaches a synthetic calendar, `ts = Date(1) .+ Day.(0:(T - 1))`.
  - The last fold of the Result was partial. With `reduce_test = true` that fold predicted the rows left at the end of the old carrier, and in the longer history those rows are the first part of a full window. The fold can be neither skipped nor completed. Resume the same Result with `reduce_test = true` for the live view of the leftover rows, and resume it again with `reduce_test = false` when the rows arrive.
  - The head is an [`OnlinePortfolioSelection`](@ref) whose fees carry a turnover term, and the scheme has no Previous-Weights Source. The one-shot online arm refuses the same head.

The timestamp check reads the training ends of the scheme. It refuses a scheme with a different window, step or purge, because such a scheme moves the training ends. A change of the Weight Drift or of the Previous-Weights Source moves no training end, so the check does not see it. The new folds then run under the scheme handed, and the identity above does not hold. Under a cap the state holds its last rows only, and the check pins those rows and nothing before them.

The previous weights come from the Result alone. When no fold of `res` passes its weights on, for example because every fold of a short resumed Result failed, the first new fold gets no previous weights. The one-shot run hands it the last threadable fold of the earlier run instead. Resume `vcat(old, res)` in that case, because the stacked Result holds the earlier folds too.

Every state is an immutable struct of arrays, names, integers and a timestamp vector. The Result of an online run therefore round-trips through the stdlib `Serialization` as it stands, and the deserialised Result resumes to the same weights. That format is bound to the Julia version that wrote it, and the package gives no other format.

The constructor refuses a batch Result and a [`PopulationPredictionResult`](@ref) from a [`MultipleRandomised`](@ref) run. `cross_val_predict` refuses a scheme that is not a walk-forward, a walk-forward that is not an Online Scheme, and a search. A [`Pipeline`](@ref) host resumes on the same terms, through the state that its row owner keeps.

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
  - [`OnlineIndexWalkForward`](@ref)
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
                  ArgumentError("`Resume` continues an online run, and this Result carries no estimator (`res.opt === nothing`), so it is a batch run and there is nothing to continue. Run the walk-forward as an Online Scheme (`OnlineIndexWalkForward`, `OnlineDateWalkForward` or `OnlineHindsightSplit`), and resume the Result it returns."))
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

Answers the three traits of the fold loop for the estimator that the Result of a [`Resume`](@ref) carries.

The fold loop reads the traits off the root that it is handed. The loop resolves a `Resume` at entry and makes the per-fold copy of the estimator that its Result carries, so that estimator answers.

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
    assert_resume_scheme(::Online{<:WalkForwardEstimator})
    assert_resume_scheme(cv::WFCVER)
    assert_resume_scheme(cv::CVER)

Refuses, by name, a scheme that a [`Resume`](@ref) cannot re-enter.

Two kinds of scheme are refused: a scheme that is not a walk-forward, and a walk-forward that is not an Online Scheme. Each refusal is a method on the refused type, so dispatch decides it and the message names the cause.

# Validation

  - `cv` is an Online Scheme over a walk-forward. An `ArgumentError` is thrown otherwise.

# Related

  - [`Resume`](@ref)
  - [`folds_are_stepped`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
"""
function assert_resume_scheme(::Online{<:WalkForwardEstimator})::Nothing
    return nothing
end
function assert_resume_scheme(cv::WFCVER)::Nothing
    return throw(ArgumentError("`Resume` continues the online arm of the fold loop, and this walk-forward is not an Online Scheme, so every fold would refit from its training window and there is no state to continue. Build the scheme the Result came from with `OnlineIndexWalkForward`, `OnlineDateWalkForward` or `OnlineHindsightSplit`."))
end
function assert_resume_scheme(cv::CVER)::Nothing
    return throw(ArgumentError("`Resume` continues an online walk-forward, and a `$(typeof(cv).name.name)` is not one: only a walk-forward threads one estimator from fold to fold, so only a walk-forward has a fold to continue from. A `MultipleRandomised` run is a population of paths, and its resume is not established."))
end
"""
    resume_fold_count(opt, rd, train_idx) -> Int

Finds, by timestamp, the last fold that a [`Resume`](@ref) holds, and refuses a carrier whose rows do not carry the state.

The last timestamp that the state holds names the last row it folded. The fold whose training window ends at that row is the last fold that the run holds, and the loop continues from the fold after it. The count comes from the state and not from `length(res.pred)`, because a resumed Result holds the new folds only, and the chain `Resume(res2)` must skip every fold before it.

The check is exact over the held span, and it costs one comparison per held timestamp. Without a cap the state holds every row that it folded, so the check pins the whole prefix. A row dropped or inserted before `r` moves the row of the carrier at `r`, and a scheme with a different window, step or purge moves the training ends. Under a cap the state holds its last rows only, so the check pins those rows and nothing before them. A carrier that differs before the held span then resumes to the same weights, because no row before the span reaches the state.

# Algorithm

 1. Read the timestamps of the carrier through [`carrier_timestamps`](@ref), giving `given`. Refuse `nothing`.
 2. Read the timestamps of the state through [`held_timestamps`](@ref), giving `held`. Refuse `nothing`.
 3. Take the last entry of `held`, giving `stop`.
 4. Find the first fold whose training window ends at a row with timestamp `stop`, giving `n_old`. Refuse when no fold ends there.
 5. Take the last row of that training window, giving `r`, and the length of `held`, giving `h`.
 6. Refuse unless `h <= r` and `held` equals `given[(r - h + 1):r]`.
 7. Return `n_old`.

# Arguments

  - `opt`: The stepped estimator that the Result carries.
  - `rd`: The carrier of the full history extended.
  - `train_idx`: The training windows of every fold that the scheme enumerates over `rd`.

# Validation

  - The carrier holds timestamps. An `ArgumentError` is thrown otherwise.
  - The state holds timestamps. An `ArgumentError` is thrown otherwise.
  - The training window of some fold ends at the last timestamp that the state holds. An `ArgumentError` is thrown otherwise.
  - The held timestamps equal the timestamps of the carrier over the held span. An `ArgumentError` is thrown otherwise.

# Returns

  - `n_old::Int`: The index of the last fold that the run holds.

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

Refuses a carrier that adds no fold to the run that a [`Resume`](@ref) continues.

# Arguments

  - `n_old`: The index of the last fold that the Result holds.
  - `n`: The number of folds that the scheme enumerates over the carrier.

# Validation

  - `n > n_old`. An `ArgumentError` is thrown otherwise.

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

Refuses a [`Resume`](@ref) whose last held fold was partial.

A fold was partial when its test window in the longer history runs past the last row that the fold predicted. With `reduce_test = true` the last fold of the old run predicted the rows left at the end of its carrier. In the longer history those rows are the first part of a full window, with the same training end and the same weights. The fold cannot be skipped without a loss of rows, and it cannot be completed without an entry in the middle of a window, so it is terminal.

The check compares timestamps and not counts. A price-level window of `L` rows predicts `L - 1` returns, and the last return has the timestamp of the last price, so one comparison of timestamps holds at both levels. The message names the workaround with two resumes.

# Arguments

  - `test_idx`: The test window of the last held fold, over the extended carrier.
  - `pred`: The prediction of the last held fold.
  - `ts`: The timestamps of the extended carrier.

# Validation

  - The last timestamp that `pred` predicted equals `ts[last(test_idx)]`. An `ArgumentError` is thrown otherwise.

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
    online_folds(fit_fold, r::Resume, n::Integer, ::Type{ElT}, path_id = nothing; rd, train_idx, test_idx, fold_view, pws)

Runs the resumed arm of [`fold_loop`](@ref), which continues an online walk-forward from the folds that its Result holds.

The loop takes this arm when the estimator slot holds a [`Resume`](@ref). The arm resolves the declaration at entry, so no verb below the loop meets it. The new folds run through [`thread_online_folds!`](@ref), the body that the online arm runs, so a resume continues exactly the loop that started the run. The [`TimeDependentContext`](@ref) of a schedule carries the `i` and `n` of the enumeration over the extended history. Every step checks the context that the state pinned, such as the asset names, a static panel and the column presence, as the one-shot run checks it.

# Algorithm

 1. Refuse, through [`assert_online_fee_source`](@ref), a head whose fee cannot be measured without a Previous-Weights Source, as the one-shot online arm refuses it.
 2. Find the index of the last fold that the Result holds through [`resume_fold_count`](@ref), giving `n_old`.
 3. Refuse a carrier that adds no fold, through [`assert_resume_folds`](@ref).
 4. Take the last row of the training window of fold `n_old`, giving `last_end`.
 5. Refuse a partial last held fold, through [`assert_resume_full_fold`](@ref).
 6. Advance over every fold of `res.pred` through [`advance_previous_fold`](@ref), as the one-shot loop advanced over them, giving `prev`, the last threadable fold.
 7. Copy every state of `res.opt` through [`copy_states`](@ref).
 8. Run the folds `(n_old + 1):n` on the copy from `last_end` through [`thread_online_folds!`](@ref), giving `predictions` and the threaded `est`.

# Arguments

  - `fit_fold`: The per-fold resolution and callback that [`fold_loop`](@ref) builds.
  - `r`: The declaration, holding the Result to continue.
  - `n`: The number of folds that the scheme enumerates over the extended history.
  - `path_id`: The path that the folds belong to, or `nothing`. Only the context of a fold reads it.
  - `rd`: The carrier of the full history extended.
  - `train_idx`: The training windows of every fold, in split order.
  - `test_idx`: The test windows of every fold, in split order.
  - `fold_view`: Not read, because a multiple-randomised path never reaches this arm.
  - `pws`: The Previous-Weights Source of the scheme, or `nothing`.

# Validation

  - Everything that [`assert_online_fee_source`](@ref), [`resume_fold_count`](@ref), [`assert_resume_folds`](@ref) and [`assert_resume_full_fold`](@ref) refuse.

# Returns

  - `predictions::Vector{ElT}`: One prediction per new fold, in split order.
  - `est`: The threaded estimator, folded through `last(train_idx[n])`.

# Related

  - [`Resume`](@ref)
  - [`fold_loop`](@ref)
  - [`thread_online_folds!`](@ref)
  - [`resume_fold_count`](@ref)
  - [`copy_states`](@ref)
"""
function online_folds(fit_fold, r::Resume, n::Integer, ::Type{ElT}, path_id = nothing; rd,
                      train_idx, test_idx, fold_view = nothing, pws = nothing) where {ElT}
    res = r.res
    assert_online_fee_source(res.opt, pws)
    n_old = resume_fold_count(res.opt, rd, train_idx)
    assert_resume_folds(n_old, n)
    last_end = last(train_idx[n_old])
    assert_resume_full_fold(test_idx[n_old], res.pred[end], carrier_timestamps(rd))
    prev = nothing
    for pred in res.pred
        prev = advance_previous_fold(pws, prev, pred)
    end
    predictions = Vector{ElT}(undef, n - n_old)
    est = thread_online_folds!(predictions, fit_fold, copy_states(res.opt), (n_old + 1):n,
                               prev; rd = rd, train_idx = train_idx, test_idx = test_idx,
                               n = n, last_end = last_end, pws = pws, path_id = path_id)
    return predictions, est
end
"""
    const OptimiserResume = Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any, <:NonFiniteAllocationOptimisationEstimator}}

Groups the [`Resume`](@ref) declarations whose Result carries an optimiser, so that the optimiser doors can take them.

The doors of [`cross_val_predict`](@ref) and [`fit_and_predict`](@ref) that take an optimiser dispatch on this alias. A `Resume` whose Result carries a [`Pipeline`](@ref) takes the pipeline door instead, through [`PipelineResume`](@ref).

# Related

  - [`Resume`](@ref)
  - [`PipelineResume`](@ref)
  - [`cross_val_predict`](@ref)
"""
const OptimiserResume = Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any,
                                                             <:NonFiniteAllocationOptimisationEstimator}}
"""
    cross_val_predict(r::OptimiserResume, rd::ReturnsResult, cv::CVER; cols = :, ex = FLoops.ThreadedEx())
    fit_and_predict(r::OptimiserResume, rd::ReturnsResult, cv::CVER; cols = :, ex = FLoops.ThreadedEx(), id = nothing)

Continues an online walk-forward from a Result, over the full history extended.

The doors that take an optimiser in the estimator slot take a [`Resume`](@ref) too. `cross_val_predict` views the carrier by `cols` as the one-shot door views it, but it leaves the estimator of the Result as it is. That estimator was threaded over the view of the old run already, and its state refuses a different view at the first step. The Result that the doors return holds the new folds only.

# Algorithm

 1. For `cross_val_predict` only, run [`assert_internal_optimiser`](@ref) and [`assert_external_optimiser`](@ref) on `r.res.opt`, and view `rd` by `cols` when `cols` is not `:`.
 2. Refuse a scheme that a `Resume` cannot re-enter, through [`assert_resume_scheme`](@ref).
 3. Split `rd` by `cv`, giving `train_idx` and `test_idx`, and refuse shuffled folds through [`assert_unshuffled_folds`](@ref).
 4. Read the evaluation settings of `cv` through [`fold_evaluation`](@ref), giving `wd`, `pws`, `fa`, `store_weight_path` and `strict`, and the drift of the held weights through [`held_weights_drift`](@ref), giving `hwd`.
 5. Run [`fold_loop`](@ref) on `r`, which takes the resumed arm of [`online_folds`](@ref). Each fold calls the one-fold `fit_and_predict` with its estimator, its test window, its previous weights and the settings of step 4, giving `predictions` and the threaded `est`.
 6. Return a [`MultiPeriodPredictionResult`](@ref) of `predictions`, with `id`, and with `est` as `opt`.

# Arguments

  - `r`: The declaration, holding the Result to continue.
  - $(arg_dict[:rd])
  - `cv`: The scheme that the Result came from, an Online Scheme.
  - `cols`: The asset view. `cross_val_predict` views the carrier by it. `fit_and_predict` hands it to every fold, which views its estimator and its prediction by it.
  - `ex`: Not read, because the resumed arm runs the folds in order.
  - `id`: The identifier that the Result carries.

# Validation

  - Everything that [`assert_resume_scheme`](@ref) and the resumed arm of [`online_folds`](@ref) refuse.
  - For `cross_val_predict`, everything that [`assert_internal_optimiser`](@ref) and [`assert_external_optimiser`](@ref) refuse on `r.res.opt`.

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
    assert_internal_optimiser(r.res.opt)
    assert_external_optimiser(r.res.opt)
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

Refuses a [`Resume`](@ref) in a search, by name.

A search scores every candidate through one fold loop. A resumed search would need one Result per candidate, and the Result of a search does not carry them. The message names the resume of the walk-forward of the chosen candidate instead.

# Validation

  - A search never takes a `Resume`. An `ArgumentError` is always thrown.

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

The stacked Result carries the `id` of `a` and the `opt` of `b`, which is the estimator that the later run threaded, so it can be resumed too. The two Results must abut in time. The check reads the timestamps of the stacked carriers, so a pair that overlaps and a pair from carriers without timestamps are refused by name.

# Algorithm

 1. Read the timestamps of the stacked carriers, giving `ta = a.mrd.ts` and `tb = b.mrd.ts`. Refuse when either is `nothing`.
 2. Refuse unless the last entry of `ta` is before the first entry of `tb`.
 3. Build a [`MultiPeriodPredictionResult`](@ref) from `vcat(a.pred, b.pred)`, the `id` of `a` and the `opt` of `b`. Its constructor stacks `mrd` again from the folds.

# Arguments

  - `a`: The earlier Result.
  - `b`: The later Result, a resume of `a`.

# Validation

  - Both carry timestamps, and `a` ends before `b` starts. An `ArgumentError` is thrown otherwise.

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
