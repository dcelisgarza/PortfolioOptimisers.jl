"""
    fit_fold_result(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult, train_idx::VecInt, cols)
    fit_fold_result(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult, ::Nothing, cols)

Fits the estimator of one fold. Dispatch on `train_idx` selects the arm.

These are the two arms of the estimator form of [`fit_and_predict`](@ref). A training window selects the refit that every fold of the batch arms runs, `optimise(opt, port_opt_view(rd, train_idx, cols))`. `nothing` means that the estimator holds its window, because the online arm of [`fold_loop`](@ref) folded every row of it. This arm reads the estimator out through `optimise(opt)` with no returns, and the read-out rebuilds the carrier from the state and runs the batch path over it. When `cols` is not `:`, the caller takes the asset view of `opt` before either arm, so it slices a stepped estimator and a cold one by asset in the same way.

# Arguments

  - `opt`: The estimator of the fold, already viewed by asset.
  - `rd`: The carrier of the fold. The read-out arm does not read it.
  - `train_idx`: The training window of the fold, or `nothing` for a stepped estimator.
  - `cols`: The asset columns of the fold. The read-out arm does not read them.

# Validation

  - On the read-out arm, everything that `optimise(opt)` refuses. A cold estimator holds no window, and the read-out refuses it with an `ArgumentError`.

# Returns

  - `res::OptimisationResult`: The fitted result of the fold.

# Related

  - [`fit_and_predict`](@ref)
  - [`online_folds`](@ref)
  - [`optimise`](@ref)
  - [`Fold`](@ref)
"""
function fit_fold_result(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                         train_idx::VecInt, cols)
    return optimise(opt, port_opt_view(rd, train_idx, cols))
end
function fit_fold_result(opt::NonFiniteAllocationOptimisationEstimator, ::ReturnsResult,
                         ::Nothing, ::Any)
    return optimise(opt)
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult;
                         train_idx::Option{<:VecInt} = nothing, test_idx::VecInt_VecVecInt,
                         cols = :, wd::Option{<:AbstractWeightDrift} = nothing,
                         hwd::Option{<:AbstractWeightDrift} = wd,
                         fa::Option{<:AbstractFeeAmortisation} = nothing,
                         store_weight_path::Bool = false, strict::Bool = false,
                         w_prev::Option{<:VecNum_VecVecNum} = nothing)
    if !isa(cols, Colon)
        opt = port_opt_view(opt, cols, rd.X)
    end
    #! Add ability to do callbacks
    res = fit_fold_result(opt, rd, train_idx, cols)
    return StatsAPI.predict(res, rd, test_idx, cols; wd = wd, hwd = hwd, fa = fa,
                            store_weight_path = store_weight_path, strict = strict,
                            w_prev = w_prev)
end
"""
    online_step_fold(est, ctx::Option{<:TimeDependentContext}, rd::Prices_RR)

Folds the rows that a fold gained into the estimator that the online arm threads. The step reads each of its schedules at the entry of that fold.

The default is [`partial_fit!`](@ref), and every family except one takes it. [`assert_stateless_schedule`](@ref) lets a schedule reach only a field that carries no state. On every other family the step reads none of those fields. The read-out of the fold reads them, and it resolves them itself. For example, the programme that the read-out solves reads a scheduled weight bound, and the buffer that the step fills never reads it. [`OnlinePortfolioSelection`](@ref) is the exception. Its step is the optimisation, because the step projects every row onto `set`, so the head has its own method.

`ctx` is `nothing` when the estimator carries no schedule, which is the usual case. The loop then builds no context.

# Arguments

  - `est`: The estimator that the loop threads, with its schedules unresolved.
  - `ctx`: The context of the fold, or `nothing`.
  - `rd`: The carrier of the rows that the fold gained. It holds returns, or the prices that a [`Pipeline`](@ref) host threads.

# Validation

  - Everything that [`partial_fit!`](@ref) refuses on `est` and `rd`.

# Returns

  - `est`: The estimator, with the rows of the fold folded in and its schedules unresolved.

# Related

  - [`thread_online_folds!`](@ref)
  - [`partial_fit!`](@ref)
  - [`assert_stateless_schedule`](@ref)
  - [`TimeDependentContext`](@ref)
"""
function online_step_fold(est, ::Option{<:TimeDependentContext}, rd::Prices_RR)
    return partial_fit!(est, rd)
end
"""
    one_previous_portfolio(est, w_prev::VecNum)
    one_previous_portfolio(est, w_prev::VecVecNum)

Returns the one portfolio that a fold reads as its previous weights, and refuses a population.

[`fold_loop`](@ref) passes the weights of the previous fold to [`factory`](@ref) when `est` [`needs_previous_weights`](@ref). A term that reads them measures the trade from one portfolio. A turnover, a tracking error, a fee, and a custom constraint or objective are such terms. A frontier sweep gives a population, with one portfolio per sweep point. Each fold resolves the span of its frontier from its own data, so point `k` at one fold and point `k` at the next fold sit at different return levels. No portfolio of the population is the previous one. So the loop refuses the combination by name, and it does not charge the term against a portfolio that it would have to choose. Fold 1 reads no previous weights, so the refusal comes after the solve of fold 1. The batch arms and the online arm refuse alike.

# Arguments

  - `est`: The estimator of the fold. The error names its type.
  - `w_prev`: The weights that the previous fold gives.

# Validation

  - `w_prev` is one portfolio. The second method throws an `ArgumentError`, because a population holds more than one.

# Returns

  - `w_prev::VecNum`: The weights, unchanged.

# Related

  - [`fold_loop`](@ref)
  - [`previous_weights`](@ref)
  - [`needs_previous_weights`](@ref)
  - [`Frontier`](@ref)
"""
function one_previous_portfolio(::Any, w_prev::VecNum)
    return w_prev
end
function one_previous_portfolio(est, w_prev::VecVecNum)
    return throw(ArgumentError("`$(typeof(est).name.name)` reads the previous fold's weights (`needs_previous_weights` is `true`), but the previous fold gave a population of $(length(w_prev)) portfolios, one per point of a frontier sweep. Each fold resolves its own frontier span from its own data, so a sweep point at one fold is not the same portfolio as the point with the same index at the next fold, and no one portfolio is the previous one. A frontier sweep and a term that reads the previous weights (a turnover, a tracking error, a fee, or a custom constraint or objective on the weights) contradict each other in a walk-forward. Optimise one portfolio per fold, or fix the previous weights of the term, for example `fixed = true` on a turnover, so that no fold reads them."))
end
"""
    fold_context(i::Integer, n::Integer, rd, train_idx, test_idx, w_prev, path_id)

Builds the [`TimeDependentContext`](@ref) of fold `i`. Both places that resolve a schedule against a fold read this record.

The per-fold copy of [`fold_loop`](@ref) builds it and swaps every schedule in before the callback fits the fold. The online arm builds the same record earlier in the same fold, before it folds the rows of the fold, for the schedules that its step reads ([`online_step_fold`](@ref)). Both places call this one constructor, so the step and the read-out of fold `i` resolve a schedule against equal records.

# Arguments

  - `i`: The index of the fold, in the enumeration of the scheme.
  - `n`: The number of folds that the scheme enumerates.
  - `rd`: The carrier of the fold, already viewed by asset where the scheme takes a view.
  - `train_idx`: The training windows of every fold, in split order.
  - `test_idx`: The test windows of every fold, in split order.
  - `w_prev`: The weights of the previous fold, or `nothing`.
  - `path_id`: The path that the fold belongs to, or `nothing`.

# Returns

  - `ctx::TimeDependentContext`: The context of the fold.

# Related

  - [`fold_loop`](@ref)
  - [`online_step_fold`](@ref)
  - [`TimeDependentContext`](@ref)
"""
function fold_context(i::Integer, n::Integer, rd, train_idx, test_idx, w_prev, path_id)
    return TimeDependentContext(; i = i, n = n, rd = rd, train_idx = train_idx,
                                test_idx = test_idx, w_prev = w_prev, path_id = path_id)
end
"""
    step_context(td::Bool, i::Integer, n::Integer, rd, train_idx, test_idx, w_prev, path_id)

Returns the context against which the online arm resolves the schedules that the step of fold `i` reads, or `nothing` when `td` is `false`.

A usual run carries no schedule. [`online_folds`](@ref) and [`thread_online_folds!`](@ref) each read [`is_time_dependent`](@ref) once per call, and not once per fold. The estimator that they thread keeps its schedules unresolved from fold to fold, so one answer holds for every fold.

# Arguments

  - `td`: Whether the threaded estimator carries a schedule.
  - `i`, `n`, `rd`, `train_idx`, `test_idx`, `w_prev`, `path_id`: As [`fold_context`](@ref) takes them.

# Returns

  - `ctx::Option{<:TimeDependentContext}`: The context of the fold, or `nothing`.

# Related

  - [`fold_context`](@ref)
  - [`online_step_fold`](@ref)
  - [`is_time_dependent`](@ref)
"""
function step_context(td::Bool, i::Integer, n::Integer, rd, train_idx, test_idx, w_prev,
                      path_id)
    return td ? fold_context(i, n, rd, train_idx, test_idx, w_prev, path_id) : nothing
end
"""
    thread_online_folds!(predictions, fit_fold, est, folds, prev; rd, train_idx, test_idx, n, last_end, pws, path_id)

Folds and reads out the folds `folds` of a walk-forward, and threads `est` from one fold to the next.

The online arm and the resumed arm of [`fold_loop`](@ref) share this body, so a resume continues the loop that started the run.

# Algorithm

 1. Read [`is_time_dependent`](@ref) on `est` once, giving `td_flag`.
 2. For each fold `i` in `folds`, take the last row of its training window, giving `stop`.
 3. When `stop > last_end`, build the context of the step through [`step_context`](@ref), with the weights that [`previous_weights`](@ref) reads from `prev`, giving `ctx`. Fold the rows `(last_end + 1):stop` of `rd` into `est` through [`online_step_fold`](@ref), and set `last_end` to `stop`. Otherwise the fold gained no row, which is the case of fold 1 after the warm-up, and `est` does not change.
 4. Call `fit_fold(i, prev, est, rd, nothing)`, which makes the per-fold copy as the batch arms make it, giving the prediction of fold `i`. Store it at `predictions[i - first(folds) + 1]`.
 5. Advance `prev` past the prediction through [`advance_previous_fold`](@ref).
 6. After the last fold, return `est`.

# Arguments

  - `predictions`: The vector that receives the predictions, with one slot per fold of `folds`.
  - `fit_fold`: The per-fold resolution and callback that [`fold_loop`](@ref) builds.
  - `est`: The estimator, folded through `last_end`.
  - `folds`: The indices of the folds to run, a contiguous range of the enumeration of the scheme.
  - `prev`: The prediction of the last threadable fold before `first(folds)`, or `nothing`.
  - `rd`: The carrier.
  - `train_idx`: The training windows of every fold, in split order.
  - `test_idx`: The test windows of every fold, in split order. The context of each fold reads them.
  - `n`: The number of folds that the scheme enumerates. The context of each fold reads it.
  - `last_end`: The last row that `est` has folded.
  - `pws`: The Previous-Weights Source of the scheme, or `nothing`.
  - `path_id`: The path that the folds belong to, or `nothing`. Only the context of a fold reads it.

# Returns

  - `est`: The estimator, folded through `last(train_idx[last(folds)])`, with its schedules unresolved.

# Related

  - [`online_folds`](@ref)
  - [`Resume`](@ref)
  - [`advance_previous_fold`](@ref)
"""
function thread_online_folds!(predictions, fit_fold, est, folds::AbstractUnitRange, prev;
                              rd, train_idx, test_idx, n::Integer, last_end::Integer, pws,
                              path_id = nothing)
    # A schedule reaches this arm only through the estimator the loop threads, which stays
    # unresolved from fold to fold, so the estimator itself answers whether to build one.
    td_flag = is_time_dependent(est)
    offset = first(folds) - 1
    for i in folds
        stop = last(train_idx[i])
        if stop > last_end
            ctx = step_context(td_flag, i, n, rd, train_idx, test_idx,
                               previous_weights(pws, prev), path_id)
            est = online_step_fold(est, ctx, port_opt_view(rd, (last_end + 1):stop, :))
            last_end = stop
        end
        predictions[i - offset] = fit_fold(i, prev, est, rd, nothing)
        prev = advance_previous_fold(pws, prev, predictions[i - offset])
    end
    return est
end
"""
    online_folds(fit_fold, est, n::Integer, ::Type{ElT}, path_id = nothing; rd, train_idx, test_idx, fold_view, pws)

Runs the `n` folds of a walk-forward through the online step, and threads one estimator from fold to fold.

This is the online arm of [`fold_loop`](@ref), which the loop takes when the scheme is an Online Scheme ([`folds_are_stepped`](@ref)). The arm logs no message. The caller chooses an Online Scheme by name through its constructor, so the arm has no accident to report. [`run_folds`](@ref) is different, because it reports a sequential run that the loop chose on its own. A [`Resume`](@ref) in the estimator slot takes the method of `Resume`, which skips the warm-up and the folds that the Result holds, and runs the same body from the fold after them.

Fold for fold, the run reaches the weights of the batch expanding walk-forward, to the tolerance of the moment layer and of the solver. A purged scheme folds a row when the training window reaches it, and it never drops a row. So fold `i` has folded the rows `1:last(train_idx[i])`, which are the rows that the batch expanding fold reads, and the read-out rebuilds the training window of the batch fold.

# Algorithm

 1. Refuse, through [`assert_online_entry`](@ref), an estimator that is not the configuration alone. The check refuses a partial-fit state at entry, because the loop starts cold. It also refuses a [`TimeDependent`](@ref) schedule on a field that carries a state, because a schedule replaces the value through which the loop threads the state.
 2. Refuse, through [`assert_online_fee_source`](@ref), an [`OnlinePortfolioSelection`](@ref) head whose fees carry a turnover term when the scheme threads no Previous-Weights Source.
 3. Under a multiple-randomised path, take the asset view of the path through `fold_view(1)`, giving `est` and `rd`. A path is one asset subset crossed with the folds of the walk-forward, and a view slices a state by asset, so the arm threads the sliced estimator.
 4. Resolve every [`Online`](@ref) wrapper through [`update_online_estimator`](@ref), giving `est`.
 5. Build the context of fold 1 through [`step_context`](@ref), giving `ctx`. Fold the first training window `train_idx[1]` into `est` through [`online_step_fold`](@ref). This is the warm-up. The loop folds no more rows for fold 1, so the warm-up is the step that runs inside the schedule entries of fold 1.
 6. Run the folds `1:n` through [`thread_online_folds!`](@ref) from `last_end = last(train_idx[1])`, giving `predictions` and the threaded `est`.

# Arguments

  - `fit_fold`: The per-fold resolution and callback that [`fold_loop`](@ref) builds. It takes `(i, prev, est, rd, train)`: the index of the fold, the prediction of the last threadable fold or `nothing`, the estimator to resolve, the carrier, and the training window, which is `nothing` here. The arm advances `prev` through [`advance_previous_fold`](@ref), as [`run_folds`](@ref) advances it, so a failed step does not move the previous weights.
  - `est`: The estimator, as the caller gave it to the loop.
  - `n`: The number of folds.
  - `ElT`: The element type of the per-fold result. It is positional for the reason that [`parallel_folds`](@ref) gives.
  - `path_id`: The path that the folds belong to, or `nothing`. Only the context of a fold reads it.
  - `rd`: The carrier.
  - `train_idx`: The training windows of every fold, in split order.
  - `test_idx`: The test windows of every fold, in split order. The context of each fold reads them.
  - `fold_view`: The asset view of a multiple-randomised path, or `nothing`.
  - `pws`: The Previous-Weights Source of the scheme, or `nothing`. It decides what [`threads_weights`](@ref) tests, and [`assert_online_fee_source`](@ref) reads it.

# Validation

  - Everything that [`assert_online_entry`](@ref) and [`assert_online_fee_source`](@ref) refuse.

# Returns

  - `predictions::Vector{ElT}`: One prediction per fold, in split order.
  - `est`: The threaded estimator, folded through `last(train_idx[n])`, for the Result to carry.

# Related

  - [`fold_loop`](@ref)
  - [`thread_online_folds!`](@ref)
  - [`online_step_fold`](@ref)
  - [`run_folds`](@ref)
  - [`parallel_folds`](@ref)
  - [`assert_online_entry`](@ref)
  - [`update_online_estimator`](@ref)
  - [`partial_fit!`](@ref)
  - [`fit_and_predict`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
  - [`Resume`](@ref)
"""
function online_folds(fit_fold, est, n::Integer, ::Type{ElT}, path_id = nothing; rd,
                      train_idx, test_idx = nothing, fold_view = nothing,
                      pws = nothing) where {ElT}
    assert_online_entry(est)
    assert_online_fee_source(est, pws)
    (est, rd) = isnothing(fold_view) ? (est, rd) : fold_view(1)
    est = update_online_estimator(est)
    # The warm-up is fold 1's training window, and the loop below folds nothing more for
    # fold 1, so the warm-up is the step that must run inside fold 1's own schedule entry.
    ctx = step_context(is_time_dependent(est), 1, n, rd, train_idx, test_idx,
                       previous_weights(pws, nothing), path_id)
    est = online_step_fold(est, ctx, port_opt_view(rd, train_idx[1], :))
    predictions = Vector{ElT}(undef, n)
    est = thread_online_folds!(predictions, fit_fold, est, 1:n, nothing; rd = rd,
                               train_idx = train_idx, test_idx = test_idx, n = n,
                               last_end = last(train_idx[1]), pws = pws, path_id = path_id)
    return predictions, est
end
