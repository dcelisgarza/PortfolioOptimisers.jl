"""
    fit_fold_result(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult, train_idx::VecInt, cols)
    fit_fold_result(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult, ::Nothing, cols)

The fit of one fold, by whether the fold carries a training window.

The two arms of [`fit_and_predict`](@ref)'s estimator form, chosen by dispatch on `train_idx`. A window fits the estimator over it, `optimise(opt, port_opt_view(rd, train_idx, cols))`, which is the refit every fold of the batch arms runs. `nothing` says *the estimator holds its window* — the online arm of [`fold_loop`](@ref) has already folded every row of it — so the fold reads the estimator out through `optimise(opt)` with no returns, and the read-out rebuilds the carrier from the state and runs the ordinary batch path over it (ADR 0137). The asset view, when `cols` is not `:`, is taken by the caller before either arm, so a stepped estimator is sliced by asset as a cold one is.

# Arguments

  - `opt`: The fold's estimator, asset-viewed by the caller.
  - `rd`: The carrier the fold reads.
  - `train_idx`: The fold's training window, or `nothing` for a stepped estimator.
  - `cols`: The fold's asset columns.

# Returns

  - `res::OptimisationResult`: The fold's fitted result.

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
    cv_online_info()

Build the informational message emitted when a cross-validation run takes the online arm.
[`online_folds`](@ref) is the only site that emits it, as [`run_folds`](@ref) is the only site
that emits [`cv_sequential_info`](@ref), and it states the one fact that sent the run there:
the scheme declares a Fold Fit.

# Returns

  - `msg::String`: The message.

# Related

  - [`online_folds`](@ref)
  - [`cv_sequential_info`](@ref)
  - [`fold_fit`](@ref)
  - [`OnlineStep`](@ref)
"""
function cv_online_info()
    return "Running cross-validation online because the scheme declares a Fold Fit (fold_fit(cv) == OnlineStep()). The loop warms one estimator up on the first training window, folds each fold's new observations into it, and reads it out where a refit would have run, so the folds run in order and the estimator is threaded from fold to fold. To refit every fold from its training window, and to run the folds in parallel where the optimiser allows it, leave the scheme's `ff` unset."
end
"""
    thread_online_folds!(predictions, fit_fold, est, folds, prev; rd, train_idx, last_end, pws)

Fold and read out the folds `folds` of a walk-forward, threading `est` from one to the next.

The per-fold body the online arm and the resumed arm of [`fold_loop`](@ref) share, written once so that a resume continues exactly the loop that started the run. For each fold `i` in `folds`, it folds the rows the training window has gained since the last fold, `(last_end + 1):last(train_idx[i])`, into `est` with [`partial_fit!`](@ref); hands the threaded estimator to `fit_fold(i, prev, est, rd, nothing)`, which makes the per-fold copy exactly as the batch arms do; stores the prediction at `predictions[i - first(folds) + 1]`; and advances `prev` by [`advance_previous_fold`](@ref).

# Arguments

  - `predictions`: The vector the predictions are written into, one slot per fold of `folds`.
  - `fit_fold`: The per-fold resolution and callback [`fold_loop`](@ref) builds.
  - `est`: The estimator, folded through `last_end`.
  - `folds`: The fold indices to run, a contiguous range of the scheme's enumeration.
  - `prev`: The last threadable fold's prediction before `first(folds)`, or `nothing`.
  - `rd`: The carrier.
  - `train_idx`: The training windows of every fold, in split order.
  - `last_end`: The last row `est` has folded.
  - `pws`: The scheme's Previous-Weights Source, or `nothing`.

# Returns

  - `est`: The estimator, folded through `last(train_idx[last(folds)])`.

# Related

  - [`online_folds`](@ref)
  - [`Resume`](@ref)
  - [`advance_previous_fold`](@ref)
"""
function thread_online_folds!(predictions, fit_fold, est, folds::AbstractUnitRange, prev;
                              rd, train_idx, last_end::Integer, pws)
    offset = first(folds) - 1
    for i in folds
        stop = last(train_idx[i])
        if stop > last_end
            est = partial_fit!(est, port_opt_view(rd, (last_end + 1):stop, :))
            last_end = stop
        end
        predictions[i - offset] = fit_fold(i, prev, est, rd, nothing)
        prev = advance_previous_fold(pws, prev, predictions[i - offset])
    end
    return est
end
"""
    online_folds(fit_fold, est, n::Integer, ::Type{ElT}; rd, train_idx, test_idx, fold_view, pws)

Run `n` folds of a walk-forward by the online step, threading one estimator from fold to
fold, and emit [`cv_online_info`](@ref).

This is the third arm of [`fold_loop`](@ref), taken when the scheme declares a Fold Fit
([`fold_fit`](@ref)). It does five things, in order.

 1. It refuses an estimator that is not the configuration alone, through
    [`assert_online_entry`](@ref): one carrying a partial-fit state at entry, because the
    loop starts cold, and one carrying a [`TimeDependent`](@ref) schedule on a field that
    carries a state, because a schedule replaces the value a state is threaded through. Both
    are refused by name before any solve.
 2. It warms up once. Under a multiple-randomised path it takes the path's asset view
    through `fold_view(1)` here, because a path is one asset subset crossed with the
    walk-forward's folds, and the sliced estimator is what it threads (ADR 0107: a view
    slices a state by asset). It then resolves every [`Online`](@ref) wrapper through
    [`update_online_estimator`](@ref), and folds the first training window `train_idx[1]`
    into the estimator with [`partial_fit!`](@ref).
 3. Per fold `i`, it folds the rows the training window has gained since the last fold,
    `(last_end + 1):last(train_idx[i])`, into the threaded estimator. A purged scheme folds
    a row when the training window reaches it and never drops one, so fold `i` has folded
    exactly `1:last(train_idx[i])`, which are the rows the batch expanding fold reads.
 4. It hands the threaded estimator to `fit_fold`, which makes the per-fold copy exactly as
    the batch arms do — schedules resolved against the fold's [`TimeDependentContext`](@ref),
    the previous fold's weights threaded through [`factory`](@ref) — and calls the callback
    with a [`Fold`](@ref) whose `train` is `nothing`.
 5. It stores the prediction and threads the estimator on.

Steps 3 to 5 are [`thread_online_folds!`](@ref), which the resumed arm shares: a
[`Resume`](@ref) in the estimator slot takes the method below, which skips the warm-up and
the folds the Result holds and runs the same body from the fold after them.

The identity the arm keeps is the seam's: the run reaches the weights of the batch expanding
walk-forward fold for fold, to the tolerance of the moment layer and of the solver, and the
carrier the read-out rebuilds is exactly the batch fold's training window.

`fit_fold` takes `(i, prev, est, rd, train)`: the fold index, the last threadable fold's
prediction or `nothing` — advanced by [`threads_weights`](@ref) exactly as [`run_folds`](@ref)
advances it, so a failed step leaves the previous weights where they were — the estimator to
resolve, the carrier, and the training window, which is `nothing` here. `ElT` is the per-fold result element type, positional for the reason given in
[`parallel_folds`](@ref).

# Arguments

  - `fit_fold`: The per-fold resolution and callback [`fold_loop`](@ref) builds.
  - `est`: The estimator, as the caller handed it to the loop.
  - `n`: The number of folds.
  - `rd`: The carrier.
  - `train_idx`: The training windows of every fold, in split order.
  - `test_idx`: The test windows of every fold, in split order. Unread here; the resumed arm checks the last held fold against it.
  - `fold_view`: The asset view of a multiple-randomised path, or `nothing`.
  - `pws`: The scheme's Previous-Weights Source, or `nothing`; decides what [`threads_weights`](@ref) tests.

# Validation

  - Everything [`assert_online_entry`](@ref) refuses.

# Returns

  - `predictions::Vector{ElT}`: One prediction per fold, in split order.
  - `est`: The threaded estimator, folded through `last(train_idx[n])`, for the Result to carry (ADR 0144).

# Related

  - [`fold_loop`](@ref)
  - [`thread_online_folds!`](@ref)
  - [`run_folds`](@ref)
  - [`parallel_folds`](@ref)
  - [`assert_online_entry`](@ref)
  - [`update_online_estimator`](@ref)
  - [`partial_fit!`](@ref)
  - [`fit_and_predict`](@ref)
  - [`OnlineStep`](@ref)
  - [`Resume`](@ref)
"""
function online_folds(fit_fold, est, n::Integer, ::Type{ElT}; rd, train_idx,
                      test_idx = nothing, fold_view = nothing, pws = nothing) where {ElT}
    @info(cv_online_info())
    assert_online_entry(est)
    (est, rd) = isnothing(fold_view) ? (est, rd) : fold_view(1)
    est = update_online_estimator(est)
    est = partial_fit!(est, port_opt_view(rd, train_idx[1], :))
    predictions = Vector{ElT}(undef, n)
    est = thread_online_folds!(predictions, fit_fold, est, 1:n, nothing; rd = rd,
                               train_idx = train_idx, last_end = last(train_idx[1]),
                               pws = pws)
    return predictions, est
end
