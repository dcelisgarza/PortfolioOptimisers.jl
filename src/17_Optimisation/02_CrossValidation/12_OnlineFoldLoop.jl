"""
    fit_fold_result(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult, train_idx::VecInt, cols)
    fit_fold_result(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult, ::Nothing, cols)

The fit of one fold, by whether the fold carries a training window.

The two arms of [`fit_and_predict`](@ref)'s estimator form, chosen by dispatch on `train_idx`. A window fits the estimator over it, `optimise(opt, port_opt_view(rd, train_idx, cols))`, which is the refit every fold of the batch arms runs. `nothing` says *the estimator holds its window* — the online arm of [`fold_loop`](@ref) has already folded every row of it — so the fold reads the estimator out through `optimise(opt)` with no returns, and the read-out rebuilds the carrier from the state and runs the ordinary batch path over it. The asset view, when `cols` is not `:`, is taken by the caller before either arm, so a stepped estimator is sliced by asset as a cold one is.

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
    online_step_fold(est, ctx::Option{<:TimeDependentContext}, rd::Prices_RR)

Folds a fold's rows into the threaded estimator, inside the fold's own values of the schedules its *step* reads.

The default is [`partial_fit!`](@ref), and it is the answer for every family but one. A schedule reaches stateless fields only ([`assert_stateless_schedule`](@ref)), and on every other family those fields are read by the fold's read-out, which resolves them itself: a scheduled weight bound is read by the programme the read-out solves, never by the buffer the step fills. [`OnlinePortfolioSelection`](@ref) is the exception, because its step *is* the optimisation — every row is projected onto `set` — so it takes its own method.

`ctx` is `nothing` when the estimator carries no schedule at all, which is the ordinary case, and the loop then never builds a context it would not read.

# Arguments

  - `est`: The estimator the loop threads, with its schedules unresolved.
  - `ctx`: The fold's context, or `nothing`.
  - `rd`: The carrier of the rows the fold has gained: returns, or the prices a [`Pipeline`](@ref) host threads.

# Returns

  - `est`: The estimator, with the fold's rows folded in and its schedules unresolved.

# Related

  - [`thread_online_folds!`](@ref)
  - [`partial_fit!`](@ref)
  - [`assert_stateless_schedule`](@ref)
  - [`TimeDependentContext`](@ref)
"""
function online_step_fold(est, ::Nothing, rd::Prices_RR)
    return partial_fit!(est, rd)
end
function online_step_fold(est, ::TimeDependentContext, rd::Prices_RR)
    return partial_fit!(est, rd)
end
"""
    one_previous_portfolio(est, w_prev::VecNum)
    one_previous_portfolio(est, w_prev::VecVecNum)

Give the one portfolio a fold reads as its previous weights, and refuse a population.

[`fold_loop`](@ref) hands the previous fold's weights to [`factory`](@ref) when `est` [`needs_previous_weights`](@ref). A term that reads them, such as a turnover, a tracking error, a fee, or a custom constraint or objective, measures the trade from **one** portfolio. A frontier sweep gives a population, one portfolio per sweep point, and each fold resolves its own span from its own data. So point `k` at one fold and point `k` at the next sit at different return levels, and no one portfolio is the previous one. The two ideas contradict each other, so the loop refuses the combination by name instead of charging the term against a portfolio it would have to choose. The first fold reads no previous weights, so the refusal comes after its solve.

# Arguments

  - `est`: The estimator of the fold, named in the error.
  - `w_prev`: The weights the previous fold gives.

# Returns

  - `w_prev::VecNum`: The weights, unchanged, when they are one portfolio.

# Throws

  - `ArgumentError` when `w_prev` is a population.

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

The [`TimeDependentContext`](@ref) of fold `i`, built once here and read by both places that resolve a schedule against a fold.

[`fold_loop`](@ref)'s per-fold copy builds it to swap every schedule in before the callback fits the fold. The online arm builds the same record one fold earlier, before the fold's rows are folded, for the schedules its step reads ([`online_step_fold`](@ref)). Writing the record once keeps the two readings of fold `i` identical, which is what the arm's equality with the batch arm rests on.

# Arguments

  - `i`: The fold index, in the scheme's enumeration.
  - `n`: The number of folds the scheme enumerates.
  - `rd`: The carrier the fold reads, already asset-viewed where the scheme takes a view.
  - `train_idx`: The training windows of every fold, in split order.
  - `test_idx`: The test windows of every fold, in split order.
  - `w_prev`: The previous fold's weights, or `nothing`.
  - `path_id`: The path the fold belongs to, or `nothing`.

# Returns

  - `ctx::TimeDependentContext`: The fold's context.

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

The context the online arm's step resolves fold `i`'s schedules against, or `nothing` when `td` says there is no schedule to resolve.

The ordinary run carries none, and the arm asks once per run rather than once per fold: [`online_folds`](@ref) and [`thread_online_folds!`](@ref) read [`is_time_dependent`](@ref) on the estimator they thread, which stays unresolved from fold to fold and therefore answers for every fold alike.

# Arguments

  - `td`: Whether the threaded estimator carries a schedule.
  - `i`, `n`, `rd`, `train_idx`, `test_idx`, `w_prev`, `path_id`: As [`fold_context`](@ref) takes them.

# Returns

  - `ctx::Option{<:TimeDependentContext}`: The fold's context, or `nothing`.

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

Fold and read out the folds `folds` of a walk-forward, threading `est` from one to the next.

The per-fold body the online arm and the resumed arm of [`fold_loop`](@ref) share, written once so that a resume continues exactly the loop that started the run. For each fold `i` in `folds`, it folds the rows the training window has gained since the last fold, `(last_end + 1):last(train_idx[i])`, into `est` through [`online_step_fold`](@ref), which takes the step inside fold `i`'s own values of the schedules the step reads; hands the threaded estimator to `fit_fold(i, prev, est, rd, nothing)`, which makes the per-fold copy exactly as the batch arms do; stores the prediction at `predictions[i - first(folds) + 1]`; and advances `prev` by [`advance_previous_fold`](@ref).

# Arguments

  - `predictions`: The vector the predictions are written into, one slot per fold of `folds`.
  - `fit_fold`: The per-fold resolution and callback [`fold_loop`](@ref) builds.
  - `est`: The estimator, folded through `last_end`.
  - `folds`: The fold indices to run, a contiguous range of the scheme's enumeration.
  - `prev`: The last threadable fold's prediction before `first(folds)`, or `nothing`.
  - `rd`: The carrier.
  - `train_idx`: The training windows of every fold, in split order.
  - `last_end`: The last row `est` has folded.
  - `test_idx`: The test windows of every fold, in split order, for the fold's context.
  - `n`: The number of folds the scheme enumerates, for the fold's context.
  - `pws`: The scheme's Previous-Weights Source, or `nothing`.
  - `path_id`: The path the folds belong to, or `nothing`, for the fold's context.

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
    online_folds(fit_fold, est, n::Integer, ::Type{ElT}, path_id; rd, train_idx, test_idx, fold_view, pws)

Run `n` folds of a walk-forward by the online step, threading one estimator from fold to
fold.

This is the third arm of [`fold_loop`](@ref), taken when the scheme is an Online Scheme
([`folds_are_stepped`](@ref)). It announces nothing: the scheme is chosen by name through its
constructor, so there is no accident to report, where [`run_folds`](@ref) reports a
sequential run the loop took on its own. It does five things, in order.

 1. It refuses an estimator that is not the configuration alone, through
    [`assert_online_entry`](@ref): one carrying a partial-fit state at entry, because the
    loop starts cold, and one carrying a [`TimeDependent`](@ref) schedule on a field that
    carries a state, because a schedule replaces the value a state is threaded through. Both
    are refused by name before any solve. It then refuses, through
    [`assert_online_fee_source`](@ref), an [`OnlinePortfolioSelection`](@ref) head whose
    fees carry a turnover term while the scheme threads no Previous-Weights Source.
 2. It warms up once. Under a multiple-randomised path it takes the path's asset view
    through `fold_view(1)` here, because a path is one asset subset crossed with the
    walk-forward's folds, and the sliced estimator is what it threads (a view
    slices a state by asset). It then resolves every [`Online`](@ref) wrapper through
    [`update_online_estimator`](@ref), and folds the first training window `train_idx[1]`
    into the estimator through [`online_step_fold`](@ref), inside fold 1's own values of
    the schedules the step reads.
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
  - `path_id`: The path the folds belong to, or `nothing`; read by the fold's context alone.
  - `pws`: The scheme's Previous-Weights Source, or `nothing`; decides what [`threads_weights`](@ref) tests.

# Validation

  - Everything [`assert_online_entry`](@ref) and [`assert_online_fee_source`](@ref) refuse.

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
