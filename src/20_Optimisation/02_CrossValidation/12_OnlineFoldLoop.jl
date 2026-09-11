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
                         store_weight_path::Bool = false, strict::Bool = false)
    if !isa(cols, Colon)
        opt = port_opt_view(opt, cols, rd.X)
    end
    #! Add ability to do callbacks
    res = fit_fold_result(opt, rd, train_idx, cols)
    return StatsAPI.predict(res, rd, test_idx, cols; wd = wd, hwd = hwd, fa = fa,
                            store_weight_path = store_weight_path, strict = strict)
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
    online_folds(fit_fold, est, n::Integer, ::Type{ElT}; rd, train_idx, fold_view)

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

The identity the arm keeps is the seam's: the run reaches the weights of the batch expanding
walk-forward fold for fold, to the tolerance of the moment layer and of the solver, and the
carrier the read-out rebuilds is exactly the batch fold's training window.

`fit_fold` takes `(i, prev, est, rd, train)`: the fold index, the previous fold's prediction
or `nothing`, the estimator to resolve, the carrier, and the training window, which is
`nothing` here. `ElT` is the per-fold result element type, positional for the reason given in
[`parallel_folds`](@ref).

# Arguments

  - `fit_fold`: The per-fold resolution and callback [`fold_loop`](@ref) builds.
  - `est`: The estimator, as the caller handed it to the loop.
  - `n`: The number of folds.
  - `rd`: The carrier.
  - `train_idx`: The training windows of every fold, in split order.
  - `fold_view`: The asset view of a multiple-randomised path, or `nothing`.

# Validation

  - Everything [`assert_online_entry`](@ref) refuses.

# Returns

  - `predictions::Vector{ElT}`: One prediction per fold, in split order.

# Related

  - [`fold_loop`](@ref)
  - [`run_folds`](@ref)
  - [`parallel_folds`](@ref)
  - [`assert_online_entry`](@ref)
  - [`update_online_estimator`](@ref)
  - [`partial_fit!`](@ref)
  - [`fit_and_predict`](@ref)
  - [`OnlineStep`](@ref)
"""
function online_folds(fit_fold, est, n::Integer, ::Type{ElT}; rd, train_idx,
                      fold_view = nothing) where {ElT}
    @info(cv_online_info())
    assert_online_entry(est)
    (est, rd) = isnothing(fold_view) ? (est, rd) : fold_view(1)
    est = update_online_estimator(est)
    est = partial_fit!(est, port_opt_view(rd, train_idx[1], :))
    last_end = last(train_idx[1])
    predictions = Vector{ElT}(undef, n)
    for i in 1:n
        stop = last(train_idx[i])
        if stop > last_end
            est = partial_fit!(est, port_opt_view(rd, (last_end + 1):stop, :))
            last_end = stop
        end
        predictions[i] = fit_fold(i, i > 1 ? predictions[i - 1] : nothing, est, rd, nothing)
    end
    return predictions
end
"""
    assert_batch_fold_fit(cv, site::AbstractString, issue::AbstractString)

Refuse a scheme that declares a Fold Fit at an entry point that does not take the online step yet.

The search scores every fold of a contiguous scheme independently, and the Pipeline decides what a fold with no training window means to its `fit`; neither reaches the online arm of [`fold_loop`](@ref) today, and each is its own ticket. Until it lands, a scheme carrying an [`OnlineStep`](@ref) is refused at the door by name rather than run as a refit in silence, because a caller who declared the step would otherwise read a batch answer as an online one.

# Arguments

  - `cv`: The scheme.
  - `site`: The entry point, as the message names it.
  - `issue`: The ticket that builds the step there.

# Related

  - [`fold_fit`](@ref)
  - [`OnlineStep`](@ref)
  - [`search_cross_validation`](@ref)
  - [`cross_val_predict`](@ref)
"""
function assert_batch_fold_fit(cv, site::AbstractString, issue::AbstractString)
    @argcheck(isnothing(fold_fit(cv)),
              ArgumentError("$(site) does not take the online step yet ($(issue)): the scheme declares `ff = $(nameof(typeof(fold_fit(cv))))()`, and this entry point refits every fold from its training window. Leave the scheme's `ff` unset here, or run the scheme through `fit_and_predict(opt, rd, cv)`."))
    return nothing
end
