"""
    fit_predict(opt::OptE_Opt, rd::ReturnsResult)

Fit optimisation estimator `opt` on returns data `rd`, and predict over the same data.

The prediction is in-sample, so it measures the fit and not a backtest. Use
[`fit_and_predict`](@ref) or [`cross_val_predict`](@ref) for an out-of-sample series.

# Algorithm

 1. Fit `opt` on `rd` with [`optimise`](@ref), giving `res`.
 2. Predict `res` over the whole of `rd` with [`predict`](@ref).

# Arguments

  - `opt`: Optimisation estimator or result.
  - `rd::ReturnsResult`: Returns data.

# Returns

  - [`PredictionResult`](@ref).

# Related

  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`PredictionResult`](@ref)
  - [`fit_and_predict`](@ref)
"""
function fit_predict(opt::OptE_Opt, rd::ReturnsResult)
    res = optimise(opt, rd)
    return StatsAPI.predict(res, rd)
end
function StatsAPI.predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                          test_idx::VecInt, cols = :;
                          wd::Option{<:AbstractWeightDrift} = nothing,
                          hwd::Option{<:AbstractWeightDrift} = wd,
                          fa::Option{<:AbstractFeeAmortisation} = nothing,
                          store_weight_path::Bool = false, strict::Bool = false,
                          w_prev::Option{<:VecNum_VecVecNum} = nothing)
    rdi = port_opt_view(rd, test_idx, cols)
    fees = extract_fees(res, nothing)
    # The mask view and the Held Gap filter, in that order — see the whole-sample method.
    imsk = result_investable_mask(res)
    w, rdi, fees = investable_fold_view(imsk, res.w, rdi, fees)
    fees = override_fee_amortisation(fees, fa)
    obs = drift_observations(rdi.ts, test_idx)
    Xf = filter_held_gaps(w, rdi.X, strict; nx = rdi.nx)
    X = calc_net_returns(w, Xf, fees, wd, obs)
    w0 = held_start_weights(res.retcode, w, investable_weights_view(imsk, w_prev))
    (hw, ruined) = held_weights_result(hwd, w0, Xf, store_weight_path, obs)
    warn_ruined_members(wd, ruined, length(res.w))
    res = mark_ruined_members(res, ruined)
    rdi = reconstruct_rd(res, rdi, X, hw, w)
    return PredictionResult(; res = res, rd = rdi, hw = expand_held_weights(imsk, hw))
end
function StatsAPI.predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                          test_idxs::VecVecInt, cols = :; kwargs...)
    return [StatsAPI.predict(res, rd, test_idx, cols; kwargs...) for test_idx in test_idxs]
end
"""
    fit_and_predict(opt, rd::ReturnsResult, cv::NonSeqCVER; cols, ex, id) -> MultiPeriodPredictionResult
    fit_and_predict(opt, rd::ReturnsResult, cv::CombCVER; cols, ex) -> PopulationPredictionResult
    fit_and_predict(opt, rd::ReturnsResult; train_idx = nothing, test_idx, cols) -> PredictionResult
    fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult; test_idx, cols) -> PredictionResult

Fit an optimisation estimator on training data and predict on test data using cross-validation.

The method over a scheme `cv` predicts over every fold of `cv`. The methods without a scheme predict over one train/test split, or apply a result that is already fitted.

A combinatorial `cv` takes its own method, because its folds recombine into several paths and not one. That method regroups the fold predictions by path, and returns one [`MultiPeriodPredictionResult`](@ref) per path in a [`PopulationPredictionResult`](@ref).

The estimator form reads `train_idx = nothing` as *the estimator holds its window*. It reads the estimator out through `optimise(opt)` in place of a fit over `port_opt_view(rd, train_idx, cols)`, and predicts over `test_idx` as before. The online arm of [`fold_loop`](@ref) reads a fold out this way. It is also the public entry for an estimator stepped by hand, warmed up with [`update_online_estimator`](@ref) and folded with [`partial_fit!`](@ref). `fit_and_predict(opt, rd; test_idx)` on a stepped estimator equals `fit_and_predict(opt, rd; train_idx, test_idx)` on the cold one over the same rows. The two arms belong to [`fit_fold_result`](@ref), and the method is defined beside them.

The result form is fitted already, so it ignores `train_idx` and every keyword that [`predict`](@ref) does not take. The fold loop passes the same keywords to an estimator and to a result.

# Algorithm

For a scheme `cv` that is not combinatorial:

 1. Split `rd` with `cv`, giving `train_idx` and `test_idx`.
 2. Check that the folds are not shuffled with [`assert_unshuffled_folds`](@ref).
 3. Read the evaluation settings of `cv` with [`fold_evaluation`](@ref), and the drift of the Held Weights record with [`held_weights_drift`](@ref).
 4. Run the folds with [`fold_loop`](@ref). Each fold fits its estimator on its training rows and predicts over its test rows, giving one [`PredictionResult`](@ref) per fold, in split order.
 5. Concatenate the folds into a [`MultiPeriodPredictionResult`](@ref) carrying `id`.

# Arguments

  - `opt`: Optimisation estimator or an existing optimisation result.
  - `rd::ReturnsResult`: Returns data.
  - `cv::NonSeqCVER`: Non-sequential cross-validation estimator or result, such as [`KFold`](@ref).
  - `cv::CombCVER`: Combinatorial cross-validation estimator or result ([`CombinatorialCrossValidation`](@ref)).

# Keyword Arguments

  - `train_idx::Option{<:VecInt}`: Training indices, or `nothing` to read a stepped estimator out.
  - `test_idx`: Test indices, one vector or one vector per fold.
  - `cols = :`: Column selector.
  - `ex::FLoops.Transducers.Executor = FLoops.ThreadedEx()`: Executor of the folds that run in parallel.
  - `id = nothing`: Identifier of the path.
  - `wd`, `hwd`, `fa`, `store_weight_path`, `strict`, `w_prev`: The keywords of [`predict`](@ref), which the methods over one split pass on.

# Returns

  - [`MultiPeriodPredictionResult`](@ref), [`PopulationPredictionResult`](@ref), or [`PredictionResult`](@ref).

# Related

  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`optimise`](@ref)
  - [`fit_fold_result`](@ref)
  - [`KFold`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
function fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult;
                         test_idx::VecInt_VecVecInt, cols = :,
                         wd::Option{<:AbstractWeightDrift} = nothing,
                         hwd::Option{<:AbstractWeightDrift} = wd,
                         fa::Option{<:AbstractFeeAmortisation} = nothing,
                         store_weight_path::Bool = false, strict::Bool = false,
                         w_prev::Option{<:VecNum_VecVecNum} = nothing, kwargs...)
    return StatsAPI.predict(res, rd, test_idx, cols; wd = wd, hwd = hwd, fa = fa,
                            store_weight_path = store_weight_path, strict = strict,
                            w_prev = w_prev)
end
"""
    sort_predictions(res::VecVecInt, predictions::VecPredRes) -> VecPredRes
    sort_predictions(res::CrossValidationResult, predictions::VecPredRes) -> VecPredRes

Sort the prediction results of the folds by the first observation of their test windows.

The function returns a sorted copy, and does not change `predictions`. The key is the first entry of each test window, so the folds come back in time order when each window starts at its earliest row, which is true of every scheme in the library.

[`CombinatorialCrossValidationResult`](@ref) has its own method with a different shape and a different job. Its folds do not form one timeline, so that method takes a vector of per-split vectors and regroups them by path into [`MultiPeriodPredictionResult`](@ref)s.

# Algorithm

 1. Check that every test window holds unique indices.
 2. Compute the permutation that sorts the test windows by their first entry, giving `idx`.
 3. Return `predictions[idx]`.

# Arguments

  - `res`:

      + `::VecVecInt`: Vector of test index vectors.
      + `::CrossValidationResult`: Cross validation result object, uses the test indices stored in `res.test_idx`.

  - `predictions`: Vector of prediction results, one per fold, in split order.

# Validation

  - Every test window holds unique indices, else an `ArgumentError` is raised.

# Returns

  - A new vector of the predictions, sorted.

# Related

  - [`fit_and_predict`](@ref)
  - [`path_fit_and_predict`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
"""
function sort_predictions(test_idx::VecVecInt, predictions::VecPredRes)
    @argcheck(all(x -> allunique(x), test_idx), "Test indices must be unique.")
    idx = sortperm(test_idx; by = x -> x[1])
    return predictions[idx]
end
function sort_predictions(res::CrossValidationResult, predictions::VecPredRes)
    return sort_predictions(res.test_idx, predictions)
end
"""
    cv_sequential_info()

Build the informational message emitted when a cross-validation run runs its folds
sequentially. [`run_folds`](@ref) is the only site that emits it, and it is the sequential
loop, so the message states the two facts that sent the run there rather than quoting a
value back.

The two facts are the conjunction [`fold_loop`](@ref) computes. The fold enumeration of the
scheme is a timeline ([`folds_are_time_ordered`](@ref)), and the estimator needs the
previous fold's weights ([`needs_previous_weights`](@ref)). Either one alone leaves the
folds independent. Time dependence is neither of them: a [`TimeDependent`](@ref) schedule is
known for every fold before the loop starts, so [`fold_loop`](@ref) resolves it in parallel.

# Returns

  - `String`: The message.

# Related

  - [`run_folds`](@ref)
  - [`fold_loop`](@ref)
  - [`folds_are_time_ordered`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function cv_sequential_info()
    return "Running cross-validation sequentially because the folds of the cross-validation scheme are a timeline (folds_are_time_ordered(cv) == true) and the optimiser must use the previous optimisation's weights (needs_previous_weights(opt) == true). The second fact is because somewhere within the optimisation estimator is contained at least one of the following:\n\t- Turnover and/or TurnoverEstimator,\n\t- WeightsTracking,\n\t- TurnoverRiskMeasure,\n\t- custom constraints which use asset weights,\n\t- custom objective penalties which use asset weights,\n\t- a time-dependent constraint whose entries need previous weights (e.g. a PreviousWeightsFunction).\nTo enable parallel processing please either mark the weights as fixed or remove the offending component(s). Time-dependent constraints alone do not force sequential processing."
end
"""
    parallel_folds(fit_fold, n::Integer, ex::FLoops.Transducers.Executor,
                   ::Type{ElT} = PredictionResult)

Run `n` cross-validation folds in parallel over the executor `ex`. `ElT` is the element type of
the result of one fold: a single [`PredictionResult`](@ref) for a time-ordered scheme, and a
`Vector{PredictionResult}` for the multi-path combinatorial scheme.

This is the sibling of [`run_folds`](@ref). A fold here takes no previous fold, so `fit_fold`
takes the fold index alone. [`fold_loop`](@ref) decides which of the two runs.

`ElT` is a *positional* `::Type{ElT}` argument, not a keyword, so a method always
specialises on it and `Vector{ElT}(undef, n)` is built at compile time. The value of a
keyword survives only by constant propagation, which one forwarding call can lose.

# Algorithm

 1. Allocate `predictions = Vector{ElT}(undef, n)`.
 2. For each `i in 1:n`, in parallel over `ex`, set `predictions[i] = fit_fold(i)`.

# Related

  - [`run_folds`](@ref)
  - [`fold_loop`](@ref)
  - [`fit_and_predict`](@ref)
"""
function parallel_folds(fit_fold, n::Integer, ex::FLoops.Transducers.Executor,
                        ::Type{ElT} = PredictionResult) where {ElT}
    predictions = Vector{ElT}(undef, n)
    FLoops.@floop ex for i in 1:n
        predictions[i] = fit_fold(i)
    end
    return predictions
end
"""
    run_folds(fit_fold, n::Integer, ::Type{ElT} = PredictionResult; pws = nothing)

Run `n` cross-validation folds in order, and emit [`cv_sequential_info`](@ref).

`prev` is the last fold whose weights the next fold can be handed. Fold 1 takes `nothing`,
because it has no fold behind it. A failed fold is skipped, and the fold before it stays
`prev`. The caller uses `prev` to thread its weights into fold `i`. `pws` is the
Previous-Weights Source of the scheme, and it decides what [`threads_weights`](@ref) tests.

[`fold_loop`](@ref) is the only caller, and it calls this loop only when the folds are a
timeline *and* the estimator needs the weights of the previous fold. The loop therefore takes
no executor, and its sibling [`parallel_folds`](@ref) runs every other case. `ElT` is a
*positional* `::Type{ElT}` argument for the reason that [`parallel_folds`](@ref) gives.

# Algorithm

 1. Emit [`cv_sequential_info`](@ref) as an `@info` message.
 2. Allocate `predictions = Vector{ElT}(undef, n)`, and set `prev = nothing`.
 3. For each `i in 1:n` in order, set `predictions[i] = fit_fold(i, prev)`.
 4. After fold `i`, set `prev` with [`advance_previous_fold`](@ref): `predictions[i]` when [`threads_weights`](@ref) holds of it, and `prev` unchanged otherwise.

# Related

  - [`parallel_folds`](@ref)
  - [`fold_loop`](@ref)
  - [`threads_weights`](@ref)
  - [`cv_sequential_info`](@ref)
  - [`folds_are_time_ordered`](@ref)
  - [`fit_and_predict`](@ref)
"""
function run_folds(fit_fold, n::Integer, ::Type{ElT} = PredictionResult;
                   pws = nothing) where {ElT}
    @info(cv_sequential_info())
    predictions = Vector{ElT}(undef, n)
    prev = nothing
    for i in 1:n
        predictions[i] = fit_fold(i, prev)
        prev = advance_previous_fold(pws, prev, predictions[i])
    end
    return predictions
end
"""
    advance_previous_fold(pws, prev, pred::PredictionResult)

Give the fold the next fold is handed: `pred` when [`threads_weights`](@ref) holds of it, `prev` otherwise.

[`run_folds`](@ref) and [`online_folds`](@ref) both call it, so the two sequential loops advance by the same rule.

# Related

  - [`threads_weights`](@ref)
  - [`run_folds`](@ref)
  - [`online_folds`](@ref)
"""
function advance_previous_fold(pws, prev, pred::PredictionResult)
    return threads_weights(pws, pred) ? pred : prev
end
"""
    assert_unshuffled_folds(cv, train_idx)

Assert that the cross-validation scheme `cv` enumerates unshuffled folds.

A shuffled fold breaks the time order in which the fold loop, the [`TimeDependentContext`](@ref)
schedules and the rolling transforms read the rows of the fold. A scheme may leave gaps, as
purging, embargoing and the combinatorial splits do, but it must never reorder rows.

# Arguments

  - `cv`: The cross-validation scheme, or any value.
  - `train_idx`: The training indices of every fold.

# Validation

  - `cv` does not have a `shuffle` field set to `true`. The check reads the field by `hasfield`, so it holds for a scheme a user defines. No scheme of the library has such a field.
  - The training indices of every fold increase strictly.
  - Either failure raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`fold_loop`](@ref)
  - [`fit_and_predict`](@ref)
  - [`cross_val_predict`](@ref)
"""
function assert_unshuffled_folds(cv, train_idx)
    @argcheck(!(hasfield(typeof(cv), :shuffle) && cv.shuffle),
              "Cross validation estimator must not be shuffled.")
    @argcheck(all(x -> all(>(zero(eltype(x))), diff(x)), train_idx),
              "Cross validation estimator must not be shuffled.")
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

One fold of a cross-validation scheme, as [`fold_loop`](@ref) hands it to its callback.

The record holds everything the fold loop gives its callback. `est` and `rd` are already
resolved: the loop took the asset view, swapped every [`TimeDependent`](@ref) schedule for its
value at fold `i`, and threaded in the weights of the previous fold. `train` and `test` are the
windows of this fold, so a callback never indexes `train_idx` or `test_idx` itself. `w_prev`
holds the weights that were threaded, a second time, so that a failed fold can hold them.
[`held_start_weights`](@ref) reads it inside `predict`.

`train === nothing` says *the estimator holds its window*. The online arm of the loop,
[`online_folds`](@ref), hands its callback a `Fold` of that shape. `est` has already folded
every row of the training window through [`partial_fit!`](@ref), so a callback reads it out
with `optimise(est)` and no returns, in place of a fit over rows the record does not carry.
[`fit_and_predict`](@ref) takes `train_idx = nothing` for that case, so the callback of every
entry point is the same under both arms.

The type is immutable and every field is concretely typed at the construction site, so the
record costs nothing at run time. [`fold_loop`](@ref) is the only site that builds one,
which is why there is no keyword constructor.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`fold_loop`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`fit_and_predict`](@ref)
"""
@concrete struct Fold
    """
    Index of the fold within the scheme's `split` enumeration (1-based).
    """
    i
    """
    Number of folds in the enumeration.
    """
    n
    """
    The fold's resolved estimator: asset-viewed, schedule-swapped, weights-threaded.
    """
    est
    """
    The fold's (possibly asset-viewed) input data.
    """
    rd
    """
    The fold's training indices, or `nothing` when the estimator holds its window.
    """
    train
    """
    The fold's test indices.
    """
    test
    """
    The previous weights threaded into `est`, or `nothing` when there are none.
    """
    w_prev
end
"""
    folds_are_time_ordered(cv)

Return `true` if the fold enumeration of a cross-validation scheme is a timeline.

This is one half of the conjunction [`fold_loop`](@ref) computes, so a scheme states for
itself whether its folds carry history. A walk-forward, a multiple-randomised path, and any
scheme with no more specific method enumerate their folds in time order. Fold `i` has fold
`i - 1` behind it, so the loop may run the folds in order and thread the previous fold's
weights. [`needs_previous_weights`](@ref) is the other half, and it decides whether the loop
does so.

A [`NonSeqCVER`](@ref) scheme answers `false`. A k-fold and a combinatorial enumeration are
not timelines: each fold is independent of the others, so no fold has a previous fold, and
the loop runs them in parallel. A `KFold` training window holds rows that follow its test
window, so a quantity measured against another fold's weights is not a backtest reading.

The method is per type and takes the scheme itself, so inference reads the answer from the
type of `cv` and never needs the value. The fallback over `::Any` also answers `true` for
`nothing`, which is what [`fold_loop`](@ref) receives from a call site that holds no scheme.

# Related

  - [`fold_loop`](@ref)
  - [`NonSeqCVER`](@ref)
  - [`needs_previous_weights`](@ref)
  - [`cross_val_predict`](@ref)
"""
folds_are_time_ordered(::Any)::Bool = true
folds_are_time_ordered(::NonSeqCVER)::Bool = false

"""
    fold_evaluation(cv)

Read the evaluation switches of a cross-validation scheme, in one named tuple.

Every scheme entry point reads the same settings before it runs its folds, and each scheme states them through a method of its own. A scheme that carries none of them, and a call site that holds a split result in place of the scheme that made it, reach the fallback. The fallback sets no drift, no drifted previous weights, no override of the fee clock, no stored weight path, and a Held Gap that warns.

The method is per type and takes the scheme itself, so inference reads the answer from the type of `cv`, exactly as [`folds_are_time_ordered`](@ref) does.

# Returns

  - `(; wd, pws, fa, store_weight_path, strict)`: The Weight Drift, the Previous-Weights Source, the Fee Clock of the fold's realised series, the flag that stores a fold's weight path, and the flag that makes a Held Gap raise rather than warn.

# Related

  - [`folds_are_time_ordered`](@ref)
  - [`held_weights_drift`](@ref)
  - [`override_fee_amortisation`](@ref)
  - [`AbstractWeightDrift`](@ref)
  - [`AbstractPreviousWeightsSource`](@ref)
  - [`AbstractFeeAmortisation`](@ref)
  - [`fold_loop`](@ref)
"""
function fold_evaluation(::Any)
    return (; wd = nothing, pws = nothing, fa = nothing, store_weight_path = false,
            strict = false)
end
"""
    folds_are_stepped(cv)

Whether [`fold_loop`](@ref) fits each fold of `cv` by the online step, or by a refit from the fold's training window.

`false` is a refit every fold, which is what every scheme answers unless it is an Online Scheme: an [`Online`](@ref) around a walk-forward, built by [`OnlineIndexWalkForward`](@ref), [`OnlineDateWalkForward`](@ref) or [`OnlineHindsightSplit`](@ref), answers `true` through a method of its own, and a [`MultipleRandomised`](@ref) answers for the walk-forward it wraps. A plain scheme, a split result, and a call site that holds no scheme reach this fallback.

The method is per type and takes the scheme itself, so inference reads the answer from the type of `cv`, exactly as [`fold_evaluation`](@ref) and [`folds_are_time_ordered`](@ref) do, and the arm that cannot run is eliminated.

# Related

  - [`Online`](@ref)
  - [`fold_evaluation`](@ref)
  - [`folds_are_time_ordered`](@ref)
  - [`fold_loop`](@ref)
"""
folds_are_stepped(::Any)::Bool = false
"""
    fold_loop(fit_fold, est, n::Integer, ex::FLoops.Transducers.Executor,
              ::Type{ElT} = PredictionResult; rd, train_idx, test_idx,
              path_id = nothing, cv = nothing, fold_view = nothing, pws = nothing)

Run the `n` folds of a cross-validation scheme over `est`, and resolve the estimator of each
fold before the callback sees it.

This is the fold loop of the package. Every cross-validation entry point goes through it, the
optimiser-level schemes and the [`Pipeline`](@ref) ones alike. The callback takes the one
[`Fold`](@ref) record, so a call site names what it reads (`fold.est`, `fold.train`) and does
not rely on the position of an argument.

The loop is also the one site that decides how the folds run, and it has three arms. The online
arm, [`online_folds`](@ref), runs when the scheme is an Online Scheme. It warms one estimator up
on the first training window, folds the new rows of each fold into it, and hands the callback a
[`Fold`](@ref) whose `train` is `nothing`. Steps 5 and 6 below run on a per-fold copy of the
threaded estimator, and a schedule that the *step* reads is resolved earlier in the same fold
through [`online_step_fold`](@ref). Otherwise the folds run in order only when the folds of `cv`
are a timeline *and* `est` needs the weights of the previous fold. Every other case runs in
parallel, because such a fold does not depend on the other folds.

The loop reads the three predicates of `cv` by type, so inference folds the choice and removes
the arms that cannot run. A `Bool` keyword cannot do this: its value survives only by constant
propagation, which one call can lose, and the sequential arm is then inferred where it can never
run. The two path-level sites, of the optimiser and of the Pipeline, pass the
[`MultipleRandomised`](@ref) that they run, which answers for the walk-forward that it wraps.

`ElT` is the element type of the result of one fold: a single [`PredictionResult`](@ref) for a
time-ordered scheme, and a `Vector{PredictionResult}` for the multi-path combinatorial scheme.
It is positional for the reason that [`parallel_folds`](@ref) gives.

# Algorithm

 1. When `est` [`is_time_dependent`](@ref), check the number of folds against its schedules with [`assert_time_dependent_fold_count`](@ref).
 2. When `cv` is not an Online Scheme ([`folds_are_stepped`](@ref)), refuse an [`Online`](@ref) anywhere in `est` with [`assert_batch_entry`](@ref). The batch arms refit every fold from its training window and run no warm-up, so the loop refuses it once here and not once per fold on the workers of `ex`.
 3. Choose the arm: [`online_folds`](@ref) for an Online Scheme; [`run_folds`](@ref) when [`folds_are_time_ordered`](@ref) holds of `cv` and `est` [`needs_previous_weights`](@ref); and [`parallel_folds`](@ref) otherwise.
 4. For fold `i`, take the view of `(est, rd)` that `fold_view(i)` gives, or `(est, rd)` when `fold_view` is `nothing`. An asset-resampling scheme gives a view.
 5. Read the previous weights with [`previous_weights`](@ref), giving `w_prev`, and swap every [`TimeDependent`](@ref) schedule for its value at fold `i` against a [`TimeDependentContext`](@ref). The swap runs first, so a per-fold entry that it swaps in also receives the weights of step 6.
 6. When `w_prev` is not `nothing` and `est` needs previous weights, thread them in through [`factory`](@ref). [`one_previous_portfolio`](@ref) refuses the population of a frontier, because no one portfolio of it is the previous one.
 7. Call `fit_fold` with the [`Fold`](@ref) of fold `i`.

# Returns

  - `predictions::Vector{ElT}`: One result per fold, in split order. Under a [`Resume`](@ref), the new folds only.
  - `opt`: The estimator that the online arm threaded, folded through the last training end, or
    `nothing` from the batch arms. The Result of an online walk-forward carries it.

# Related

  - [`Fold`](@ref)
  - [`online_folds`](@ref)
  - [`Resume`](@ref)
  - [`run_folds`](@ref)
  - [`parallel_folds`](@ref)
  - [`assert_unshuffled_folds`](@ref)
  - [`folds_are_stepped`](@ref)
  - [`folds_are_time_ordered`](@ref)
  - [`fit_and_predict`](@ref)
  - [`cross_val_predict`](@ref)
"""
function fold_loop(fit_fold, est, n::Integer, ex::FLoops.Transducers.Executor,
                   ::Type{ElT} = PredictionResult; rd, train_idx, test_idx,
                   path_id = nothing, cv = nothing, fold_view = nothing,
                   pws = nothing) where {ElT}
    td_flag = is_time_dependent(est)
    if td_flag
        assert_time_dependent_fold_count(est, n)
    end
    if !folds_are_stepped(cv)
        assert_batch_entry(est, "the fold loop under a scheme that is not an Online Scheme")
    end
    prev_w_flag = needs_previous_weights(est)
    # The per-fold copy. `esti` is the fold's estimator before resolution: the configuration
    # in the batch arms, the threaded estimator in the online arm.
    function resolve(i, prev, esti, rdi, train)
        w_prev = previous_weights(pws, prev)
        # Resolve time-dependent entries first, so a freshly swapped-in per-fold entry also
        # receives the previous weights from the factory pass below. The online arm builds
        # the same record earlier in the fold, for the schedules its step reads.
        if td_flag
            esti = update_time_dependent_estimator(esti,
                                                   fold_context(i, n, rdi, train_idx,
                                                                test_idx, w_prev, path_id))
        end
        if !isnothing(w_prev) && prev_w_flag
            esti = factory(esti, one_previous_portfolio(esti, w_prev))
        end
        return fit_fold(Fold(i, n, esti, rdi, train, test_idx[i], w_prev))
    end
    function fold(i, prev)
        (esti, rdi) = isnothing(fold_view) ? (est, rd) : fold_view(i)
        return resolve(i, prev, esti, rdi, train_idx[i])
    end
    # All three predicates are per-type methods over the concretely-typed `cv` and `est`,
    # so inference decides the route from types alone and eliminates the arms that cannot
    # run. A `Bool` keyword would leave the `run_folds` arm inferred, and its
    # abstractly-typed `predictions[i - 1]` is a runtime dispatch. See the ADR 0067
    # amendments.
    return if folds_are_stepped(cv)
        online_folds(resolve, est, n, ElT, path_id; rd = rd, train_idx = train_idx,
                     test_idx = test_idx, fold_view = fold_view, pws = pws)
    elseif folds_are_time_ordered(cv) && prev_w_flag
        (run_folds(fold, n, ElT; pws = pws), nothing)
    else
        (parallel_folds(i -> fold(i, nothing), n, ex, ElT), nothing)
    end
end
function fit_and_predict(opt::OptE_Opt_TD, rd::ReturnsResult, cv::NonSeqCVER; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         id = nothing)
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    (; wd, pws, fa, store_weight_path, strict) = fold_evaluation(cv)
    hwd = held_weights_drift(wd, pws)
    predictions, _ = fold_loop(opt, length(train_idx), ex; rd = rd, train_idx = train_idx,
                               test_idx = test_idx, cv = cv, pws = pws) do fold
        return fit_and_predict(fold.est, fold.rd; train_idx = fold.train,
                               test_idx = fold.test, cols = cols, wd = wd, hwd = hwd,
                               fa = fa, store_weight_path = store_weight_path,
                               strict = strict, w_prev = fold.w_prev)
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id)
end

export fit, fit_predict
