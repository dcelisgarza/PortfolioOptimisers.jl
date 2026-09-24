"""
    fit_predict(opt::OptE_Opt, rd::ReturnsResult)

Fit optimisation estimator `opt` on returns data `rd` and immediately produce a
[`PredictionResult`](@ref) for the same data.

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

The three-argument method (`opt`, `rd`, `cv`) performs full cross-validated prediction over all folds of `cv`.
The two-argument methods operate on a single pre-defined train/test split or on a pre-existing result.

The estimator form takes `train_idx = nothing` to mean *the estimator holds its window*: it reads the estimator out through `optimise(opt)` instead of fitting it over `port_opt_view(rd, train_idx, cols)`, and predicts over `test_idx` as before. That is the read-out of the online arm of [`fold_loop`](@ref), and it is also a public entry for a hand-stepped estimator — one warmed up with [`update_online_estimator`](@ref) and folded with [`partial_fit!`](@ref) — so `fit_and_predict(opt, rd; test_idx)` on a stepped estimator equals `fit_and_predict(opt, rd; train_idx, test_idx)` on the cold one over the same rows. The two arms are [`fit_fold_result`](@ref)'s, and the method lives beside them.

# Arguments

  - `opt`: Optimisation estimator or an existing optimisation result.
  - `rd::ReturnsResult`: Returns data.
  - `cv::NonSeqCVER`: Non-sequential cross-validation estimator (e.g. [`KFold`](@ref) or [`CombinatorialCrossValidation`](@ref)).
  - `cv::CombCVER`: Combinatorial cross-validation estimator or result ([`CombinatorialCrossValidation`](@ref)).
  - `train_idx::Option{<:VecInt}`: Training indices, or `nothing` to read a stepped estimator out.
  - `test_idx`: Test indices (vector or vector of vectors).
  - `cols`: Column selector (default `:` for all assets).

# Returns

  - [`MultiPeriodPredictionResult`](@ref), [`PopulationPredictionResult`](@ref), or [`PredictionResult`](@ref).

# Details

  - A combinatorial `cv` takes its own method, because its folds recombine into several paths rather than one. That method regroups the fold predictions by path and returns one [`MultiPeriodPredictionResult`](@ref) per path, wrapped in a [`PopulationPredictionResult`](@ref).

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
    sort_predictions!(res::VecVecInt, predictions::VecPredRes) -> VecPredRes
    sort_predictions!(res::CrossValidationResult, predictions::VecPredRes) -> VecPredRes

Sort prediction results to match the order of test indices.

Reorders `predictions` so that they align with the original time ordering of `test_idx`. The key is the first observation of each fold, so the folds come back in the order the timeline visits them.

# Arguments

  - `res`:

      + `::VecVecInt`: Vector of test index vectors.
      + `::CrossValidationResult`: Cross validation result object, uses the test indices stored in `res.test_idx`.

  - `predictions`: Vector of prediction results, one per fold, in split order.

# Validation

  - Every element of `test_idx` holds unique indices.

# Returns

  - Sorted predictions vector.

# Details

  - [`CombinatorialCrossValidationResult`](@ref) has its own method with a different shape and a different job. Its folds do not form one timeline, so that method takes a vector of per-split vectors and regroups them by path into [`MultiPeriodPredictionResult`](@ref)s rather than sorting one timeline.

# Related

  - [`fit_and_predict`](@ref)
  - [`path_fit_and_predict`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
"""
function sort_predictions!(test_idx::VecVecInt, predictions::VecPredRes)
    @argcheck(all(x -> allunique(x), test_idx), "Test indices must be unique.")
    idx = sortperm(test_idx; by = x -> x[1])
    return predictions[idx]
end
function sort_predictions!(res::CrossValidationResult, predictions::VecPredRes)
    return sort_predictions!(res.test_idx, predictions)
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

Run `n` cross-validation folds in parallel, filling `predictions[i] = fit_fold(i)` for `i in 1:n`
over executor `ex`. `ElT` is the per-fold result element type (a single [`PredictionResult`](@ref)
for time-ordered schemes, a `Vector{PredictionResult}` for the multi-path combinatorial scheme).

This is the sibling of [`run_folds`](@ref), and the two divide the work by name. A fold here
takes no previous fold, so `fit_fold` takes the fold index alone. [`fold_loop`](@ref)
decides which of the two runs, and neither one re-decides.

`ElT` is a *positional* `::Type{ElT}` argument, not a keyword, so a method always
specialises on it and `Vector{ElT}(undef, n)` stays a compile-time construction. As a
keyword its value only survives constant propagation, which one forwarding hop is enough
to lose.

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

Run `n` cross-validation folds in order, filling `predictions[i] = fit_fold(i, prev)` for
`i in 1:n`, and emit [`cv_sequential_info`](@ref). `prev` is the last fold whose weights the
next fold can be handed: fold 1 takes `nothing`, because it has no fold behind it, and after
fold `i` the loop advances `prev` to `predictions[i]` only when [`threads_weights`](@ref)
holds of it, so a fold whose solve failed is skipped over and the fold before it is read
instead. The caller uses `prev` to thread its weights into fold `i`. `ElT` is the per-fold
result element type, and `pws` is the scheme's Previous-Weights Source, which decides what
[`threads_weights`](@ref) tests.

This is the sequential loop, and it does that one job. [`fold_loop`](@ref) is the only site
that calls it, and it calls it only when the folds are a timeline *and* the estimator needs
the previous fold's weights. The loop therefore neither re-decides nor takes an executor:
its sibling [`parallel_folds`](@ref) owns the other case.

`ElT` is a *positional* `::Type{ElT}` argument for the reason given in
[`parallel_folds`](@ref).

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

One line shared by [`run_folds`](@ref) and [`online_folds`](@ref), so the two sequential loops advance by the same rule.

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

Two checks, applied to every scheme:

 1. `cv` must not declare a set `shuffle` field. The check is by `hasfield`, so it holds
    for a user-defined scheme too — no scheme in this package has such a field.
 2. Every fold's training indices must increase strictly. A scheme may leave gaps (purging,
    embargoing and the combinatorial splits all do), but it must never reorder rows.

A shuffled fold breaks the timeline that the fold loop, the [`TimeDependentContext`](@ref)
schedules and the rolling transforms all read the fold's rows in.

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

The record is the fold loop's whole hand-off. `est` and `rd` are already resolved: the
asset view is taken, every [`TimeDependent`](@ref) schedule is swapped for its fold-`i`
value, and the previous fold's weights are threaded in. `train` and `test` are this fold's
own windows, so a callback never indexes `train_idx`/`test_idx` itself. `w_prev` is the
weights that were threaded, handed over a second time so a fold whose solve fails can hold
them: [`held_start_weights`](@ref) reads it inside `predict`.

`train === nothing` says *the estimator holds its window*. The online arm of the loop,
[`online_folds`](@ref), hands its callback a `Fold` of that shape: `est` has already folded
every row of the fold's training window through [`partial_fit!`](@ref), so a callback reads
it out — `optimise(est)` with no returns — rather than fitting it over rows the record does
not carry. [`fit_and_predict`](@ref) takes `train_idx = nothing` for exactly that, so every
entry point's callback is unchanged across the two arms.

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
struct Fold{T1, T2, T3, T4, T5, T6, T7}
    """
    Index of the fold within the scheme's `split` enumeration (1-based).
    """
    i::T1
    """
    Number of folds in the enumeration.
    """
    n::T2
    """
    The fold's resolved estimator: asset-viewed, schedule-swapped, weights-threaded.
    """
    est::T3
    """
    The fold's (possibly asset-viewed) input data.
    """
    rd::T4
    """
    The fold's training indices, or `nothing` when the estimator holds its window.
    """
    train::T5
    """
    The fold's test indices.
    """
    test::T6
    """
    The previous weights threaded into `est`, or `nothing` when there are none.
    """
    w_prev::T7
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
type of `cv` and never needs the value. `folds_are_time_ordered(::Any)` answers `nothing`
too, which is what [`fold_loop`](@ref) receives from a call site that holds no scheme.

# Related

  - [`fold_loop`](@ref)
  - [`NonSeqCVER`](@ref)
  - [`needs_previous_weights`](@ref)
  - [`cross_val_predict`](@ref)
"""
folds_are_time_ordered(::Any) = true
folds_are_time_ordered(::NonSeqCVER) = false

"""
    fold_evaluation(cv)

Read the evaluation switches of a cross-validation scheme, in one named tuple.

Every scheme entry point reads the same settings before it runs its folds, and each scheme states them for itself through a method of its own. A scheme that carries none of them, and a call site that holds a split result rather than the scheme that made it, reach the fallback and get today's behaviour: no drift, no drifted previous weights, no fee-clock override, and no stored weight path.

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
folds_are_stepped(::Any) = false
"""
    fold_loop(fit_fold, est, n::Integer, ex::FLoops.Transducers.Executor,
              ::Type{ElT} = PredictionResult; rd, train_idx, test_idx,
              path_id = nothing, cv = nothing, fold_view = nothing)

Run the `n` folds of a cross-validation scheme over `est`, and resolve each fold's estimator
before the callback sees it.

This is the fold loop of the package. Every cross-validation entry point goes through
it: the optimiser-level schemes and the [`Pipeline`](@ref) ones alike. For fold `i` it
does four steps.

 1. It takes the fold's view of `(est, rd)` through `fold_view`. An asset-resampling scheme
    gives one; the other schemes do not.
 2. It swaps every [`TimeDependent`](@ref) schedule for its fold-`i` value against a
    [`TimeDependentContext`](@ref), if `est` [`is_time_dependent`](@ref). The swap runs
    first, so a freshly swapped-in per-fold entry also gets the weights of step 3.
 3. It threads the previous fold's weights in through [`factory`](@ref), if `est`
    [`needs_previous_weights`](@ref). [`one_previous_portfolio`](@ref) refuses the
    population a frontier sweep gives, because no one portfolio of it is the previous one.
 4. It calls `fit_fold(fold)`, with `fold` a [`Fold`](@ref).

The callback takes the one [`Fold`](@ref) record, so a call site names what it reads
(`fold.est`, `fold.train`) instead of relying on the position of an argument.

[`assert_time_dependent_fold_count`](@ref) runs once, before the loop, and so does
[`assert_batch_entry`](@ref) when the scheme is not an Online Scheme: the batch arms refit
every fold from its training window and run no warm-up, so an [`Online`](@ref) anywhere
in `est` would reach `prior(pe, X)` unresolved, and it is refused by name instead — once,
here, rather than up to once per fold on the workers of `ex`.

This is also the one site that decides how the folds run, and it has three arms. The
online arm, [`online_folds`](@ref), is taken first, when the scheme is an Online Scheme
([`folds_are_stepped`](@ref)): the loop then warms one estimator up on the first training window,
folds each fold's new rows into it, and hands the callback a [`Fold`](@ref) whose `train` is
`nothing` — steps 2 and 3 run on a per-fold copy of the threaded estimator, so a schedule and
the previous weights still reach the fold, and a schedule the *step* reads is resolved one
fold earlier through [`online_step_fold`](@ref). Otherwise a run is sequential only when two
facts hold at once: the fold enumeration of `cv` is a timeline
([`folds_are_time_ordered`](@ref)), *and* `est` needs the previous fold's weights
([`needs_previous_weights`](@ref)). The conjunction routes through [`run_folds`](@ref).
Every other case routes through [`parallel_folds`](@ref), because a fold with no fold behind
it, or a fold whose estimator reads no previous weights, is independent of the other folds.
No loop re-decides.

`cv` is the scheme, and the loop reads its three per-type predicates rather than a
value a call site computes. All are decided by the *types* of `cv` and `est`, so inference
folds the conjunction and eliminates the arm that cannot run. A `Bool` keyword cannot do
this: its value survives only by constant propagation, which one call hop loses, and the
sequential arm is then inferred even where it can never run. The two path-level sites enumerate an inner walk-forward; the optimiser's passes the
[`MultipleRandomised`](@ref) it runs, which answers for the walk-forward it wraps, and the Pipeline's holds no
scheme and omits `cv`; `folds_are_time_ordered(nothing)` answers `true`.

`ElT` is the per-fold result element type: a single
[`PredictionResult`](@ref) for a time-ordered scheme, a `Vector{PredictionResult}` for the
multi-path combinatorial scheme. It is positional for the reason given in
[`parallel_folds`](@ref).

# Returns

  - `predictions::Vector{ElT}`: One result per fold, in split order — the new folds only under
    a [`Resume`](@ref).
  - `opt`: The estimator the online arm threaded, folded through the last training end, or
    `nothing` from the batch arms. An online walk-forward's Result carries it.

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
        # the same record one fold earlier, for the schedules its step reads.
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
