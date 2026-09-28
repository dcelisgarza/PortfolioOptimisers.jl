"""
    const Pipeline_OnlPipe = Union{<:Pipeline, <:Online{<:Pipeline}, <:Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any, <:Pipeline}}}

Alias for the three estimators that the `cross_val_predict` methods of a Pipeline give to the fold loop. The group exists because these methods share one body, [`pipeline_cross_val_predict`](@ref), and this alias is the type that the body dispatches on.

A plain pipeline takes the fold route. `Online(pipe)` declares a refit from a buffer of the input data. `Resume(res)` continues an online run whose Result holds a pipeline.

# Related

  - [`Pipeline`](@ref)
  - [`Online`](@ref)
  - [`Resume`](@ref)
  - [`PipelineBufferState`](@ref)
"""
const Pipeline_OnlPipe = Union{<:Pipeline, <:Online{<:Pipeline},
                               <:Resume{<:MultiPeriodPredictionResult{<:Any, <:Any, <:Any,
                                                                      <:Pipeline}}}
#! Begin: TimeDependent schedules as pipeline optimisation steps.
"""
    pipeline_step_is_time_dependent(x)

Return `true` when a [`Pipeline`](@ref) step holds a [`TimeDependent`](@ref) schedule.

An optimisation estimator or result, a schedule, a nested pipeline and a [`PipelineStep`](@ref) answer through [`is_time_dependent`](@ref). Every other step answers `false`. A preprocessing step, a prior, a phylogeny, an uncertainty set, a constraint estimator and a callable hold no schedule. A schedule for one of these families is a field of the optimisation step.

# Related

  - [`is_time_dependent`](@ref)
  - [`Pipeline`](@ref)
"""
pipeline_step_is_time_dependent(::Any)::Bool = false
function pipeline_step_is_time_dependent(x::Union{<:OptE_Opt, <:TimeDependent, <:Pipeline,
                                                  <:PipelineStep})
    return is_time_dependent(x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when a step of the [`Pipeline`](@ref) holds a schedule.

Three steps can hold one: a [`TimeDependent`](@ref) step in the place of the optimiser, an optimisation step with a scheduled field, and a nested pipeline that holds one of the two. A [`PipelineStep`](@ref) answers for the estimator that it wraps.

# Related

  - [`pipeline_step_is_time_dependent`](@ref)
  - [`is_time_dependent`](@ref)
"""
function is_time_dependent(p::Pipeline)
    return any(pipeline_step_is_time_dependent, p.steps)
end
function is_time_dependent(ps::PipelineStep)
    return pipeline_step_is_time_dependent(ps.est)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when a step of the [`Pipeline`](@ref) reads the weights of the previous fold.

When the folds of the scheme are a timeline ([`folds_are_time_ordered`](@ref)), a `true` answer makes [`fold_loop`](@ref) run the folds in sequence. The loop then fills `w_prev` of the [`TimeDependentContext`](@ref) with the weights of the previous fold. The folds of any other scheme run in parallel with no previous weights.

A [`PipelineStep`](@ref) answers for the estimator that it wraps, and a wrapped callable answers `false`. A nested pipeline answers for its own steps. A schedule answers by the conventions of [`needs_previous_weights`](@ref). Its entries count, and its `default` does not.

# Related

  - [`needs_previous_weights`](@ref)
  - [`fold_loop`](@ref)
  - [`run_folds`](@ref)
"""
function needs_previous_weights(p::Pipeline)
    return any(needs_previous_weights, p.steps)
end
function needs_previous_weights(ps::PipelineStep)
    return isa(ps.est, Function) ? false : needs_previous_weights(ps.est)
end
"""
    assert_pipeline_step_fold_count(x, n::Integer, all_binds::Bool)

Check the number of folds `n` against the schedules of one [`Pipeline`](@ref) step.

An optimisation step, a [`TimeDependent`](@ref) step and a nested pipeline answer through [`assert_time_dependent_fold_count`](@ref). That check sizes the entries of a schedule to the loop, and it does not read the `default`. A [`PipelineStep`](@ref) answers for the estimator that it wraps, and a wrapped callable passes. Every other step passes.

# Validation

  - Everything [`assert_time_dependent_fold_count`](@ref) refuses.

# Related

  - [`assert_time_dependent_fold_count`](@ref)
"""
assert_pipeline_step_fold_count(::Any, ::Integer, ::Bool)::Nothing = nothing
function assert_pipeline_step_fold_count(x::Union{<:OptE_Opt, <:TD_OptE_Opt, <:Pipeline},
                                         n::Integer, all_binds::Bool)::Nothing
    assert_time_dependent_fold_count(x, n, all_binds)
    return nothing
end
function assert_pipeline_step_fold_count(ps::PipelineStep, n::Integer,
                                         all_binds::Bool)::Nothing
    if !isa(ps.est, Function)
        assert_pipeline_step_fold_count(ps.est, n, all_binds)
    end
    return nothing
end
function assert_time_dependent_fold_count(p::Pipeline, n::Integer,
                                          all_binds::Bool = true)::Nothing
    for est in p.steps
        assert_pipeline_step_fold_count(est, n, all_binds)
    end
    return nothing
end
"""
    update_time_dependent_step(est, ctx::TimeDependentContext, all_binds::Bool)

Resolve one [`Pipeline`](@ref) step for the fold that `ctx` describes. This is the part of [`update_time_dependent_estimator`](@ref) that applies to one step.

# Algorithm

 1. Resolve an optimisation step, a [`TimeDependent`](@ref) step and a nested pipeline through [`update_time_dependent_estimator`](@ref). A schedule step gives its entry for the fold, an estimator or a precomputed result. An optimisation step resolves its scheduled fields, and a nested pipeline resolves its own steps.
 2. Resolve a [`PipelineStep`](@ref) that wraps a schedule to the entry of step 1, with no wrapper. The wrapper only declares the slot that dispatch cannot infer, and [`run_step`](@ref) dispatches on the entry. An entry can be a result, and a wrapper cannot hold one.
 3. Return a [`PipelineStep`](@ref) that wraps a callable unchanged. Resolve the estimator of any other [`PipelineStep`](@ref) by this algorithm, giving `newest`. Rebuild the wrapper around `newest` when `newest` is a new value.
 4. Return every other step unchanged.

# Validation

  - Everything [`update_time_dependent_estimator`](@ref) refuses. This includes an entry of a callable schedule that is not an optimiser.

# Related

  - [`update_time_dependent_estimator`](@ref)
  - [`PipelineStep`](@ref)
"""
update_time_dependent_step(est, ::TimeDependentContext, ::Bool) = est
function update_time_dependent_step(x::Union{<:OptE_Opt, <:TD_OptE_Opt, <:Pipeline},
                                    ctx::TimeDependentContext, all_binds::Bool)
    return update_time_dependent_estimator(x, ctx, all_binds)
end
function update_time_dependent_step(ps::PipelineStep, ctx::TimeDependentContext,
                                    all_binds::Bool)
    est = ps.est
    if isa(est, TimeDependent)
        return update_time_dependent_estimator(est, ctx, all_binds)
    end
    if isa(est, Function)
        return ps
    end
    newest = update_time_dependent_step(est, ctx, all_binds)
    return newest === est ? ps : PipelineStep(newest, ps.reads, ps.writes, ps.target)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the schedules of a [`Pipeline`](@ref) for the fold that `ctx` describes.

The fold loop does this swap before [`fit`](@ref) runs, so `fit` and [`run_step`](@ref) never see a fold. When `fit` runs on the training window of the fold, each schedule step is an optimiser or a precomputed result. So [`inject_context`](@ref) and [`maybe_inject_step`](@ref) never receive a schedule.

# Algorithm

 1. Resolve each step with [`update_time_dependent_step`](@ref), giving the new steps.
 2. Return a pipeline with the same names, the new steps and the same `cache`.

# Validation

  - Everything [`update_time_dependent_step`](@ref) refuses.

# Related

  - [`update_time_dependent_step`](@ref)
  - [`cross_val_predict`](@ref)
"""
function update_time_dependent_estimator(p::Pipeline, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    return Pipeline(p.names,
                    map(est -> update_time_dependent_step(est, ctx, all_binds), p.steps),
                    p.cache)
end
"""
    reset_time_dependent_step(est)

Replace one [`Pipeline`](@ref) step with its value outside a fold.

# Algorithm

 1. Replace a [`TimeDependent`](@ref) step with its `default`, reset in turn through [`reset_time_dependent_estimator`](@ref). An optimisation step has a static value to fall back to, and a schedule step has none, so the `default` is the only value outside a fold.
 2. Reset an optimisation step and a nested pipeline through [`reset_time_dependent_estimator`](@ref).
 3. Replace a [`PipelineStep`](@ref) that wraps a schedule with the value of step 1, with no wrapper. Return a [`PipelineStep`](@ref) that wraps a callable unchanged. Reset the estimator of any other [`PipelineStep`](@ref) by this algorithm, giving `newest`. Rebuild the wrapper around `newest` when `newest` is a new value.
 4. Return every other step unchanged.

# Validation

  - A [`TimeDependent`](@ref) step has a `default`. A [`TimeDependentDefaultError`](@ref) that names [`cross_val_predict`](@ref) is thrown otherwise.

# Related

  - [`reset_time_dependent_estimator`](@ref)
  - [`TimeDependentDefaultError`](@ref)
"""
reset_time_dependent_step(est) = est
function reset_time_dependent_step(x::Union{<:OptE_Opt, <:Pipeline})
    return reset_time_dependent_estimator(x)
end
function reset_time_dependent_step(td::TD_OptE_Opt)
    if isa(td.default, NoDefault)
        throw(TimeDependentDefaultError("a TimeDependent schedule is the optimisation step of a Pipeline but supplies no `default`, so a fold-less fit has no optimiser to run. A schedule is defined only over the folds of a cross-validation scheme; fit has none. Give the schedule a fold-less optimiser (TimeDependent(val; default = opt)), or backtest the pipeline with cross_val_predict, whose folds the schedule resolves against."))
    end
    return reset_time_dependent_estimator(td.default)
end
function reset_time_dependent_step(ps::PipelineStep)
    est = ps.est
    if isa(est, TimeDependent)
        return reset_time_dependent_step(est)
    end
    if isa(est, Function)
        return ps
    end
    newest = reset_time_dependent_step(est)
    return newest === est ? ps : PipelineStep(newest, ps.reads, ps.writes, ps.target)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace each schedule of a [`Pipeline`](@ref) with its value outside a fold, through [`reset_time_dependent_step`](@ref).

The fold-less [`fit`](@ref) calls this function first when the pipeline [`is_time_dependent`](@ref). A pipeline that [`update_time_dependent_estimator`](@ref) resolved for one fold holds no schedule, so a fit inside the fold loop does not call it.

# Validation

  - Everything [`reset_time_dependent_step`](@ref) refuses.

# Related

  - [`reset_time_dependent_step`](@ref)
  - [`fit`](@ref)
"""
function reset_time_dependent_estimator(p::Pipeline)
    return Pipeline(p.names, map(reset_time_dependent_step, p.steps), p.cache)
end
"""
    pipeline_step_factory(est, w)

Give the weights `w` of the previous fold to one [`Pipeline`](@ref) step.

# Algorithm

 1. Give `w` to an optimisation step and to a nested pipeline through [`factory`](@ref). A turnover, a fee and a tracking term read the weights, and every other field passes through.
 2. Return a [`PipelineStep`](@ref) that wraps a callable unchanged. Give `w` to the estimator of any other [`PipelineStep`](@ref) by this algorithm, giving `newest`. Rebuild the wrapper around `newest` when `newest` is a new value.
 3. Return every other step unchanged.

# Related

  - [`factory`](@ref)
  - [`cross_val_predict`](@ref)
"""
pipeline_step_factory(est, ::Any) = est
function pipeline_step_factory(x::Union{<:OptE_Opt, <:Pipeline}, w)
    return factory(x, w)
end
function pipeline_step_factory(ps::PipelineStep, w)
    est = ps.est
    if isa(est, Function)
        return ps
    end
    newest = pipeline_step_factory(est, w)
    return newest === est ? ps : PipelineStep(newest, ps.reads, ps.writes, ps.target)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a [`Pipeline`](@ref) with the weights `w` of the previous fold in each optimisation step, through [`pipeline_step_factory`](@ref).

The fold loop applies it after the swap of the schedules, so an optimiser that the swap puts in also receives the weights.

# Related

  - [`pipeline_step_factory`](@ref)
  - [`cross_val_predict`](@ref)
"""
function factory(p::Pipeline, w::VecNum)
    return Pipeline(p.names, map(est -> pipeline_step_factory(est, w), p.steps), p.cache)
end
"""
    cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CombinatorialCrossValidation; ex = FLoops.ThreadedEx(), kwargs...) -> PopulationPredictionResult

Run combinatorial cross-validation over a [`Pipeline`](@ref) on price-level or returns-level data.

Each split fits the whole pipeline on its training rows and predicts each of its test groups. [`sort_predictions`](@ref) then puts the predictions of the test groups into the paths of the scheme, as the combinatorial loop of an optimiser does. A schedule step resolves for each split against the [`TimeDependentContext`](@ref) of the split, before `fit` runs.

The training rows of a split can be non-contiguous, because a test group that the split holds out can sit between them. On returns-level data, each fold fits on returns that the data holds, and the method adds no approximation. On price-level data, [`PricesToReturns`](@ref) converts the training prices of the fold, and it makes one return across each gap. That return spans the prices from the last row before the gap to the first row after it. So a training window in `b` blocks holds `b - 1` returns that span a gap. The test groups are contiguous, so no prediction holds such a return. Use [`MultipleRandomised`](@ref) for contiguous training rows on price-level data.

`ex` is the FLoops executor of the splits. The method accepts every other keyword, such as the `id` of the single-path method, and ignores it.

# Algorithm

 1. Check the entry with [`assert_pipeline_entry`](@ref).
 2. Split `data` with `cv`, giving `train_idx` and `test_idx`, and refuse shuffled folds with [`assert_unshuffled_folds`](@ref).
 3. Read the evaluation keywords of `cv` with [`fold_evaluation`](@ref), giving `wd`, `pws`, `fa`, `store_weight_path` and `strict`. Read the drift of the held weights with [`held_weights_drift`](@ref), giving `hwd`.
 4. Through [`fold_loop`](@ref), fit the pipeline on the training rows of each split with [`pipeline_fold_fit`](@ref), and predict each test group of the split. This gives one vector of predictions for each split.
 5. Put the predictions into the paths of the scheme with [`sort_predictions`](@ref), and return the paths as a [`PopulationPredictionResult`](@ref).

# Validation

  - Everything [`assert_pipeline_entry`](@ref), [`assert_unshuffled_folds`](@ref) and [`fold_loop`](@ref) refuse.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
"""
function cross_val_predict(pipe::Pipeline, data::Prices_RR,
                           cv::CombinatorialCrossValidation;
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(), kwargs...)
    assert_pipeline_entry(pipe, cv)
    cv_res = split(cv, data)
    (; train_idx, test_idx) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    (; wd, pws, fa, store_weight_path, strict) = fold_evaluation(cv)
    hwd = held_weights_drift(wd, pws)
    predictions, _ = fold_loop(pipe, length(train_idx), ex, Vector{PredictionResult};
                               rd = data, train_idx = train_idx, test_idx = test_idx,
                               cv = cv) do fold
        res = pipeline_fold_fit(fold.est, fold.rd, fold.train)
        return [StatsAPI.predict(res, fold.rd, group; wd = wd, hwd = hwd, fa = fa,
                                 store_weight_path = store_weight_path, strict = strict,
                                 w_prev = fold.w_prev) for group in fold.test]
    end
    return PopulationPredictionResult(; pred = sort_predictions(cv_res, predictions))
end
"""
    pipeline_path_fit_and_predict(pipe::Pipeline_OnlPipe, data::Prices_RR, folds, path_id; ex, wd, hwd, fa, pws, store_weight_path, strict, cv) -> MultiPeriodPredictionResult

Run one path of [`MultipleRandomised`](@ref) over a [`Pipeline`](@ref) on price-level or returns-level data. The path is an inner walk-forward over a subset of the assets.

# Algorithm

 1. Read the training rows `train_idx` and the test rows `test_idx` of each fold from `folds`. `folds` holds one `(train, test, asset)` tuple for each fold of the path, in the order of the split.
 2. For fold `i`, view `data` on the asset subset of the fold with [`pipeline_asset_view`](@ref), which indexes the assets the same way at both levels. The pipeline fits again on each subset, so it never selects from a fitted universe.
 3. Through [`fold_loop`](@ref), fit the pipeline on the training rows of each fold with [`pipeline_fold_fit`](@ref), and predict its test rows. The loop resolves each schedule for the fold, and the context holds `path_id`. When the folds of `cv` are a timeline and the pipeline [`needs_previous_weights`](@ref), the loop runs the folds of the path in sequence and gives each fold the weights of the previous fold. This gives the predictions and `est`, the estimator that the online arm threaded.
 4. Sort the predictions by their test rows with [`sort_predictions`](@ref).

# Returns

  - [`MultiPeriodPredictionResult`](@ref): The predictions of the path, with the `id` `path_id` and the estimator `est`.

# Related

  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::MultipleRandomised)`](@ref)
  - [`path_fit_and_predict`](@ref)
"""
function pipeline_path_fit_and_predict(pipe::Pipeline_OnlPipe, data::Prices_RR, folds,
                                       path_id;
                                       ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                       wd::Option{<:AbstractWeightDrift} = nothing,
                                       hwd::Option{<:AbstractWeightDrift} = wd,
                                       fa::Option{<:AbstractFeeAmortisation} = nothing,
                                       pws::Option{<:AbstractPreviousWeightsSource} = nothing,
                                       store_weight_path::Bool = false,
                                       strict::Bool = false, cv = nothing)
    train_idx = map(x -> x[1], folds)
    test_idx = map(x -> x[2], folds)
    asset_view(i) = (pipe, pipeline_asset_view(data, folds[i][3]))
    predictions, est = fold_loop(pipe, length(folds), ex; rd = data, train_idx = train_idx,
                                 test_idx = test_idx, path_id = path_id,
                                 fold_view = asset_view, pws = pws, cv = cv) do fold
        res = pipeline_fold_fit(fold.est, fold.rd, fold.train)
        return StatsAPI.predict(res, fold.rd, fold.test; wd = wd, hwd = hwd, fa = fa,
                                store_weight_path = store_weight_path, strict = strict,
                                w_prev = fold.w_prev)
    end
    return MultiPeriodPredictionResult(; pred = sort_predictions(test_idx, predictions),
                                       id = path_id, opt = est)
end
"""
    cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::MultipleRandomised; ex = FLoops.ThreadedEx(), kwargs...) -> PopulationPredictionResult

Run asset-resampling cross-validation, [`MultipleRandomised`](@ref), over a [`Pipeline`](@ref) on price-level or returns-level data.

Each path is an inner walk-forward over a random subset of the assets. The subset applies to the input data as a view through [`pipeline_asset_view`](@ref), and the pipeline fits again on the subset. So the pipeline never selects from its fitted universe. The scheme draws assets and not rows, so each window of observations stays contiguous. Unlike combinatorial cross-validation, the method adds no approximation on price-level data.

`ex` is the FLoops executor of the paths, and of the folds in each path. The method accepts every other keyword and ignores it. [`pipeline_cross_val_predict`](@ref) holds the steps.

# Returns

  - [`PopulationPredictionResult`](@ref): One [`MultiPeriodPredictionResult`](@ref) for each path.

# Related

  - [`MultipleRandomised`](@ref)
  - [`pipeline_path_fit_and_predict`](@ref)
  - [`pipeline_asset_view`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
"""
function cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::MultipleRandomised;
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(), kwargs...)
    return pipeline_cross_val_predict(pipe, data, cv; ex = ex)
end
"""
    pipeline_cross_val_predict(pipe::Pipeline_OnlPipe, data::Prices_RR, cv::MultipleRandomised; ex = FLoops.ThreadedEx())
    pipeline_cross_val_predict(pipe::Pipeline_OnlPipe, data::Prices_RR, cv::CVER; ex = FLoops.ThreadedEx(), id = nothing)

Run the body of the `cross_val_predict` methods of a Pipeline, for a [`Pipeline`](@ref), for `Online(pipe)` and for `Resume(res)`.

# Algorithm

 1. Check the entry with [`assert_pipeline_entry`](@ref).
 2. Split `data` with `cv`, giving `train_idx` and `test_idx`, and refuse shuffled folds with [`assert_unshuffled_folds`](@ref).
 3. Read the evaluation keywords of `cv` with [`fold_evaluation`](@ref), and the drift of the held weights with [`held_weights_drift`](@ref), giving `hwd`.
 4. For [`MultipleRandomised`](@ref), put the `(train, test, asset)` tuples of the folds into one vector for each path, giving `dict`. Run each path with [`pipeline_path_fit_and_predict`](@ref) through [`parallel_folds`](@ref), and return the paths as a [`PopulationPredictionResult`](@ref).
 5. For a scheme with contiguous test rows, run the folds through [`fold_loop`](@ref). Each fold fits with [`pipeline_fold_fit`](@ref) and predicts its test rows. The online arm of the loop resolves `Online(pipe)` at its warm-up, and passes the pipeline from each fold to the next. Return the predictions as a [`MultiPeriodPredictionResult`](@ref) with `id` and the estimator that the online arm threaded.

# Validation

  - Everything [`assert_pipeline_entry`](@ref), [`assert_unshuffled_folds`](@ref) and [`fold_loop`](@ref) refuse.

# Related

  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
  - [`cross_val_predict(o::Online{<:Pipeline}, data::Prices_RR, cv::CVER)`](@ref)
  - [`Pipeline_OnlPipe`](@ref)
"""
function pipeline_cross_val_predict(pipe::Pipeline_OnlPipe, data::Prices_RR,
                                    cv::MultipleRandomised;
                                    ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    assert_pipeline_entry(pipe, cv)
    cv_res = split(cv, data)
    (; train_idx, test_idx, asset_idx, path_ids) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    unique_ids = unique(path_ids)
    dict = [Vector{Tuple{eltype(train_idx), eltype(test_idx), eltype(asset_idx)}}(undef, 0)
            for _ in unique_ids]
    for (train, test, asset, path_id) in zip(train_idx, test_idx, asset_idx, path_ids)
        push!(dict[path_id], (train, test, asset))
    end
    (; wd, pws, fa, store_weight_path, strict) = fold_evaluation(cv)
    hwd = held_weights_drift(wd, pws)
    predictions = parallel_folds(length(unique_ids), ex, MultiPeriodPredictionResult) do i
        return pipeline_path_fit_and_predict(pipe, data, dict[i], i; ex = ex, wd = wd,
                                             hwd = hwd, fa = fa, pws = pws,
                                             store_weight_path = store_weight_path,
                                             strict = strict, cv = cv)
    end
    return PopulationPredictionResult(; pred = predictions)
end
"""
    cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER = KFold(); ex = FLoops.ThreadedEx(), id = nothing)

Run cross-validated prediction over a whole [`Pipeline`](@ref), and return a [`MultiPeriodPredictionResult`](@ref).

# Return type by cross-validation scheme

`cross_val_predict` is the one entry point for every scheme, but the type of the Result depends on `cv`. A single-path scheme returns one series, and a multi-path scheme returns one series for each path. Code that reads the Result of one scheme, such as a `KFold` run, fails on the Result of the other kind. Branch on the scheme, not on the run.

| `cv` scheme                                     | Return type                           | `.pred` holds                       |
|:----------------------------------------------- |:------------------------------------- |:----------------------------------- |
| [`KFold`](@ref) / walk-forward ([`CVER`](@ref)) | [`MultiPeriodPredictionResult`](@ref) | one series, one prediction per fold |
| [`CombinatorialCrossValidation`](@ref)          | [`PopulationPredictionResult`](@ref)  | a per-path collection               |
| [`MultipleRandomised`](@ref)                    | [`PopulationPredictionResult`](@ref)  | a per-path collection               |

The combinatorial scheme and the asset-resampling scheme have their own methods, and both take price-level or returns-level data. See [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CombinatorialCrossValidation)`](@ref) and [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::MultipleRandomised)`](@ref). The rest of this docstring is about the method for the contiguous, single-path schemes, [`KFold`](@ref) and the walk-forwards.

The method splits the input at its own level. The `split` methods of price-level data give contiguous windows, so a preprocessing step with state stays inside the fold. For each fold, the method fits the whole pipeline on the training window and predicts the test window, as [`fit`](@ref) and [`predict`](@ref) do for a holdout. [`pipeline_cross_val_predict`](@ref) holds the steps.

When the pipeline is time-dependent, fold `i` makes a [`TimeDependentContext`](@ref), and [`update_time_dependent_estimator`](@ref) swaps each schedule for its value at fold `i` before `fit` runs. The `rd` of the context is the input `data` before any step transforms it, so a callable of the pipeline sees the raw data of the fold. A schedule step can resolve to an estimator, and the fold optimises, or to a precomputed result, and the fold only predicts.

The scheme states through [`folds_are_time_ordered`](@ref) whether its folds are a timeline. A walk-forward answers `true`. So the folds of a pipeline that [`needs_previous_weights`](@ref) run in sequence. Each fold receives the weights of the previous fold in `w_prev` of the context. After the swap, [`factory`](@ref) also gives them to the optimisation steps. A [`KFold`](@ref) answers `false`, because its folds do not depend on each other. Its folds run in parallel with no previous weights and no [`factory`](@ref) pass, as the `KFold` path of an optimiser does.

Under an Online Scheme, the loop takes its online arm. The loop warms the pipeline up once, on the first training window. [`partial_fit!`](@ref) folds the new rows of each fold through the steps into the row owner. Where a refit would run, the fold calls `fit(pipe)` with no data, through [`pipeline_fold_fit`](@ref). At each fold, the run gives the same weights as the batch expanding walk-forward. `Online(pipe)`, the declared refit from a buffer of the input data, goes through the same `cross_val_predict`.

# Arguments

  - `pipe`: The pipeline.
  - `data`: Price-level or returns-level input data ([`Prices_RR`](@ref)).
  - `cv::CVER`: The cross-validation scheme, with contiguous test folds and one path. The default is `KFold()`. [`folds_are_time_ordered`](@ref) decides whether its folds receive the weights of the previous fold, and [`folds_are_stepped`](@ref) decides whether the folds refit or fold.
  - `ex`: The FLoops executor of the folds. The default is `FLoops.ThreadedEx()`.
  - `id`: The identifier that the Result stores.

# Validation

  - The pipeline holds no holdout step, and an `Online(pipe)` runs only under an Online Scheme ([`assert_pipeline_entry`](@ref)).
  - The folds of `cv` are not shuffled ([`assert_unshuffled_folds`](@ref)).
  - The number of entries of each schedule is the number of folds, and the batch arms hold no [`Online`](@ref) ([`fold_loop`](@ref)).

# Returns

  - [`MultiPeriodPredictionResult`](@ref): One prediction for each fold, in the order of the folds.

# Related

  - [`Pipeline`](@ref)
  - [`fit`](@ref)
  - [`TimeDependent`](@ref)
  - [`folds_are_time_ordered`](@ref)
  - [`search_cross_validation`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CombinatorialCrossValidation)`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::MultipleRandomised)`](@ref)
"""
function cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER = KFold();
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                           id = nothing)
    return pipeline_cross_val_predict(pipe, data, cv; ex = ex, id = id)
end
function pipeline_cross_val_predict(pipe::Pipeline_OnlPipe, data::Prices_RR, cv::CVER;
                                    ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                    id = nothing)
    assert_pipeline_entry(pipe, cv)
    cv_res = split(cv, data)
    (; train_idx, test_idx) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    (; wd, pws, fa, store_weight_path, strict) = fold_evaluation(cv)
    hwd = held_weights_drift(wd, pws)
    predictions, est = fold_loop(pipe, length(train_idx), ex; rd = data,
                                 train_idx = train_idx, test_idx = test_idx, cv = cv,
                                 pws = pws) do fold
        res = pipeline_fold_fit(fold.est, fold.rd, fold.train)
        return StatsAPI.predict(res, fold.rd, fold.test; wd = wd, hwd = hwd, fa = fa,
                                store_weight_path = store_weight_path, strict = strict,
                                w_prev = fold.w_prev)
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id, opt = est)
end
#! End: TimeDependent schedules as pipeline optimisation steps.
