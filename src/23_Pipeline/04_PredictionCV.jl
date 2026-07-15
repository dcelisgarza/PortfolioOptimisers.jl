"""
    Base.split(ccv::CombinatorialCrossValidation, pr::AbstractPricesResult)

Unsupported at the **price level**: combinatorial cross-validation recombines non-contiguous test groups, and a pipeline that *starts from prices* runs a rolling, order-dependent transform (a [`PricesToReturns`](@ref) needs each row's predecessor; any windowed preprocessing needs contiguous history) that cannot be fitted or replayed across the gaps between groups. Throws an `ArgumentError`. This is the *rolling-window rule* — it applies only to price-starting pipelines; a **returns-level** pipeline has no such transform, so `cross_val_predict(pipe, rd::AbstractReturnsResult, ccv)` runs combinatorial folds like the plain-optimiser path (training rows may be non-contiguous, predictions recombine into paths).

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::CombinatorialCrossValidation)`](@ref)
  - [`Base.split(kf::KFold, rd::Prices_RR)`](@ref)
"""
function Base.split(::CombinatorialCrossValidation, ::AbstractPricesResult)
    return throw(ArgumentError("CombinatorialCrossValidation is unsupported for a price-starting pipeline: its recombined, non-contiguous test groups break the rolling/price-level preprocessing (e.g. PricesToReturns) that needs contiguous windows — the rolling-window rule. Run it on a returns-level pipeline instead (cross_val_predict on an AbstractReturnsResult), or use KFold / a walk-forward scheme at the price level."))
end
"""
    Base.split(mrcv::MultipleRandomised, pr::AbstractPricesResult)

Unsupported at the **price level**, for the same *rolling-window rule* as [`Base.split(ccv::CombinatorialCrossValidation, pr::AbstractPricesResult)`](@ref): a price-starting pipeline's rolling, order-dependent preprocessing needs contiguous input-row windows, which the resampled paths do not guarantee. Throws an `ArgumentError`. A **returns-level** pipeline has no such transform, so `cross_val_predict(pipe, rd::AbstractReturnsResult, mrcv)` resamples asset subsets and runs each path's inner walk-forward folds normally.

# Related

  - [`MultipleRandomised`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::MultipleRandomised)`](@ref)
  - [`Base.split(kf::KFold, rd::Prices_RR)`](@ref)
"""
function Base.split(::MultipleRandomised, ::AbstractPricesResult)
    return throw(ArgumentError("MultipleRandomised is unsupported for a price-starting pipeline: its resampled paths do not guarantee the contiguous input-row windows that rolling/price-level preprocessing needs — the rolling-window rule. Run it on a returns-level pipeline instead (cross_val_predict on an AbstractReturnsResult), or use KFold / a walk-forward scheme at the price level."))
end
#! Begin: TimeDependent schedules as pipeline optimisation steps.
"""
    pipeline_step_is_time_dependent(x)

Return `true` if a [`Pipeline`](@ref) step carries time-dependent constraints.

Steps that participate in the time-dependent machinery — optimisation estimators and results, [`TimeDependent`](@ref) schedules, nested [`Pipeline`](@ref)s, and [`PipelineStep`](@ref) wrappers around any of them — delegate to [`is_time_dependent`](@ref); every other step (preprocessing, prior, phylogeny, uncertainty, constraint estimators, and custom callables) contributes `false`, because no non-optimiser family hosts schedules (they are spelled as fields of the optimisation step instead).

# Related

  - [`is_time_dependent`](@ref)
  - [`Pipeline`](@ref)
"""
pipeline_step_is_time_dependent(::Any) = false
function pipeline_step_is_time_dependent(x::Union{<:OptE_Opt, <:TimeDependent, <:Pipeline,
                                                  <:PipelineStep})
    return is_time_dependent(x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any step of the [`Pipeline`](@ref) carries time-dependent constraints: a [`TimeDependent`](@ref) schedule standing in for the optimisation step, an optimisation step whose fields hold schedules, or a nested pipeline containing either.

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

Return `true` if any step of the [`Pipeline`](@ref) requires the previous fold's portfolio weights, forcing sequential fold execution and a populated `w_prev` in the [`TimeDependentContext`](@ref). Recurses through [`PipelineStep`](@ref) wrappers and nested pipelines; schedule entries are inspected per the [`needs_previous_weights`](@ref) conventions (entries yes, `default` no).

# Related

  - [`needs_previous_weights`](@ref)
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

Assert the fold count of one [`Pipeline`](@ref) step: optimisation steps, [`TimeDependent`](@ref) schedule steps, and nested pipelines delegate to [`assert_time_dependent_fold_count`](@ref) (schedule entries are sized to the loop, the `default` is not); every other step is a no-op.

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

Resolve one [`Pipeline`](@ref) step for the fold described by `ctx` — the per-step leg of the pipeline swap.

A [`TimeDependent`](@ref) schedule step resolves to its fold-`i` entry (an estimator or a precomputed result) and is recursed into with the same context; an optimisation step resolves its own scheduled fields; a nested pipeline recurses. A [`PipelineStep`](@ref) wrapping a schedule is *unwrapped* to the resolved value — the wrapper existed only to declare the slots dispatch could not infer, and after the swap [`run_step`](@ref) dispatches on the resolved value directly (which a wrapper could not hold anyway when the entry is a result). Every other step passes through unchanged.

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

Resolve the time-dependent steps of a [`Pipeline`](@ref) for the fold described by `ctx` — the swap of ADR 0030's pipeline integration.

The swap happens in the fold loop, *outside* [`fit`](@ref) entirely: it maps [`update_time_dependent_step`](@ref) over the steps (names preserved), so by the time `fit` runs on the fold's training window every schedule step is already a plain optimiser or precomputed result and injection ([`inject_context`](@ref) / [`maybe_inject_step`](@ref)) never sees a schedule — `fit` and [`run_step`](@ref) never learn about folds.

# Related

  - [`update_time_dependent_step`](@ref)
  - [`cross_val_predict`](@ref)
"""
function update_time_dependent_estimator(p::Pipeline, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    return Pipeline(p.names,
                    map(est -> update_time_dependent_step(est, ctx, all_binds), p.steps))
end
"""
    reset_time_dependent_step(est)

Replace one [`Pipeline`](@ref) step with its fold-less value.

A [`TimeDependent`](@ref) schedule step resolves to its explicit `default` (there is no static default an optimisation step could fall back to), throwing a [`TimeDependentDefaultError`](@ref) that points at [`cross_val_predict`](@ref) when it has none; optimisation steps and nested pipelines recurse through [`reset_time_dependent_estimator`](@ref); a [`PipelineStep`](@ref) wrapping a schedule is unwrapped to the fold-less optimiser. Every other step passes through unchanged.

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

Replace every time-dependent step of a [`Pipeline`](@ref) with its fold-less value (see [`reset_time_dependent_step`](@ref)). Called at the top of the fold-less [`fit`](@ref); per-fold pipelines produced by [`update_time_dependent_estimator`](@ref) contain no schedules, so they pass through unchanged.

# Related

  - [`reset_time_dependent_step`](@ref)
  - [`fit`](@ref)
"""
function reset_time_dependent_estimator(p::Pipeline)
    return Pipeline(p.names, map(reset_time_dependent_step, p.steps))
end
"""
    pipeline_step_factory(est, w)

Deliver the previous fold's weights `w` to one [`Pipeline`](@ref) step: optimisation steps and nested pipelines go through [`factory`](@ref) (turnover, fees and tracking pick the weights up; everything else passes through), [`PipelineStep`](@ref)-wrapped optimisers are rebuilt around the updated estimator, and non-optimiser steps pass through unchanged.

# Related

  - [`factory`](@ref)
  - [`cross_val_predict`](@ref)
"""
pipeline_step_factory(est, ::Any) = est
function pipeline_step_factory(x::Union{<:OptE_Opt, <:Pipeline}, w)
    return factory(x, w)
end
function pipeline_step_factory(ps::PipelineStep, w)
    if isa(ps.est, OptimisationEstimator)
        return PipelineStep(factory(ps.est, w), ps.reads, ps.writes, ps.target)
    end
    return ps
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a [`Pipeline`](@ref) with the previous fold's weights delivered to every optimisation step (see [`pipeline_step_factory`](@ref)). Applied by the fold loop *after* the swap, so freshly swapped-in per-fold optimisers receive the previous weights too.

# Related

  - [`pipeline_step_factory`](@ref)
  - [`cross_val_predict`](@ref)
"""
function factory(p::Pipeline, w::VecNum)
    return Pipeline(p.names, map(est -> pipeline_step_factory(est, w), p.steps))
end
"""
    cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::CombinatorialCrossValidation; ex = FLoops.ThreadedEx(), kwargs...) -> PopulationPredictionResult

Run combinatorial cross-validation over a **returns-level** [`Pipeline`](@ref).

A returns-level pipeline runs no rolling, price-level transform, so the rolling-window rule that blocks combinatorial for price-starting pipelines (see [`Base.split(ccv::CombinatorialCrossValidation, pr::AbstractPricesResult)`](@ref)) does not apply. Each split fits the whole workflow on its (possibly non-contiguous) training rows — moment-style fitted steps are order-independent — and predicts each of the split's disjoint test groups; [`sort_predictions!`](@ref) then recombines the per-split test-group predictions into the scheme's paths, exactly like the plain-optimiser combinatorial loop. Time-dependent steps resolve per split against the fold's [`TimeDependentContext`](@ref) before `fit`.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
"""
function cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult,
                           cv::CombinatorialCrossValidation;
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(), kwargs...)
    assert_no_holdout(pipe)
    cv_res = split(cv, data)
    (; train_idx, test_idx) = cv_res
    n = length(train_idx)
    td_flag = is_time_dependent(pipe)
    if td_flag
        assert_time_dependent_fold_count(pipe, n)
    end
    predictions = parallel_folds(n, ex; ElT = Vector{PredictionResult}) do i
        pipei = pipe
        if td_flag
            ctx = TimeDependentContext(; i = i, n = n, rd = data, train_idx = train_idx,
                                       test_idx = test_idx)
            pipei = update_time_dependent_estimator(pipei, ctx)
        end
        res = StatsAPI.fit(pipei, pipeline_data_view(data, train_idx[i]))
        return [StatsAPI.predict(res, data, group) for group in test_idx[i]]
    end
    return PopulationPredictionResult(; pred = sort_predictions!(cv_res, predictions))
end
"""
    pipeline_path_fit_and_predict(pipe::Pipeline, data::AbstractReturnsResult, folds, path_id; ex) -> MultiPeriodPredictionResult

Run one [`MultipleRandomised`](@ref) path of a returns-level [`Pipeline`](@ref): fit and predict the path's inner walk-forward `folds` over the path's asset subset.

`folds` is the path's `(train_idx, test_idx, asset_idx)` tuples in split-enumeration order. Each fold takes the asset-subset view of `data` (via [`port_opt_view`](@ref)) — the pipeline fits fresh on the sub-universe, so it never sub-selects fitted state — then fits on the training window and predicts on the test window. Time-dependent steps resolve per fold (the context carries `path_id`); when the pipeline [`needs_previous_weights`](@ref), [`run_folds`](@ref) runs the path sequentially and threads the previous fold's weights. Predictions are sorted by test index for reporting.

# Related

  - [`cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::MultipleRandomised)`](@ref)
  - [`path_fit_and_predict`](@ref)
"""
function pipeline_path_fit_and_predict(pipe::Pipeline, data::AbstractReturnsResult, folds,
                                       path_id;
                                       ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    n = length(folds)
    td_flag = is_time_dependent(pipe)
    if td_flag
        assert_time_dependent_fold_count(pipe, n)
    end
    prev_w_flag = needs_previous_weights(pipe)
    train_idx = map(x -> x[1], folds)
    test_idx = map(x -> x[2], folds)
    predictions = run_folds(pipe, n, ex) do i, prev
        (tr, te, as) = folds[i]
        rdi = port_opt_view(data, as)
        pipei = pipe
        if td_flag
            ctx = TimeDependentContext(; i = i, n = n, rd = rdi, train_idx = train_idx,
                                       test_idx = test_idx,
                                       w_prev = isnothing(prev) ? nothing : prev.res.w,
                                       path_id = path_id)
            pipei = update_time_dependent_estimator(pipei, ctx)
        end
        if !isnothing(prev) && prev_w_flag
            pipei = factory(pipei, prev.res.w)
        end
        res = StatsAPI.fit(pipei, pipeline_data_view(rdi, tr))
        return StatsAPI.predict(res, rdi, te)
    end
    return MultiPeriodPredictionResult(; pred = sort_predictions!(test_idx, predictions),
                                       id = path_id)
end
"""
    cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::MultipleRandomised; ex = FLoops.ThreadedEx(), kwargs...) -> PopulationPredictionResult

Run asset-resampling (multiple-randomised) cross-validation over a **returns-level** [`Pipeline`](@ref).

The rolling-window rule that blocks this for price-starting pipelines (see [`Base.split(mrcv::MultipleRandomised, pr::AbstractPricesResult)`](@ref)) does not apply to a returns-level pipeline. Each resampled path is an inner walk-forward over a random asset subset; the subset is applied to the *input data* (an asset view), and the pipeline is fitted fresh on the sub-universe — so, contrary to the earlier restriction, the pipeline never needs to sub-select its fitted universe. Paths are run by [`pipeline_path_fit_and_predict`](@ref) and returned as a [`PopulationPredictionResult`](@ref).

# Related

  - [`MultipleRandomised`](@ref)
  - [`pipeline_path_fit_and_predict`](@ref)
  - [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER)`](@ref)
"""
function cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult,
                           cv::MultipleRandomised;
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(), kwargs...)
    assert_no_holdout(pipe)
    cv_res = split(cv, data)
    (; train_idx, test_idx, asset_idx, path_ids) = cv_res
    unique_ids = unique(path_ids)
    dict = [Vector{Tuple{eltype(train_idx), eltype(test_idx), eltype(asset_idx)}}(undef, 0)
            for _ in unique_ids]
    for (train, test, asset, path_id) in zip(train_idx, test_idx, asset_idx, path_ids)
        push!(dict[path_id], (train, test, asset))
    end
    predictions = parallel_folds(length(unique_ids), ex; ElT = MultiPeriodPredictionResult
                                 ) do i
        return pipeline_path_fit_and_predict(pipe, data, dict[i], i; ex = ex)
    end
    return PopulationPredictionResult(; pred = predictions)
end
"""
    cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER = KFold(); ex = FLoops.ThreadedEx(), id = nothing)

Run cross-validated prediction over an entire [`Pipeline`](@ref) workflow and return a [`MultiPeriodPredictionResult`](@ref).

The input is split at its own level — price-level data by the prices-aware `split` methods (contiguous windows, so stateful preprocessing stays inside the fold), returns-level data as usual — and for each fold the whole workflow is fitted on the training window and predicts on the test window, exactly as [`fit`](@ref)/[`predict`](@ref) do for a holdout. This method covers the contiguous, single-path schemes ([`KFold`](@ref) and the walk-forwards). Combinatorial and asset-resampling schemes have their own methods for a **returns-level** pipeline (see [`cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::CombinatorialCrossValidation)`](@ref) and [`cross_val_predict(pipe::Pipeline, data::AbstractReturnsResult, cv::MultipleRandomised)`](@ref)); for a **price-starting** pipeline they are rejected at `split` by the rolling-window rule.

This is the fold loop that consumes [`TimeDependent`](@ref) schedules in a pipeline (ADR 0030): when the pipeline is time-dependent, fold `i` builds a [`TimeDependentContext`](@ref) — with `rd` the *raw, pre-preprocessing* input `data`, so pipeline-level callables see the fold's data before any step has transformed it — and swaps every schedule for its fold-`i` value via [`update_time_dependent_estimator`](@ref) **before** `fit` runs. A schedule step may resolve to an estimator (the fold optimises) or a precomputed result (the fold predicts only); injection never sees a schedule. When the pipeline [`needs_previous_weights`](@ref), [`run_folds`](@ref) runs sequentially and threads the previous fold's weights into the context's `w_prev` and, post-swap, into the optimisation steps via [`factory`](@ref).

# Arguments

  - `pipe`: The pipeline.
  - `data`: Price- or returns-level input data ([`Prices_RR`](@ref)).
  - `cv::CVER`: Cross-validation scheme with contiguous, non-combinatorial folds. Defaults to `KFold()`.
  - `ex`: FLoops executor controlling parallelism. Defaults to `FLoops.ThreadedEx()`.
  - `id`: Identifier stored on the result.

# Returns

  - [`MultiPeriodPredictionResult`](@ref): One prediction per fold, in fold order.

# Related

  - [`Pipeline`](@ref)
  - [`fit`](@ref)
  - [`TimeDependent`](@ref)
  - [`search_cross_validation`](@ref)
"""
function cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CVER = KFold();
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                           id = nothing)
    assert_no_holdout(pipe)
    @argcheck(!(hasfield(typeof(cv), :shuffle) && cv.shuffle),
              "Cross validation estimator must not be shuffled.")
    cv_res = split(cv, data)
    (; train_idx, test_idx) = cv_res
    @argcheck(isa(test_idx[1], VecInt),
              ArgumentError("pipeline cross-validation requires non-combinatorial (VecInt) test indices, but got $(typeof(test_idx[1])); combinatorial schemes recombine non-contiguous test groups, which a fitted workflow cannot replay"))
    n = length(train_idx)
    td_flag = is_time_dependent(pipe)
    if td_flag
        assert_time_dependent_fold_count(pipe, n)
    end
    prev_w_flag = needs_previous_weights(pipe)
    predictions = run_folds(pipe, n, ex) do i, prev
        pipei = pipe
        if td_flag
            ctx = TimeDependentContext(; i = i, n = n, rd = data, train_idx = train_idx,
                                       test_idx = test_idx,
                                       w_prev = isnothing(prev) ? nothing : prev.res.w)
            pipei = update_time_dependent_estimator(pipei, ctx)
        end
        if !isnothing(prev) && prev_w_flag
            pipei = factory(pipei, prev.res.w)
        end
        res = StatsAPI.fit(pipei, pipeline_data_view(data, train_idx[i]))
        return StatsAPI.predict(res, data, test_idx[i])
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id)
end
#! End: TimeDependent schedules as pipeline optimisation steps.
