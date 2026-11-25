"""
    cross_val_predict(opt, rd::ReturnsResult, cv::CVER = KFold(); cols = :, ex = FLoops.ThreadedEx())
    cross_val_predict(opt, rd::ReturnsResult, cv::CVER, cv_res::CrossValidationResult; ex = FLoops.ThreadedEx())

Run cross-validated portfolio optimisation and return predictions over all folds.

Accepts either an optimisation estimator or an optimisation result. When `cols` is
provided, restricts the optimisation to that subset of assets. Parallel fold execution
is controlled by `ex`.

The form with `cv_res` runs the folds of a split that the caller made already. A caller that runs many estimators over one scheme splits once, and every estimator then scores the same folds. That form takes no `cols`, because the split belongs to the data as the caller passes it. View `rd` by the columns before the split.

# Arguments

  - `opt`: Optimisation estimator or result.
  - `rd::ReturnsResult`: Returns data used for fitting and prediction.
  - `cv::CVER`: Cross-validation scheme. Defaults to `KFold()`.
  - $(arg_dict[:cv_res])
  - `cols`: Column selector. Defaults to `:` (all assets).
  - `ex`: FLoops executor controlling parallelism. Defaults to `FLoops.ThreadedEx()`.

# Returns

  - Cross-validation prediction result.

# Related

  - [`KFold`](@ref)
  - [`WalkForwardEstimator`](@ref)
  - [`fit_and_predict`](@ref)
"""
function cross_val_predict(opt::OptE_TD, rd::ReturnsResult, cv::CVER = KFold(); cols = :,
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    assert_internal_optimiser(opt)
    assert_external_optimiser(opt)
    if !isa(cols, Colon)
        # The view indexes the matrix by `cols`, so it reads the matrix before the cut.
        opt = port_opt_view(opt, cols, rd.X)
        rd = port_opt_view(rd, cols)
    end
    return fit_and_predict(opt, rd, cv; ex = ex)
end
function cross_val_predict(opt::OptE_TD, rd::ReturnsResult, cv::CVER,
                           cv_res::CrossValidationResult;
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    assert_internal_optimiser(opt)
    assert_external_optimiser(opt)
    return fit_and_predict(opt, rd, cv, cv_res; ex = ex)
end
function cross_val_predict(opt::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                           cv::CVER = KFold(),
                           cv_res::CrossValidationResult = split(cv, rd);
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    return fit_and_predict(opt, rd, cv, cv_res; ex = ex)
end

export cross_val_predict
