"""
    cross_val_predict(opt, rd::ReturnsResult, cv::CVER = KFold(); cols = :, ex = FLoops.ThreadedEx())

Run cross-validated portfolio optimisation and return predictions over all folds.

Accepts either an optimisation estimator or an optimisation result. When `cols` is
provided, restricts the optimisation to that subset of assets. Parallel fold execution
is controlled by `ex`.

# Arguments

  - `opt`: Optimisation estimator or result.
  - `rd::ReturnsResult`: Returns data used for fitting and prediction.
  - `cv::CVER`: Cross-validation scheme. Defaults to `KFold()`.
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
        rd = port_opt_view(rd, cols)
        opt = port_opt_view(opt, cols, rd.X)
    end
    return fit_and_predict(opt, rd, cv; ex = ex)
end
function cross_val_predict(opt::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                           cv::CVER = KFold();
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    return fit_and_predict(opt, rd, cv; ex = ex)
end

export cross_val_predict
