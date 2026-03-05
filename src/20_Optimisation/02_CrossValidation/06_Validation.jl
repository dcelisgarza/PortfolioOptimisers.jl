function cross_val_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                           cv::CVER = KFold(); cols = :,
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    assert_internal_optimiser(opt)
    assert_external_optimiser(opt)
    if !isa(cols, Colon)
        rd = returns_result_view(rd, cols)
        opt = opt_view(opt, cols, rd.X)
    end
    return fit_and_predict(opt, rd, cv; ex = ex)
end
function cross_val_predict(opt::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                           cv::CVER = KFold();
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    return fit_and_predict(opt, rd, cv; ex = ex)
end

export cross_val_predict
