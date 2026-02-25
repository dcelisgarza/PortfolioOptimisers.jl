function cross_val_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                           cv::CrossValidationEstimator = KFold(); cols = :,
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    assert_internal_optimiser(opt)
    assert_external_optimiser(opt)
    if !isa(cols, Colon)
        rd = returns_result_view(rd, cols)
        opt = opt_view(opt, cols, rd.X)
    end
    @argcheck(!(hasproperty(cv, :shuffle) && cv.shuffle),
              "Cross validation estimator must not be shuffled.")
    res = split(cv, rd)
    @argcheck(all(map(x -> x > zero(x), map(x -> diff(x), res.train_idx))),
              "Cross validation estimator must not be shuffled.")
    return fit_and_predict(opt, rd, res; ex = ex)
end

export cross_val_predict
