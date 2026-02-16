function cross_val_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                           cv::CrossValidationEstimator = KFold(); cols = :,
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    if !isa(cols, Colon)
        rd = returns_result_view(rd, cols)
        opt = opt_view(opt, cols)
    end
    if hasproperty(cv, :shuffle) && cv.shuffle
        throw(ArgumentError("Cross validation estimator must not be shuffled."))
    end
    res = split(cv, rd)
    @argcheck(all(map(x -> x > zero(x), map(x -> diff(x), res.train_idx))),
              "Cross validation estimator must not be shuffled.")
    predictions = fit_and_predict(opt, rd, res; cols = cols, ex = ex)
    return sort_predictions(res, predictions)
end

export cross_val_predict
