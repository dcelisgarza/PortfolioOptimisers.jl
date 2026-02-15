function cross_val_predict(est::OptimisationEstimator, rd::ReturnsResult,
                           cv::CrossValidationEstimator = KFold(); cols = nothing,
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    if !isnothing(cols)
        rd = returns_result_view(rd, cols)
    end
    if hasproperty(cv, :shuffled) && cv.shuffled
        throw(ArgumentError("Cross validation estimator must not be shuffled."))
    end
    res = split(cv, rd)
    train_idx = res.train_idx
    test_idx = res.test_idx
    @argcheck(all(map((x, y) -> (x > zero(x) && y > zero(y)), diff([1]), diff(test_idx[1]))),
              ArgumentError("Cross validation estimator must not be shuffled."))

    return nothing
end

export cross_val_predict
