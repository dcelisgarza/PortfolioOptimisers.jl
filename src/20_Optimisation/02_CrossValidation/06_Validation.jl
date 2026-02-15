struct PredictionResult{T1, T2, T3} <: AbstractResult
    res::T1
    X::T2
    ts::T3
    function PredictionResult(res::NonFiniteAllocationOptimisationResult, X::VecNum,
                              ts::Option{<:VecDate})
        @argcheck(!isempty(X), IsEmptyError)
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
        end
        return new{typeof(res), typeof(X), typeof(ts)}(res, X, ts)
    end
end
function PredictionResult(; res::NonFiniteAllocationOptimisationResult, X::VecNum,
                          ts::Option{<:VecDate} = nothing)
    return PredictionResult(res, X, ts)
end
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)
    return PredictionResult(; res = res, X = calc_net_returns(res, rd.X), ts = rd.ts)
end
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                 test_idx::AbstractVector{<:Integer}, cols = :)
    rdi = returns_result_view(rd, test_idx, cols)
    return PredictionResult(; res = res, X = calc_net_returns(res, rdi.X), ts = rdi.ts)
end
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                 train_idxs::AbstractVector{<:AbstractVector{<:Integer}}, cols = :)
    return [predict(res, rd, test_idx, cols) for test_idx in train_idxs]
end
function fit_and_predict(est::OptimisationEstimator, rd::ReturnsResult; train_idx, test_idx,
                         cols = :)
    rd_train = returns_result_view(rd, train_idx, cols)
    res = optimise(est, rd_train)
    return predict(res, rd, test_idx, cols)
end
function fit_and_predict(est::OptimisationEstimator, rd::ReturnsResult,
                         cv::NonSequentialCrossValidationResult; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    (; train_idx, test_idx) = cv
    predictions = Vector{PredictionResult}(undef, length(train_idx))
    FLoops.@floop ex for (i, (train, test)) in enumerate(zip(train_idx, test_idx))
        predictions[i] = fit_and_predict(est, rd; train_idx = train, test_idx = test,
                                         cols = cols)
    end
    return predictions
end
function sort_predictions(res::CrossValidationResult,
                          predictions::AbstractVector{<:PredictionResult})
    test_idx = res.test_idx
    @argcheck(all(map(x -> allunique(x), test_idx)), "Test indices must be unique.")
    idx = sortperm(test_idx; by = x -> x[1])
    return predictions[idx]
end
function cross_val_predict(est::OptimisationEstimator, rd::ReturnsResult,
                           cv::CrossValidationEstimator = KFold(); cols = :,
                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    if !isa(cols, Colon)
        rd = returns_result_view(rd, cols)
    end
    if hasproperty(cv, :shuffle) && cv.shuffle
        throw(ArgumentError("Cross validation estimator must not be shuffled."))
    end
    res = split(cv, rd)
    @argcheck(all(map(x -> x > zero(x), map(x -> diff(x), res.train_idx))),
              "Cross validation estimator must not be shuffled.")
    predictions = fit_and_predict(est, rd, res; cols = cols, ex = ex)
    return sort_predictions(res, predictions)
end

export PredictionResult, predict, cross_val_predict
