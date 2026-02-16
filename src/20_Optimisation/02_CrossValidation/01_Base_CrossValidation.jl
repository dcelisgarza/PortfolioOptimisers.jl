abstract type CrossValidationEstimator <: AbstractEstimator end
abstract type CrossValidationResult <: AbstractResult end
abstract type CrossValidationAlgorithm <: AbstractAlgorithm end
abstract type SequentialCrossValidationEstimator <: CrossValidationEstimator end
abstract type NonSequentialCrossValidationEstimator <: CrossValidationEstimator end
abstract type SequentialCrossValidationResult <: CrossValidationResult end
abstract type NonSequentialCrossValidationResult <: CrossValidationResult end

function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                 test_idx::VecInt, cols = :)
    rdi = returns_result_view(rd, test_idx, cols)
    return PredictionResult(; res = res, X = calc_net_returns(res, rdi.X), ts = rdi.ts)
end
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                 train_idxs::VecVecInt, cols = :)
    return [predict(res, rd, test_idx, cols) for test_idx in train_idxs]
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult;
                         train_idx, test_idx, cols = :)
    rd_train = returns_result_view(rd, train_idx, cols)
    res = optimise(opt, rd_train)
    return predict(res, rd, test_idx, cols)
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                         cv::NonSequentialCrossValidationResult; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    (; train_idx, test_idx) = cv
    predictions = Vector{PredictionResult}(undef, length(train_idx))
    FLoops.@floop ex for (i, (train, test)) in enumerate(zip(train_idx, test_idx))
        predictions[i] = fit_and_predict(opt, rd; train_idx = train, test_idx = test,
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