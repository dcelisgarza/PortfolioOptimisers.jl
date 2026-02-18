abstract type CrossValidationEstimator <: AbstractEstimator end
abstract type CrossValidationResult <: AbstractResult end
abstract type CrossValidationAlgorithm <: AbstractAlgorithm end
abstract type SequentialCrossValidationEstimator <: CrossValidationEstimator end
abstract type NonSequentialCrossValidationEstimator <: CrossValidationEstimator end
abstract type SequentialCrossValidationResult <: CrossValidationResult end
abstract type NonSequentialCrossValidationResult <: CrossValidationResult end
struct PredictionResult{T1, T2, T3, T4} <: AbstractResult
    res::T1
    nx::T2
    X::T3
    ts::T4
    function PredictionResult(res::NonFiniteAllocationOptimisationResult,
                              nx::Option{<:VecStr}, X::VecNum, ts::Option{<:VecDate})
        @argcheck(!isempty(X), IsEmptyError)
        if !isnothing(nx)
            @argcheck(!isempty(nx), IsEmptyError)
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
        end
        return new{typeof(res), typeof(nx), typeof(X), typeof(ts)}(res, nx, X, ts)
    end
end
function PredictionResult(; res::NonFiniteAllocationOptimisationResult,
                          nx::Option{<:VecStr} = nothing, X::VecNum,
                          ts::Option{<:VecDate} = nothing)
    return PredictionResult(res, nx, X, ts)
end
struct MultiPeriodPredictionResult{T1} <: AbstractResult
    pred::T1
    function MultiPeriodPredictionResult(pred::AbstractVector{<:PredictionResult})
        return new{typeof(pred)}(pred)
    end
end
function MultiPeriodPredictionResult(;
                                     pred::AbstractVector{<:PredictionResult} = Vector{PredictionResult}(undef,
                                                                                                         0))
    return MultiPeriodPredictionResult(pred)
end
const PredRes_MultiPredRes = Union{<:PredictionResult, <:MultiPeriodPredictionResult}
struct PopulationPredictionResult{T1} <: AbstractResult
    pred::T1
    function PopulationPredictionResult(pred::AbstractVector{<:PredRes_MultiPredRes})
        return new{typeof(pred)}(pred)
    end
end
function PopulationPredictionResult(;
                                    pred::AbstractVector{<:PredRes_MultiPredRes} = Vector{<:PredRes_MultiPredRes}(undef,
                                                                                                                  0))
    return PopulationPredictionResult(pred)
end
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)
    return PredictionResult(; res = res, nx = rd.nx, X = calc_net_returns(res, rd.X),
                            ts = rd.ts)
end
function fit_predict(opt::OptE_Opt, rd::ReturnsResult)
    res = optimise(opt, rd)
    return predict(res, rd)
end
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                 test_idx::VecInt, cols = :)
    rdi = returns_result_view(rd, test_idx, cols)
    return PredictionResult(; res = res, nx = rdi.nx, X = calc_net_returns(res, rdi.X),
                            ts = rdi.ts)
end
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                 train_idxs::VecVecInt, cols = :)
    return [predict(res, rd, test_idx, cols) for test_idx in train_idxs]
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult;
                         train_idx, test_idx, cols = :)
    rd_train = returns_result_view(rd, train_idx, cols)
    if !isa(cols, Colon)
        opt = opt_view(opt, cols, rd.X)
    end
    res = optimise(opt, rd_train)
    return predict(res, rd, test_idx, cols)
end
function sort_predictions!(test_idx::VecVecInt,
                           predictions::AbstractVector{<:PredictionResult})
    @argcheck(all(map(x -> allunique(x), test_idx)), "Test indices must be unique.")
    idx = sortperm(test_idx; by = x -> x[1])
    return predictions[idx]
end
function sort_predictions!(res::CrossValidationResult,
                           predictions::AbstractVector{<:PredictionResult})
    return sort_predictions!(res.test_idx, predictions)
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                         cv::NonSequentialCrossValidationResult; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    (; train_idx, test_idx) = cv
    predictions = Vector{PredictionResult}(undef, length(train_idx))
    # FLoops.@floop ex 
    for (i, (train, test)) in enumerate(zip(train_idx, test_idx))
        predictions[i] = fit_and_predict(opt, rd; train_idx = train, test_idx = test,
                                         cols = cols)
    end
    return MultiPeriodPredictionResult(; pred = predictions)
end

export PredictionResult, MultiPeriodPredictionResult, PopulationPredictionResult, predict,
       fit_predict
