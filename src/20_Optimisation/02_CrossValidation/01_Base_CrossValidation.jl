abstract type CrossValidationEstimator <: AbstractEstimator end
abstract type CrossValidationResult <: AbstractResult end
abstract type CrossValidationAlgorithm <: AbstractAlgorithm end
abstract type OptimisationCrossValidationEstimator <: CrossValidationEstimator end
abstract type SequentialCrossValidationEstimator <: OptimisationCrossValidationEstimator end
abstract type NonSequentialCrossValidationEstimator <: OptimisationCrossValidationEstimator end
abstract type SequentialCrossValidationResult <: CrossValidationResult end
abstract type NonSequentialCrossValidationResult <: CrossValidationResult end
abstract type SequentialNonOptimisationCrossValidationResult <: CrossValidationResult end
struct PredictionReturnsResult{T1, T2, T3, T4, T5, T6, T7} <: AbstractReturnsResult
    nx::T1
    X::T2
    nf::T3
    F::T4
    ts::T5
    iv::T6
    ivpa::T7
    function PredictionReturnsResult(nx::Option{<:VecStr}, X::Option{<:VecNum},
                                     nf::Option{<:VecStr}, F::Option{<:MatNum},
                                     ts::Option{<:VecDate}, iv::Option{<:VecNum},
                                     ivpa::Option{<:Number})
        _check_names_and_returns_matrix(nf, F, :nf, :F)
        if !isnothing(X) && !isnothing(F)
            @argcheck(length(X) == size(F, 1), DimensionMismatch)
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
            @argcheck(!(isnothing(X) && isnothing(F)), IsNothingError)
            if !isnothing(X)
                @argcheck(length(ts) == length(X), DimensionMismatch)
            end
            if !isnothing(F)
                @argcheck(length(ts) == size(F, 1), DimensionMismatch)
            end
        end
        if !isnothing(iv)
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(length(iv) == length(X), DimensionMismatch)
        end
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(ts), typeof(iv),
                   typeof(ivpa)}(nx, X, nf, F, ts, iv, ivpa)
    end
end
function PredictionReturnsResult(; nx::Option{<:VecStr} = nothing,
                                 X::Option{<:VecNum} = nothing,
                                 nf::Option{<:VecStr} = nothing,
                                 F::Option{<:MatNum} = nothing,
                                 ts::Option{<:VecDate} = nothing,
                                 iv::Option{<:VecNum} = nothing,
                                 ivpa::Option{<:Number} = nothing)
    return PredictionReturnsResult(nx, X, nf, F, ts, iv, ivpa)
end
struct PredictionResult{T1, T2} <: AbstractResult
    res::T1
    rd::T2
    function PredictionResult(res::NonFiniteAllocationOptimisationResult,
                              rd::PredictionReturnsResult)
        return new{typeof(res), typeof(rd)}(res, rd)
    end
end
function PredictionResult(; res::NonFiniteAllocationOptimisationResult,
                          rd::PredictionReturnsResult)
    return PredictionResult(res, rd)
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
function get_multiperiod_returns_result(mpred::MultiPeriodPredictionResult)
    rd = mpred.rd
    nx = rd[1].nx
    X = mapreduce(x -> getproperty(x, :X), vcat, rd)
    nf = rd[1].nf
    F = isnothing(rd[1].F) ? nothing : mapreduce(x -> getproperty(x, :F), vcat, rd)
    ts = isnothing(rd[1].ts) ? nothing : mapreduce(x -> getproperty(x, :ts), vcat, rd)
    iv = isnothing(rd[1].iv) ? nothing : mapreduce(x -> getproperty(x, :iv), vcat, rd)
    ivpa = rd[1].ivpa
    return PredictionReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts, iv = iv,
                                   ivpa = ivpa)
end
function Base.getproperty(mpred::MultiPeriodPredictionResult, sym::Symbol)
    return if sym == :res
        getfield.(getfield(mpred, :pred), :res)
    elseif sym === :rd
        getfield.(getfield(mpred, :pred), :rd)
    elseif sym == :mrd
        get_multiperiod_returns_result(mpred)
    else
        getfield(mpred, sym)
    end
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
    return PredictionResult(; res = res, rd = rd)
end
function fit_predict(opt::OptE_Opt, rd::ReturnsResult)
    res = optimise(opt, rd)
    return predict(res, rd)
end
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                 test_idx::VecInt, cols = :)
    rdi = returns_result_view(rd, test_idx, cols)
    iv = isnothing(rdi.iv) ? rdi.iv : rdi.iv * res.w
    ivpa = !isa(rdi.ivpa, AbstractVector) ? rdi.ivpa : dot(rdi.ivpa, res.w)
    X = calc_net_returns(res, rdi.X)
    rdi = PredictionReturnsResult(; nx = rdi.nx, X = X, nf = rdi.nf, F = rdi.F, ts = rdi.ts,
                                  iv = iv, ivpa = ivpa)
    return PredictionResult(; res = res, rd = rdi)
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
    FLoops.@floop ex for (i, (train, test)) in enumerate(zip(train_idx, test_idx))
        predictions[i] = fit_and_predict(opt, rd; train_idx = train, test_idx = test,
                                         cols = cols)
    end
    return MultiPeriodPredictionResult(; pred = predictions)
end

export PredictionResult, MultiPeriodPredictionResult, PopulationPredictionResult, predict,
       fit_predict
