abstract type CrossValidationEstimator <: AbstractEstimator end
abstract type CrossValidationResult <: AbstractResult end
abstract type CrossValidationAlgorithm <: AbstractAlgorithm end

abstract type OptimisationCrossValidationEstimator <: CrossValidationEstimator end
abstract type SequentialCrossValidationEstimator <: OptimisationCrossValidationEstimator end
abstract type NonSequentialCrossValidationEstimator <: OptimisationCrossValidationEstimator end

abstract type OptimisationCrossValidationResult <: CrossValidationResult end
abstract type SequentialCrossValidationResult <: OptimisationCrossValidationResult end
abstract type NonSequentialCrossValidationResult <: OptimisationCrossValidationResult end

abstract type NonOptimisationCrossValidationEstimator <: CrossValidationEstimator end
abstract type NonOptimisationSequentialCrossValidationEstimator <:
              NonOptimisationCrossValidationEstimator end
abstract type NonOptimisationNonSequentialCrossValidationEstimator <:
              NonOptimisationCrossValidationEstimator end

abstract type NonOptimisationCrossValidationResult <: CrossValidationResult end
abstract type NonOptimisationSequentialCrossValidationResult <:
              NonOptimisationCrossValidationResult end
abstract type NonOptimisationNonSequentialCrossValidationResult <:
              NonOptimisationCrossValidationResult end

struct PredictionReturnsResult{T1, T2, T3, T4, T5, T6, T7} <: AbstractReturnsResult
    nx::T1
    X::T2
    nf::T3
    F::T4
    ts::T5
    iv::T6
    ivpa::T7
    function PredictionReturnsResult(nx::Option{<:VecStr}, X::Option{<:VecNum_VecVecNum},
                                     nf::Option{<:VecStr}, F::Option{<:MatNum},
                                     ts::Option{<:VecDate}, iv::Option{<:VecNum_VecVecNum},
                                     ivpa::Option{<:Num_VecNum})
        _check_names_and_returns_matrix(nf, F, :nf, :F)
        if !isnothing(X) && !isnothing(F)
            if isa(X, VecNum)
                @argcheck(length(X) == size(F, 1), DimensionMismatch)
            else
                @argcheck(all(x -> length(x) == size(F, 1), X))
            end
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
            @argcheck(!(isnothing(X) && isnothing(F)), IsNothingError)
            if isa(X, VecBaseRM)
                @argcheck(length(ts) == length(X), DimensionMismatch)
            elseif isa(X, VecVecNum)
                @argcheck(all(x -> length(x) == length(ts), X))
            end
            if !isnothing(F)
                @argcheck(length(ts) == size(F, 1), DimensionMismatch)
            end
        end
        if isa(iv, VecNum)
            @argcheck(isa(ivpa, Option{<:Number}))
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(length(iv) == length(X), DimensionMismatch)
        elseif isa(iv, VecVecNum)
            @argcheck(isa(ivpa, Option{<:VecNum}))
            @argcheck(length(iv) == length(X), DimensionMismatch)
            @argcheck(length(ivpa) == length(X), DimensionMismatch)
            for (ivi, ivpai, Xi) in zip(iv, ivpa, X)
                assert_nonempty_nonneg_finite_val(ivi, :iv)
                assert_nonempty_gt0_finite_val(ivpai, :ivpa)
                @argcheck(length(ivi) == length(Xi), DimensionMismatch)
            end
        end
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(ts), typeof(iv),
                   typeof(ivpa)}(nx, X, nf, F, ts, iv, ivpa)
    end
end
function PredictionReturnsResult(; nx::Option{<:VecStr} = nothing,
                                 X::Option{<:VecNum_VecVecNum} = nothing,
                                 nf::Option{<:VecStr} = nothing,
                                 F::Option{<:MatNum} = nothing,
                                 ts::Option{<:VecDate} = nothing,
                                 iv::Option{<:VecNum_VecVecNum} = nothing,
                                 ivpa::Option{<:Num_VecNum} = nothing)
    return PredictionReturnsResult(nx, X, nf, F, ts, iv, ivpa)
end
abstract type AbstractPredictionResult <: AbstractResult end
struct PredictionResult{T1, T2} <: AbstractPredictionResult
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
struct SingletonVector{T} <: AbstractVector{T} end
function SingletonVector()
    return SingletonVector{Int}()
end
Base.length(::SingletonVector) = 1
Base.getindex(::SingletonVector, args...) = 1
Base.:*(M::Matrix, ::SingletonVector) = dropdims(M; dims = 2)
Base.size(::SingletonVector) = (1,)
function expected_risk(pred::PredictionResult{<:Any,
                                              <:PredictionReturnsResult{<:Any, <:VecNum}},
                       r::AbstractBaseRiskMeasure; kwargs...)
    return expected_risk(r, SingletonVector{Int}(), reshape(pred.rd.X, :, 1); kwargs...)
end
function expected_risk(pred::PredictionResult{<:Any,
                                              <:PredictionReturnsResult{<:Any, <:VecVecNum}},
                       r::AbstractBaseRiskMeasure; kwargs...)
    X = pred.rd.X
    return [expected_risk(r, SingletonVector(), reshape(Xi, :, 1); kwargs...) for Xi in X]
end
struct MultiPeriodPredictionResult{T1} <: AbstractPredictionResult
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
function mapreduce_X(rd::AbstractVector{<:PredictionReturnsResult{<:Any, <:VecNum}})
    return mapreduce(x -> getproperty(x, :X), vcat, rd)
end
function mapreduce_X(rd::AbstractVector{<:PredictionReturnsResult{<:Any, <:VecVecNum}})
    N = length(rd[1].X)
    X = [eltype(rd[1].X[1])[] for _ in 1:N]
    for i in 1:N
        X[i] = mapreduce(x -> getproperty(x, :X)[i], vcat, rd)
    end
    return X
end
function get_multiperiod_returns_result(mpred::MultiPeriodPredictionResult)
    rd = mpred.rd
    nx = rd[1].nx
    X = mapreduce_X(rd)
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
function _prediction_expected_risk(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...)
    return expected_risk(r, SingletonVector{Int}(), reshape(X, :, 1); kwargs...)
end
function _prediction_expected_risk(r::AbstractBaseRiskMeasure, X::VecVecNum; kwargs...)
    return [expected_risk(r, SingletonVector{Int}(), reshape(Xi, :, 1); kwargs...)
            for Xi in X]
end
function expected_risk(mpred::MultiPeriodPredictionResult, r::AbstractBaseRiskMeasure;
                       kwargs...)
    X = mpred.mrd.X
    return _prediction_expected_risk(r, X; kwargs...)
end
const PredRes_MultiPredRes = Union{<:PredictionResult, <:MultiPeriodPredictionResult}
struct PopulationPredictionResult{T1} <: AbstractPredictionResult
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
function expected_risk(ppred::PopulationPredictionResult, r::AbstractBaseRiskMeasure;
                       kwargs...)
    return [expected_risk(pred, r; kwargs...) for pred in ppred.pred]
end
function sort_by_measure(ppred::PopulationPredictionResult, r::AbstractBaseRiskMeasure;
                         kwargs...)
    pred = filter(x -> all(y -> isa(y.res.retcode, OptimisationSuccess), x.pred),
                  ppred.pred)
    return sort(pred; by = x -> expected_risk(x, r; kwargs...), rev = bigger_is_better(r))
end
function quantile_by_measure(ppred::PopulationPredictionResult, r::AbstractBaseRiskMeasure,
                             q::Real; kwargs...)
    sorted_predictions = sort_by_measure(ppred, r; kwargs...)
    idx = max(1, round(Int, Statistics.quantile(1:length(sorted_predictions), q)))
    return sorted_predictions[idx]
end
#! Start: Use these for scoring grid/random search cv
function _map_to_population_measures(::Val{true}, rks::VecNum, f)
    return f(rks)
end
function _map_to_population_measures(::Val{false}, rks::VecNum, f)
    return rks'
end
function _map_to_population_measures(::Val{false}, rks::VecVecNum, f)
    return f(reduce(vcat, rks); dims = 1)
end
function map_to_population_measures(ppred::PopulationPredictionResult,
                                    r::AbstractBaseRiskMeasure, f = mean)
    rks = expected_risk(ppred, r)
    return _map_to_population_measures(Val(isa(ppred.pred[1].rd[1].X, VecNum)), rks, f)
end
#! End: Use these for scoring grid/random search cv
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)
    return PredictionResult(; res = res, rd = rd)
end
function fit_predict(opt::OptE_Opt, rd::ReturnsResult)
    res = optimise(opt, rd)
    return predict(res, rd)
end
function reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                        X::VecNum)
    iv = isnothing(rd.iv) ? rd.iv : rd.iv * res.w
    ivpa = !isa(rd.ivpa, AbstractVector) ? rd.ivpa : dot(rd.ivpa, res.w)
    return PredictionReturnsResult(; nx = rd.nx, X = X, nf = rd.nf, F = rd.F, ts = rd.ts,
                                   iv = iv, ivpa = ivpa)
end
function reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                        X::VecVecNum)
    iv = isnothing(rd.iv) ? rd.iv : [rd.iv * w for w in res.w]
    ivpa = !isa(rd.ivpa, AbstractVector) ? rd.ivpa : [dot(rd.ivpa, w) for w in res.w]
    return PredictionReturnsResult(; nx = rd.nx, X = X, nf = rd.nf, F = rd.F, ts = rd.ts,
                                   iv = iv, ivpa = ivpa)
end
function predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                 test_idx::VecInt, cols = :)
    rdi = returns_result_view(rd, test_idx, cols)
    X = calc_net_returns(res, rdi.X)
    rdi = reconstruct_rd(res, rdi, X)
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
       fit_predict, sort_by_measure
