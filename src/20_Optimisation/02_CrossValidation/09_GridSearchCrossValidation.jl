# randomised search cv can generate the grid after sampling
const SearchCV = Union{<:KFold, <:KFoldResult, <:WalkForwardEstimator, <:WalkForwardResult}
abstract type AbstractSearchCrossValidationEstimator <: AbstractEstimator end
abstract type AbstractSearchCrossValidationResult <: AbstractResult end
abstract type AbstractSearchCrossValidationAlgorithm <: AbstractAlgorithm end
abstract type CrossValidationSearchScorer <: AbstractEstimator end
const CrossValSearchScorer = Union{<:CrossValidationSearchScorer, <:Function}
struct HighestMeanScore <: CrossValidationSearchScorer end
function (s::HighestMeanScore)(X::MatNum)
    return argmax(mean(X; dims = 1))
end
struct SearchCrossValidationResult{T1, T2, T3, T4, T5} <:
       AbstractSearchCrossValidationResult
    opti::T1
    split_test_scores::T2
    split_train_scores::T3
    grid::T4
    idx::T5
end
struct GridSearchCrossValidation{T1, T2, T3, T4, T5, T6, T7} <:
       AbstractSearchCrossValidationEstimator
    p::T1
    cv::T2
    r::T3
    score::T4
    ex::T5
    train_score::T6
    kwargs::T7
    function GridSearchCrossValidation(p::MultiSCVValType_VecMultiSCVValType, cv::SearchCV,
                                       r::AbstractBaseRiskMeasure,
                                       score::CrossValSearchScorer,
                                       ex::FLoops.Transducers.Executor, train_score::Bool,
                                       kwargs::NamedTuple)
        @argcheck(!isempty(p), IsEmptyError)
        if isa(p, VecMultiSCVValType)
            @argcheck(all(!isempty, p), IsEmptyError)
        end
        return new{typeof(p), typeof(cv), typeof(r), typeof(score), typeof(ex),
                   typeof(train_score), typeof(kwargs)}(p, cv, r, score, ex, train_score,
                                                        kwargs)
    end
end
function GridSearchCrossValidation(; p::MultiSCVValType_VecMultiSCVValType,
                                   cv::SearchCV = KFold(),
                                   r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                   score::CrossValSearchScorer = HighestMeanScore(),
                                   ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                   train_score::Bool = false, kwargs::NamedTuple = (;))
    return GridSearchCrossValidation(p, cv, r, score, ex, train_score, kwargs)
end
function nested_lens(props::AbstractVector{Symbol})
    lens = Accessors.PropertyLens(props[1])
    for p in view(props, 2:length(props))
        lens = Accessors.PropertyLens(p) ∘ lens
    end
    return lens
end
function build_nested_lens(ks::VecStr)
    ks = split.(ks, ".")
    ks = [Symbol.(k) for k in ks]
    return [nested_lens(k) for k in ks]
end
function key_val_grid(estval::AbstractVector{<:PairSCV})
    return map(x -> x[1], estval), Iterators.product(map(x -> x[2], estval)...)
end
function key_val_grid(estval::DictSCV)
    return keys(estval), Iterators.product(values(estval)...)
end
function key_val_grid(estvals::VecMultiSCVValType)
    ks_vals = [key_val_grid(estval) for estval in estvals]
    ks = reduce(vcat, ks_val[1] for ks_val in ks_vals)
    vals = collect(Iterators.flatten(ks_val[2] for ks_val in ks_vals))
    return ks, vals
end
function fit_and_score(opt::NonFiniteAllocationOptimisationEstimator,
                       gscv::GridSearchCrossValidation, rd::ReturnsResult,
                       train_idx::VecInt, test_idx::VecInt_VecVecInt)
    rd_train = returns_result_view(rd, train_idx, :)
    res = optimise(opt, rd_train)
    test_pred = predict(res, rd, test_idx)
    r = gscv.r
    sign = ifelse(bigger_is_better(r), 1, -1)
    test_score = sign * expected_risk(gscv.r, test_pred; gscv.kwargs...)
    train_score = if gscv.train_score
        sign * expected_risk(gscv.r, res; gscv.kwargs...)
    else
        nothing
    end
    return test_score, train_score
end
function grid_search_cross_validation(opt, gscv::GridSearchCrossValidation,
                                      rd::ReturnsResult; kwargs...)
    p = gscv.p
    ks, val_grid = key_val_grid(p)
    lenses = build_nested_lens(ks)
    cv = split(gscv.cv, rd)
    @argcheck(isa(cv.test_idx[1], VecInt))
    N = length(val_grid)
    M = length(cv.train_idx)
    split_test_scores = Matrix{eltype(rd.X)}(undef, M, N)
    split_train_scores = if gscv.train_score
        Matrix{eltype(rd.X)}(undef, M, N)
    else
        nothing
    end
    let split_test_scores = split_test_scores, split_train_scores = split_train_scores
        FLoops.@floop gscv.ex for (i, v_grid) in enumerate(val_grid)
            local opti = opt
            for (lens, val) in zip(lenses, v_grid)
                opti = Accessors.set(opti, lens, val)
            end
            for (j, (train_idx, test_idx)) in enumerate(zip(cv.train_idx, cv.test_idx))
                test_score, train_score = fit_and_score(opti, gscv, rd, train_idx, test_idx)
                split_test_scores[j, i] = test_score
                if gscv.train_score
                    split_train_scores[j, i] = train_score
                end
            end
        end
    end

    split_test_scores
    split_train_scores
    opt_idx = gscv.score(split_test_scores)[2]
    opt_grid = val_grid[opt_idx]
    opti = opt
    for (lens, val) in zip(lenses, opt_grid)
        opti = Accessors.set(opti, lens, val)
    end
    return SearchCrossValidationResult(opti, split_test_scores, split_train_scores,
                                       val_grid, opt_idx)
end
export grid_search_cross_validation, GridSearchCrossValidation, SearchCrossValidationResult
