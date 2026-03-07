# randomised search cv can generate the grid after sampling
const SearchCV = Union{<:KFold, <:KFoldResult, <:WalkForwardEstimator, <:WalkForwardResult}
abstract type AbstractSearchCrossValidationEstimator <: AbstractEstimator end
abstract type AbstractSearchCrossValidationResult <: AbstractResult end
abstract type AbstractSearchCrossValidationAlgorithm <: AbstractAlgorithm end
abstract type CrossValidationSearchScorer <: AbstractEstimator end
const CrossValSearchScorer = Union{<:CrossValidationSearchScorer, <:Function}
struct HighestMeanScore <: CrossValidationSearchScorer end
function (s::HighestMeanScore)(X::MatNum)
    return argmax(dropdims(mean(X; dims = 1); dims = 1))
end
struct SearchCrossValidationResult{T1, T2, T3, T4, T5} <:
       AbstractSearchCrossValidationResult
    opti::T1
    test_scores::T2
    train_scores::T3
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
function GridSearchCrossValidation(p::MultiSCVValType_VecMultiSCVValType;
                                   cv::SearchCV = KFold(),
                                   r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                   score::CrossValSearchScorer = HighestMeanScore(),
                                   ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                   train_score::Bool = false, kwargs::NamedTuple = (;))
    return GridSearchCrossValidation(p, cv, r, score, ex, train_score, kwargs)
end
# Base case: bare symbol → PropertyLens
_expr_to_lens(ex::Symbol) = Accessors.PropertyLens(ex)
# Evaluate literal index nodes in the AST (no runtime eval needed)
_eval_index(x::Integer) = x
_eval_index(x::Symbol)  = x
_eval_index(ex::Expr)   = ex.head === :vect ? [_eval_index(a) for a in ex.args] : error("Unsupported index expression: $ex")
function _expr_to_lens_chain(ex)
    optics = Union{Accessors.PropertyLens, Accessors.IndexLens}[]
    while ex isa Expr
        if ex.head === :.
            push!(optics, Accessors.PropertyLens((ex.args[2]::QuoteNode).value))
            ex = ex.args[1]
        elseif ex.head === :ref
            indices = ntuple(i -> _eval_index(ex.args[i + 1]), length(ex.args) - 1)
            push!(optics, Accessors.IndexLens(indices))
            ex = ex.args[1]
        else
            error("Unsupported expression: $ex")
        end
    end
    push!(optics, Accessors.PropertyLens(ex))  # base case: Symbol
    return foldl(∘, optics)
end
function parse_lens(key::AbstractString)
    return _expr_to_lens_chain(Meta.parse(key))
end
function lens_val_grid(estval::AbstractVector{<:Pair{<:String, <:AbstractVector}})
    vals = vec(collect(Iterators.product(map(x -> x[2], estval)...)))
    lenses = fill(parse_lens.(map(x -> x[1], estval)), length(vals))
    return lenses, vals
end
function lens_val_grid(estval::AbstractDict{<:String, <:AbstractVector})
    vals = vec(collect(Iterators.product(values(estval)...)))
    lenses = fill(parse_lens.(keys(estval)), length(vals))
    return lenses, vals
end
function lens_val_grid(estvals::AbstractVector{<:Union{<:AbstractVector{<:Pair{<:String,
                                                                               <:AbstractVector}},
                                                       <:AbstractDict{<:String,
                                                                      <:AbstractVector}}})
    lenses_vals = [lens_val_grid(estval) for estval in estvals]
    lenses = mapreduce(x->x[1], vcat, lenses_vals)
    vals = mapreduce(x -> x[2], vcat, lenses_vals)
    return lenses, vals
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
    lens_grid, val_grid = lens_val_grid(p)
    cv = split(gscv.cv, rd)
    @argcheck(isa(cv.test_idx[1], VecInt))
    N = length(val_grid)
    M = length(cv.train_idx)
    test_scores = Matrix{eltype(rd.X)}(undef, M, N)
    train_scores = if gscv.train_score
        Matrix{eltype(rd.X)}(undef, M, N)
    else
        nothing
    end
    let opt = opt, test_scores = test_scores, train_scores = train_scores
        FLoops.@floop gscv.ex for (i, (lenses, vals)) in enumerate(zip(lens_grid, val_grid))
            local opti = opt
            for (lens, val) in zip(lenses, vals)
                opti = Accessors.set(opti, lens, val)
            end
            for (j, (train_idx, test_idx)) in enumerate(zip(cv.train_idx, cv.test_idx))
                test_score, train_score = fit_and_score(opti, gscv, rd, train_idx, test_idx)
                test_scores[j, i] = test_score
                if gscv.train_score
                    train_scores[j, i] = train_score
                end
            end
        end
    end

    opt_idx = gscv.score(test_scores)
    opt_lens = lens_grid[opt_idx]
    opt_vals = val_grid[opt_idx]
    for (lens, val) in zip(opt_lens, opt_vals)
        opt = Accessors.set(opt, lens, val)
    end
    return SearchCrossValidationResult(opt, test_scores, train_scores, val_grid, opt_idx)
end
export grid_search_cross_validation, GridSearchCrossValidation, SearchCrossValidationResult
