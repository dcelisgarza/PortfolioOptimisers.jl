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
struct SearchCrossValidationResult{T1, T2, T3, T4, T5, T6} <:
       AbstractSearchCrossValidationResult
    opt::T1
    test_scores::T2
    train_scores::T3
    lens_grid::T4
    val_grid::T5
    idx::T6
end
function fit_and_score(opt::NonFiniteAllocationOptimisationEstimator,
                       scv::AbstractSearchCrossValidationEstimator, rd::ReturnsResult,
                       train_idx::VecInt, test_idx::VecInt)
    rd_train = returns_result_view(rd, train_idx, :)
    res = optimise(opt, rd_train)
    test_pred = predict(res, rd, test_idx)
    r = scv.r
    sign = ifelse(bigger_is_better(r), 1, -1)
    test_score = sign * expected_risk(scv.r, test_pred; scv.kwargs...)
    train_score = if scv.train_score
        sign * expected_risk(scv.r, res; scv.kwargs...)
    else
        nothing
    end
    return test_score, train_score
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

export SearchCrossValidationResult
