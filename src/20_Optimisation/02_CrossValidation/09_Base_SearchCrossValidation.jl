"""
    const SearchCV = Union{<:KFold, <:KFoldResult, <:WalkForwardEstimator, <:WalkForwardResult}

Cross-validations compatible with search-based hyperparameter tuning.
"""
const SearchCV = Union{<:KFold, <:KFoldResult, <:WalkForwardEstimator, <:WalkForwardResult}
abstract type AbstractSearchCrossValidationEstimator <: AbstractEstimator end
"""
    abstract type AbstractSearchCrossValidationResult <: AbstractResult

Abstract result type for search-based cross-validation routines. Serves as the parent for all result types produced by search cross-validation algorithms, encapsulating optimal estimator, score matrices, parameter grid, and selected index.

# Interfaces

  - Subtypes must store the optimal estimator, test and train scores, parameter grid, and selected index.
"""
abstract type AbstractSearchCrossValidationResult <: AbstractResult end
abstract type AbstractSearchCrossValidationAlgorithm <: AbstractAlgorithm end
"""
    abstract type CrossValidationSearchScorer <: AbstractEstimator

Defines the interface for scoring strategies used in search cross-validation. Implementations select the optimal parameter set based on cross-validation scores.

# Interfaces

  - `(::CrossValidationSearchScorer)(scores::AbstractMatrix)`: Returns the index of the optimal parameter set.

## Arguments

  - `scores`: Matrix of scores, where each column corresponds to a parameter set and each row to a cross-validation split.

## Returns

  - `Int`: Index of the optimal parameter set.

# Examples

```jldoctest
julia> struct MyScore <: PortfolioOptimisers.CrossValidationSearchScorer end

julia> (s::MyScore)(X::Matrix{Float64}) = argmin(dropdims(mean(X; dims = 1); dims = 1))

julia> scores = [0.5 0.6; 0.7 0.8];

julia> scorer = MyScore()
MyScore()

julia> scorer(scores)
1
```

# Related Types

  - [`CrossValSearchScorer`](@ref)
"""
abstract type CrossValidationSearchScorer <: AbstractEstimator end
"""
    const CrossValSearchScorer = Union{<:CrossValidationSearchScorer, <:Function}

Union type for search cross-validation scoring strategies. Accepts either a subtype of `CrossValidationSearchScorer` or a plain function that accepts a matrix and returns an integer.
"""
const CrossValSearchScorer = Union{<:CrossValidationSearchScorer, <:Function}
struct HighestMeanScore <: CrossValidationSearchScorer end
function (s::HighestMeanScore)(X::MatNum; dims::Integer = 1)
    return argmax(dropdims(mean(X; dims = dims); dims = dims))
end
"""
    struct SearchCrossValidationResult{T1, T2, T3, T4, T5, T6} <:
           AbstractSearchCrossValidationResult
        opt::T1
        test_scores::T2
        train_scores::T3
        lens_grid::T4
        val_grid::T5
        idx::T6
    end

Result type for search-based cross-validation routines. Stores the optimal estimator, score matrices, parameter grid, and selected index for hyperparameter search.

# Fields

  - `opt`: Optimal estimator instance after search.
  - `test_scores`: Matrix of test scores for each parameter set and CV split.
  - `train_scores`: Matrix of training scores for each parameter set and CV split (or `nothing` if not recorded).
  - `lens_grid`: Vector of lens objects for parameter access.
  - `val_grid`: Vector of parameter values for each configuration.
  - `idx`: Index of the selected optimal parameter set.

# Examples

# Related

  - [`AbstractSearchCrossValidationResult`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`fit_and_score`](@ref)
"""
struct SearchCrossValidationResult{T1, T2, T3, T4, T5, T6} <:
       AbstractSearchCrossValidationResult
    opt::T1
    test_scores::T2
    train_scores::T3
    lens_grid::T4
    val_grid::T5
    idx::T6
end
"""
    fit_and_score(opt::NonFiniteAllocationOptimisationEstimator,
                  scv::AbstractSearchCrossValidationEstimator,
                  rd::ReturnsResult,
                  train_idx::VecInt,
                  test_idx::VecInt)

Fits a portfolio optimisation estimator on training data, scores it on test and train splits, and returns the scores for search-based cross-validation routines.

# Arguments

  - `opt`: Portfolio optimisation estimator to fit.
  - `scv`: Search cross-validation estimator specifying risk measure and options.
  - `rd`: Returns result containing asset returns data.
  - `train_idx`: Indices for training split.
  - `test_idx`: Indices for test split.

# Returns

  - `test_score::Number`: Test score.
  - `train_score::Option{<:Number}`: Train score.

# Details

  - Fits the estimator on training data.
  - Predicts on test data using the fitted estimator.
  - Computes risk scores for test and train splits.
  - Applies sign convention based on risk measure direction.
  - Returns scores for use in search cross-validation.

# Related

  - [`SearchCrossValidationResult`](@ref)
  - [`expected_risk`](@ref)
  - [`predict`]-(@ref)
  - [`NonFiniteAllocationOptimisationEstimator`]-(@ref)

# Examples
"""
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

export SearchCrossValidationResult, HighestMeanScore
