"""
    const SearchCV = Union{<:KFold, <:KFoldResult, <:WalkForwardEstimator, <:WalkForwardResult}

Cross-validations compatible with search-based hyperparameter tuning.
"""
const SearchCV = Union{<:KFold, <:KFoldResult, <:WalkForwardEstimator, <:WalkForwardResult}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all search-based cross-validation estimators in `PortfolioOptimisers.jl`.

Subtypes implement hyperparameter search strategies (e.g. grid search, randomised search) that use cross-validation to select the best estimator configuration.

# Related

  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`AbstractSearchCrossValidationResult`](@ref)
  - [`SearchCV`](@ref)
"""
abstract type AbstractSearchCrossValidationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract result type for search-based cross-validation routines. Serves as the parent for all result types produced by search cross-validation algorithms, encapsulating optimal estimator, score matrices, parameter grid, and selected index.

# Interfaces

  - Subtypes must store the optimal estimator, test and train scores, parameter grid, and selected index.
"""
abstract type AbstractSearchCrossValidationResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all search-based cross-validation algorithm types.

Subtypes define the strategy used to select the best hyperparameter combination from the search results (e.g. selecting by highest mean score).

# Related

  - [`AbstractSearchCrossValidationEstimator`](@ref)
  - [`SearchCrossValidationResult`](@ref)
"""
abstract type AbstractSearchCrossValidationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

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
"""
$(DocStringExtensions.TYPEDEF)

A [`CrossValidationSearchScorer`](@ref) that selects the parameter set with the highest mean score across cross-validation splits.

When called with a score matrix (rows = CV splits, columns = parameter sets), it returns the column index with the largest mean score.

# Examples

```jldoctest
julia> scorer = PortfolioOptimisers.HighestMeanScore();

julia> scores = [0.5 0.8; 0.6 0.7];

julia> scorer(scores)
2
```

# Related

  - [`CrossValidationSearchScorer`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`SearchCrossValidationResult`](@ref)
"""
struct HighestMeanScore <: CrossValidationSearchScorer end
function (s::HighestMeanScore)(X::MatNum; dims::Integer = 1)
    return argmax(dropdims(mean(X; dims = dims); dims = dims))
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for search-based cross-validation routines. Stores the optimal estimator, score matrices, parameter grid, and selected index for hyperparameter search.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`AbstractSearchCrossValidationResult`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`fit_and_score`](@ref)
"""
@concrete struct SearchCrossValidationResult <: AbstractSearchCrossValidationResult
    "$(field_dict[:opt_cv])"
    opt
    "$(field_dict[:test_scores])"
    test_scores
    "$(field_dict[:train_scores])"
    train_scores
    "$(field_dict[:lens_grid])"
    lens_grid
    "$(field_dict[:val_grid])"
    val_grid
    "$(field_dict[:idx_cv])"
    idx
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
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
"""
function fit_and_score(opt::NonFiniteAllocationOptimisationEstimator,
                       scv::AbstractSearchCrossValidationEstimator, rd::ReturnsResult,
                       train_idx::VecInt, test_idx::VecInt)
    rd_train = returns_result_view(rd, train_idx, :)
    res = optimise(opt, rd_train)
    test_pred = StatsAPI.predict(res, rd, test_idx)
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
"""
    _expr_to_lens(ex::Symbol)

Convert a bare symbol into a `PropertyLens` for field access.

Base case for the lens-building recursion: a bare symbol maps directly to an `Accessors.PropertyLens`.

# Arguments

  - `ex::Symbol`: A field name symbol.

# Returns

  - `Accessors.PropertyLens` for the symbol.

# Related

  - [`_expr_to_lens_chain`](@ref)
  - [`parse_lens`](@ref)
"""
_expr_to_lens(ex::Symbol) = Accessors.PropertyLens(ex)
"""
    _eval_index(x)

Evaluate a literal index node in the AST without runtime `eval`.

Converts integer, symbol, or vector expression AST nodes to concrete index values for use in `Accessors.IndexLens`.

# Arguments

  - `x::Integer`: An integer index.
  - `x::Symbol`: A symbolic index.
  - `ex::Expr`: A vector expression (`:vect` head).

# Returns

  - The evaluated index value.

# Related

  - [`_expr_to_lens_chain`](@ref)
"""
_eval_index(x::Integer) = x
_eval_index(x::Symbol) = x
_eval_index(ex::Expr)  = ex.head === :vect ? [_eval_index(a) for a in ex.args] : error("Unsupported index expression: $ex")
"""
    _expr_to_lens_chain(ex)

Convert a Julia expression to a chain of lens accessors.

Internal helper for parsing hyperparameter key strings into composable Accessors.jl lenses.

# Arguments

  - `ex`: Julia expression representing a field access chain.

# Returns

  - Composed lens.

# Related

  - [`parse_lens`](@ref)
  - [`_expr_to_lens`](@ref)
"""
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
"""
    parse_lens(key::AbstractString)

Parse a hyperparameter key string into an Accessors.jl lens.

Converts a dotted string path (e.g., `"opt.pe.ce"`) into a composable lens for getting and setting nested fields of an estimator object.

# Arguments

  - `key`: Dotted field path string.

# Returns

  - Composed Accessors.jl lens.

# Related

  - [`_expr_to_lens_chain`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
"""
function parse_lens(key::AbstractString)
    return _expr_to_lens_chain(Meta.parse(key))
end
function parse_lens(key::Union{Expr, Symbol})
    return _expr_to_lens_chain(key)
end
function parse_lens(key::Union{<:ComposedFunction, <:Accessors.PropertyLens,
                               <:Accessors.IndexLens})
    return key
end

export SearchCrossValidationResult, HighestMeanScore
