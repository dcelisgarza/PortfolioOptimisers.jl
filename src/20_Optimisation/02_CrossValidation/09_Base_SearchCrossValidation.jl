"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all search-based cross-validation estimators.

Subtypes implement hyperparameter search strategies (e.g. grid search, randomised search) that use cross-validation to select the best estimator configuration.

# Related

  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`AbstractSearchCrossValidationResult`](@ref)
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

  - `(::CrossValidationSearchScorer)(scores::AbstractMatrix)`: Returns the index of the optimal parameter set — the column with the best (highest) aggregate score.

## Arguments

  - `scores`: Matrix of scores, where each column corresponds to a parameter set and each row to a cross-validation split. The matrix is orientation-normalised: risk-measure scores are negated so that **higher is always better**, whatever measure `r` is. A scorer therefore returns the index of the column with the best (highest) aggregate score.

## Returns

  - `Int`: Index of the optimal parameter set.

# Examples

```jldoctest
julia> struct MyScore <: PortfolioOptimisers.CrossValidationSearchScorer end

julia> (s::MyScore)(X::Matrix{Float64}) = argmax(dropdims(mean(X; dims = 1); dims = 1))

julia> scores = [0.5 0.6; 0.7 0.8];

julia> scorer = MyScore()
MyScore()

julia> scorer(scores)
2
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

Performs grid search cross-validation for portfolio optimisation estimators. Iterates over parameter grids, applies cross-validation splits, and scores each configuration to select the optimal parameters.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GridSearchCrossValidation(
        p::MultiGSCVValType_VecMultiGSCVValType;
        cv::CrossValidationEstimator = KFold(),
        r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
        scorer::CrossValSearchScorer = HighestMeanScore(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        train_score::Bool = false,
        kwargs::NamedTuple = (;),
    ) -> GridSearchCrossValidation

Positional and keyword arguments correspond to fields above.

## Validation

  - `!isempty(p)`.
  - If `p` is a vector of parameter sets: each element must not be empty.
  - All keys in `p` must be of type `GSCVKey` (i.e. `String`, `Symbol`, or `Integer`).

# Examples

```jldoctest
julia> GridSearchCrossValidation(Dict("alpha" => [0.1, 0.2], "beta" => [1.0, 2.0]))
GridSearchCrossValidation
            p ┼ Dict{String, Vector{Float64}}: Dict("alpha" => [0.1, 0.2], "beta" => [1.0, 2.0])
           cv ┼ KFold
              │              n ┼ Int64: 5
              │    purged_size ┼ Int64: 0
              │   embargo_size ┴ Int64: 0
            r ┼ ConditionalValueatRisk
              │   settings ┼ RiskMeasureSettings
              │            │   scale ┼ Float64: 1.0
              │            │      ub ┼ nothing
              │            │     rke ┴ Bool: true
              │      alpha ┼ Float64: 0.05
              │          w ┴ nothing
       scorer ┼ HighestMeanScore()
           ex ┼ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
  train_score ┼ Bool: false
       kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`MultiGSCVValType_VecMultiGSCVValType`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`search_cross_validation`](@ref)
"""
@concrete struct GridSearchCrossValidation <: AbstractSearchCrossValidationEstimator
    """
    $(field_dict[:p_cv])
    """
    p
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:r])
    """
    r
    """
    $(field_dict[:scorer])
    """
    scorer
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:train_score])
    """
    train_score
    """
    $(field_dict[:kwargs])
    """
    kwargs
    function GridSearchCrossValidation(p::Union{<:AbstractVector{<:Pair{<:Any,
                                                                        <:AbstractVector}},
                                                <:AbstractVector{<:AbstractVector{<:Pair{<:Any,
                                                                                         <:AbstractVector}}},
                                                <:AbstractDict{<:Any, <:AbstractVector},
                                                <:AbstractVector{<:AbstractDict{<:Any,
                                                                                <:AbstractVector}}},
                                       cv::CrossValidationEstimator,
                                       r::AbstractBaseRiskMeasure,
                                       scorer::CrossValSearchScorer,
                                       ex::FLoops.Transducers.Executor, train_score::Bool,
                                       kwargs::NamedTuple)
        @argcheck(!isempty(p), IsEmptyError)
        p_flag = isa(p, AbstractVector{<:Pair})
        d_flag = isa(p, AbstractDict)
        vp_flag = isa(p, AbstractVector{<:AbstractVector{<:Pair}})
        vd_flag = isa(p, AbstractVector{<:AbstractDict})
        if p_flag
            @argcheck(all(x -> isa(x[1], GSCVKey), p),
                      ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
        elseif d_flag
            @argcheck(all(x -> isa(x, GSCVKey), keys(p)),
                      ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
        elseif vp_flag || vd_flag
            @argcheck(all(!isempty, p),
                      IsEmptyError("each parameter set in p cannot be empty"))
            if vp_flag
                for _p in p
                    @argcheck(all(x -> isa(x[1], GSCVKey), _p),
                              ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
                end
            end
            if vd_flag
                for _p in p
                    @argcheck(all(x -> isa(x, GSCVKey), keys(_p)),
                              ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
                end
            end
        end
        return new{typeof(p), typeof(cv), typeof(r), typeof(scorer), typeof(ex),
                   typeof(train_score), typeof(kwargs)}(p, cv, r, scorer, ex, train_score,
                                                        kwargs)
    end
end
function GridSearchCrossValidation(p::Union{<:AbstractVector{<:Pair{<:Any,
                                                                    <:AbstractVector}},
                                            <:AbstractVector{<:AbstractVector{<:Pair{<:Any,
                                                                                     <:AbstractVector}}},
                                            <:AbstractDict{<:Any, <:AbstractVector},
                                            <:AbstractVector{<:AbstractDict{<:Any,
                                                                            <:AbstractVector}}};
                                   cv::CrossValidationEstimator = KFold(),
                                   r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                   scorer::CrossValSearchScorer = HighestMeanScore(),
                                   ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                   train_score::Bool = false, kwargs::NamedTuple = (;))
    return GridSearchCrossValidation(p, cv, r, scorer, ex, train_score, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Randomised search cross-validation estimator for portfolio optimisation. Samples parameter sets from distributions or vectors, applies cross-validation splits, fits and scores each configuration, and selects the optimal parameters using the provided scoring strategy.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RandomisedSearchCrossValidation(
        p::Union{AbstractVector{<:Pair{<:GSCVKey, <:RSCVVal}},
                 AbstractVector{<:AbstractVector{<:Pair{<:GSCVKey,
                                                        <:RSCVVal}}},
                 AbstractDict{<:GSCVKey, <:RSCVVal},
                 AbstractVector{<:AbstractDict{<:GSCVKey,
                                               <:RSCVVal}}};
        cv::CrossValidationEstimator = KFold(),
        r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
        scorer::CrossValSearchScorer = HighestMeanScore(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        n_iter::Integer = 10,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        train_score::Bool = false,
        kwargs::NamedTuple = (;),
    ) -> RandomisedSearchCrossValidation

Keyword arguments correspond to the struct's fields.

## Validation

  - `!isempty(p)`.
  - If `p` is a vector of parameter sets: each element must not be empty.
  - All keys in `p` must be of type `GSCVKey` (i.e. `String`, `Symbol`, or `Integer`).
  - All values in `p` must be of type `RSCVVal` (i.e. an `AbstractVector` or `Distributions.Distribution`).
  - `n_iter > 0` and finite.

# Examples

```jldoctest
julia> RandomisedSearchCrossValidation(Dict("alpha" => [0.1, 0.2, 0.3],
                                            "beta" => Normal(1.0, 0.5)))
RandomisedSearchCrossValidation
            p ┼ Dict{String, Any}: Dict{String, Any}("alpha" => [0.1, 0.2, 0.3], "beta" => Distributions.Normal{Float64}(μ=1.0, σ=0.5))
           cv ┼ KFold
              │              n ┼ Int64: 5
              │    purged_size ┼ Int64: 0
              │   embargo_size ┴ Int64: 0
            r ┼ ConditionalValueatRisk
              │   settings ┼ RiskMeasureSettings
              │            │   scale ┼ Float64: 1.0
              │            │      ub ┼ nothing
              │            │     rke ┴ Bool: true
              │      alpha ┼ Float64: 0.05
              │          w ┴ nothing
       scorer ┼ HighestMeanScore()
           ex ┼ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
       n_iter ┼ Int64: 10
          rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
         seed ┼ nothing
  train_score ┼ Bool: false
       kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractSearchCrossValidationEstimator`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`SearchCrossValidationResult`](@ref)
  - [`CrossValSearchScorer`](@ref)
"""
@concrete struct RandomisedSearchCrossValidation <: AbstractSearchCrossValidationEstimator
    """
    $(field_dict[:p_cv])
    """
    p
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:r])
    """
    r
    """
    $(field_dict[:scorer])
    """
    scorer
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:n_iter])
    """
    n_iter
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
    """
    $(field_dict[:train_score])
    """
    train_score
    """
    $(field_dict[:kwargs])
    """
    kwargs
    function RandomisedSearchCrossValidation(p::Union{<:AbstractVector{<:Pair},
                                                      <:AbstractVector{<:AbstractVector{<:Pair}},
                                                      <:AbstractDict,
                                                      <:AbstractVector{<:AbstractDict}},
                                             cv::CrossValidationEstimator,
                                             r::AbstractBaseRiskMeasure,
                                             scorer::CrossValSearchScorer,
                                             ex::FLoops.Transducers.Executor,
                                             n_iter::Integer, rng::Random.AbstractRNG,
                                             seed::Option{<:Integer}, train_score::Bool,
                                             kwargs::NamedTuple)
        @argcheck(!isempty(p), IsEmptyError)
        p_flag = isa(p, AbstractVector{<:Pair})
        d_flag = isa(p, AbstractDict)
        vp_flag = isa(p, AbstractVector{<:AbstractVector{<:Pair}})
        vd_flag = isa(p, AbstractVector{<:AbstractDict})
        if p_flag
            @argcheck(all(x -> isa(x[1], GSCVKey), p),
                      ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
            @argcheck(all(x -> isa(x[2], RSCVVal), p),
                      ArgumentError("all values in p must be of type RSCVVal (AbstractVector or Distribution)"))
        elseif d_flag
            @argcheck(all(x -> isa(x, GSCVKey), keys(p)),
                      ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
            @argcheck(all(x -> isa(x, RSCVVal), values(p)),
                      ArgumentError("all values in p must be of type RSCVVal (AbstractVector or Distribution)"))
        elseif vp_flag || vd_flag
            @argcheck(all(!isempty, p),
                      IsEmptyError("each parameter set in p cannot be empty"))
            if vp_flag
                for _p in p
                    @argcheck(all(x -> isa(x[1], GSCVKey), _p),
                              ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
                    @argcheck(all(x -> isa(x[2], RSCVVal), _p),
                              ArgumentError("all values in p must be of type RSCVVal (AbstractVector or Distribution)"))
                end
            end
            if vd_flag
                for _p in p
                    @argcheck(all(x -> isa(x, GSCVKey), keys(_p)),
                              ArgumentError("all keys in p must be of type GSCVKey (String, Symbol, or Integer)"))
                    @argcheck(all(x -> isa(x, RSCVVal), values(_p)),
                              ArgumentError("all values in p must be of type RSCVVal (AbstractVector or Distribution)"))
                end
            end
        end
        assert_nonempty_gt0_finite_val(n_iter, :n_iter)
        return new{typeof(p), typeof(cv), typeof(r), typeof(scorer), typeof(ex),
                   typeof(n_iter), typeof(rng), typeof(seed), typeof(train_score),
                   typeof(kwargs)}(p, cv, r, scorer, ex, n_iter, rng, seed, train_score,
                                   kwargs)
    end
end
function RandomisedSearchCrossValidation(p::Union{<:AbstractVector{<:Pair},
                                                  <:AbstractVector{<:AbstractVector{<:Pair}},
                                                  <:AbstractDict,
                                                  <:AbstractVector{<:AbstractDict}};
                                         cv::CrossValidationEstimator = KFold(),
                                         r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                         scorer::CrossValSearchScorer = HighestMeanScore(),
                                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                         n_iter::Integer = 10,
                                         rng::Random.AbstractRNG = Random.default_rng(),
                                         seed::Option{<:Integer} = nothing,
                                         train_score::Bool = false,
                                         kwargs::NamedTuple = (;))
    return RandomisedSearchCrossValidation(p, cv, r, scorer, ex, n_iter, rng, seed,
                                           train_score, kwargs)
end
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
    """
    $(field_dict[:opt_cv])
    """
    opt
    """
    $(field_dict[:test_scores])
    """
    test_scores
    """
    $(field_dict[:train_scores])
    """
    train_scores
    """
    $(field_dict[:lens_grid])
    """
    lens_grid
    """
    $(field_dict[:val_grid])
    """
    val_grid
    """
    $(field_dict[:idx_cv])
    """
    idx
    function SearchCrossValidationResult(opt::AbstractEstimator, test_scores::MatNum,
                                         train_scores::Option{<:Union{<:MatNum,
                                                                      <:AbstractVector{<:MatNum}}},
                                         lens_grid::AbstractVector,
                                         val_grid::AbstractVector, idx::Integer)
        return new{typeof(opt), typeof(test_scores), typeof(train_scores),
                   typeof(lens_grid), typeof(val_grid), typeof(idx)}(opt, test_scores,
                                                                     train_scores,
                                                                     lens_grid, val_grid,
                                                                     idx)
    end
end
function SearchCrossValidationResult(; opt::AbstractEstimator, test_scores::MatNum,
                                     train_scores::Option{<:Union{<:MatNum,
                                                                  <:AbstractVector{<:MatNum}}},
                                     lens_grid::AbstractVector, val_grid::AbstractVector,
                                     idx::Integer)::SearchCrossValidationResult
    return SearchCrossValidationResult(opt, test_scores, train_scores, lens_grid, val_grid,
                                       idx)
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
                       scv::Union{<:GridSearchCrossValidation{<:Any, <:Any},
                                  <:RandomisedSearchCrossValidation{<:Any, <:Any}},
                       cv::CrossValidationResult, rd::ReturnsResult, i::Integer)
    prediction = fit_and_predict(opt, rd; train_idx = cv.train_idx[i],
                                 test_idx = cv.test_idx[i])
    r = scv.r
    sign = ifelse(bigger_is_better(r), 1, -1)
    test_score = sign * expected_risk(scv.r, prediction; scv.kwargs...)
    train_score = if scv.train_score
        sign * expected_risk(scv.r, prediction.res; scv.kwargs...)
    else
        nothing
    end
    return test_score, train_score
end
function fit_and_score(opt::NonFiniteAllocationOptimisationEstimator,
                       scv::Union{<:GridSearchCrossValidation{<:Any, <:MultipleRandomised},
                                  <:RandomisedSearchCrossValidation{<:Any,
                                                                    <:MultipleRandomised}},
                       cv::MultipleRandomisedResult, rd::ReturnsResult, i::Integer)
    prediction = fit_and_predict(opt, rd; train_idx = cv.train_idx[i],
                                 test_idx = cv.test_idx[i], cols = cv.asset_idx[i])
    r = scv.r
    sign = ifelse(bigger_is_better(r), 1, -1)
    test_score = sign * expected_risk(scv.r, prediction; scv.kwargs...)
    train_score = if scv.train_score
        sign * expected_risk(scv.r, prediction.res; scv.kwargs...)
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

  - [`expr_to_lens_chain`](@ref)
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

  - [`expr_to_lens_chain`](@ref)
"""
_eval_index(x::Integer) = x
_eval_index(x::Symbol) = x
_eval_index(ex::Expr)  = ex.head === :vect ? [_eval_index(a) for a in ex.args] : throw(Meta.ParseError("Unsupported index expression: $ex"))
"""
    expr_to_lens_chain(ex)

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
function expr_to_lens_chain(ex)
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
            throw(Meta.ParseError("Unsupported expression: $ex"))
        end
    end
    if !(ex isa Symbol)
        throw(Meta.ParseError("Unsupported expression: $ex"))
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

# Validation

  - String keys longer than `EQUATION_LIMITS[].max_length` are rejected before `Meta.parse`.
  - `Expr`/`Symbol` keys deeper than `EQUATION_LIMITS[].max_depth` are rejected before the lens-building walk.

# Returns

  - Composed Accessors.jl lens.

# Related

  - [`expr_to_lens_chain`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`EQUATION_LIMITS`](@ref)
"""
function parse_lens(key::AbstractString)
    # Trust boundary: cap the untrusted string length before `Meta.parse` and the
    # recursive lens-building walk, so a deeply nested key cannot exhaust the stack.
    lim = EQUATION_LIMITS[]
    @argcheck(length(key) <= lim.max_length,
              Meta.ParseError("Lens key string is too long ($(length(key)) > $(lim.max_length) characters)."))
    return expr_to_lens_chain(Meta.parse(key))
end
function parse_lens(key::Union{Expr, Symbol})
    # Trust-boundary defence for the pre-built-AST form (no string length cap applies):
    # reject an over-deep tree before `expr_to_lens_chain`'s recursive walk can exhaust the stack.
    lim = EQUATION_LIMITS[]
    @argcheck(!_expr_depth_exceeds(key, lim.max_depth),
              Meta.ParseError("Lens key expression is too deeply nested (exceeds depth $(lim.max_depth))."))
    return expr_to_lens_chain(key)
end
function parse_lens(key::Union{<:ComposedFunction, <:Accessors.PropertyLens,
                               <:Accessors.IndexLens})
    return key
end

export SearchCrossValidationResult, HighestMeanScore
