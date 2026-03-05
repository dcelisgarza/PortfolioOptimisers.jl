abstract type SearchCrossValidationEstimator <: AbstractEstimator end
abstract type SearchCrossValidationResult <: AbstractResult end
abstract type SearchCrossValidationAlgorithm <: AbstractAlgorithm end

struct GridSearchCrossValidation{T1, T2, T3} <: SearchCrossValidationEstimator
    p::T1
    cv::T2
    ex::T3
    function GridSearchCrossValidation(p::MultiSCVValType, cv::CrossValidationEstimator,
                                       ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
        @argcheck(!isempty(p), IsEmptyError)
        return new{typeof(p), typeof(cv), typeof(ex)}(p, cv, ex)
    end
end
function GridSearchCrossValidation(; p::MultiSCVValType,
                                   cv::CrossValidationEstimator = KFold(),
                                   ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    return GridSearchCrossValidation(p, cv, ex)
end
struct GridSearchCrossValidationResult <: SearchCrossValidationResult end
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
function key_vals(estval::AbstractVector{<:PairSCV})
    return map(x -> x[1], estval), map(x -> x[2], estval)
end
function key_vals(estval::DictSCV)
    return keys(estval), values(estval)
end
function grid_search_cross_validation(opt, gscv::GridSearchCrossValidation,
                                      rd::ReturnsResult)
    p = gscv.p
    ks, vals = key_vals(p)
    lenses = build_nested_lens(ks)
    val_grid = Iterators.product(vals...)
    res = Vector{AbstractPredictionResult}(undef, length(val_grid))
    cv = split(gscv.cv, rd)
    # folds = Iterators.product(zip(1:length(val_grid), val_grid),
    #                           zip(1:length(cv.train_idx), cv.train_idx, cv.test_idx))
    FLoops.@floop gscv.ex for (i, v_grid) in enumerate(val_grid)
        local opti = opt
        for (lens, val) in zip(lenses, v_grid)
            opti = Accessors.set(opti, lens, val)
        end
        res[i] = cross_val_predict(opti, rd, cv; ex = gscv.ex)
        #! Use fit predict on each test and train split and score them
    end
    return res
end
export grid_search_cross_validation, GridSearchCrossValidation
#=
#! Asked mr Gippitty to write this, it would not work but it gives ideas.
"""
    GridSearchCVResult

Result type for grid search cross-validation. Stores parameter grid, scores, and best parameters.

# Fields

  - `param_grid::Vector{NamedTuple}`: List of parameter combinations.
  - `mean_scores::Vector{Float64}`: Mean score for each parameter set.
  - `std_scores::Vector{Float64}`: Std deviation of scores for each parameter set.
  - `all_scores::Vector{Vector{Float64}}`: All fold scores for each parameter set.
  - `best_index::Int`: Index of best parameter set.
  - `best_params::NamedTuple`: Best parameter set.
  - `best_score::Float64`: Best mean score.

# Example

```julia
result = grid_search_cv(estimator_constructor, param_grid, X, y; n_folds = 5,
                        scorer = my_metric)
println(result.best_params)
println(result.best_score)
```
"""
Base.@kwdef struct GridSearchCVResult
    param_grid::Vector{NamedTuple}
    mean_scores::Vector{Float64}
    std_scores::Vector{Float64}
    all_scores::Vector{Vector{Float64}}
    best_index::Int
    best_params::NamedTuple
    best_score::Float64
end

"""
    grid_search_cv(estimator_constructor, param_grid, X, y; n_folds=5, scorer, rng=Random.GLOBAL_RNG)

Perform grid search cross-validation over a parameter grid for an estimator.

# Arguments

  - `estimator_constructor`: Function or type that constructs an estimator given keyword arguments.
  - `param_grid`: Vector of NamedTuples, each specifying a parameter combination.
  - `X`: Feature matrix or data input.
  - `y`: Target vector or data output.
  - `n_folds`: Number of CV folds (default: 5).
  - `scorer`: Function `(y_true, y_pred) -> score` to evaluate performance (higher is better).
  - `rng`: Random number generator (default: Random.GLOBAL_RNG).

# Returns

  - `GridSearchCVResult` summarizing grid search outcomes.

# Example

```julia
param_grid = [(alpha = a, beta = b) for a in 0.1:0.1:0.5, b in 1:2]
result = grid_search_cv(MyEstimator, param_grid, X, y; n_folds = 3, scorer = my_metric)
```
"""
function grid_search_cv(estimator_constructor, param_grid, X, y; n_folds = 5, scorer,
                        rng = Random.GLOBAL_RNG)
    @argcheck !isempty(param_grid) "Parameter grid must not be empty."
    n = size(X, 1)
    idx = collect(1:n)
    Random.shuffle!(rng, idx)
    fold_sizes = fill(div(n, n_folds), n_folds)
    for i in 1:rem(n, n_folds)
        fold_sizes[i] += 1
    end
    folds = []
    start = 1
    for sz in fold_sizes
        push!(folds, idx[start:(start + sz - 1)])
        start += sz
    end
    mean_scores = Float64[]
    std_scores = Float64[]
    all_scores = Vector{Vector{Float64}}()
    for params in param_grid
        scores = Float64[]
        for i in 1:n_folds
            val_idx = folds[i]
            train_idx = vcat(folds[1:(i - 1)]..., folds[(i + 1):end]...)
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_val, y_val = X[val_idx, :], y[val_idx]
            est = estimator_constructor(; params...)
            fit!(est, X_train, y_train)
            y_pred = predict(est, X_val)
            push!(scores, scorer(y_val, y_pred))
        end
        push!(mean_scores, mean(scores))
        push!(std_scores, std(scores))
        push!(all_scores, scores)
    end
    best_index = argmax(mean_scores)
    return GridSearchCVResult(; param_grid = param_grid, mean_scores = mean_scores,
                              std_scores = std_scores, all_scores = all_scores,
                              best_index = best_index, best_params = param_grid[best_index],
                              best_score = mean_scores[best_index])
end
=#
