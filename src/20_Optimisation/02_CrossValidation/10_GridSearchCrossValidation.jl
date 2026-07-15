"""
    lens_val_grid(estval)

Build a grid of (lens, value) pairs from a parameter specification.

Converts the input vector of `key => values` pairs into a grid of Accessors.jl lens and value combinations for grid search cross-validation.

# Arguments

  - `estval`: Vector of `String => AbstractVector` pairs mapping parameter key paths to their candidate values.

# Returns

  - Grid of (lens, value) combinations.

# Related

  - [`parse_lens`](@ref)
  - [`GridSearchCrossValidation`](@ref)
"""
function lens_val_grid(estval::AbstractVector{<:Pair{<:Any, <:AbstractVector}})
    vals = vec(collect(Iterators.product(map(x -> x[2], estval)...)))
    lenses = fill(map(x -> parse_lens(x[1]), estval), length(vals))
    return lenses, vals
end
function lens_val_grid(estval::AbstractDict{<:Any, <:AbstractVector})
    vals = vec(collect(Iterators.product(values(estval)...)))
    lenses = fill(map(x -> parse_lens(x), collect(keys(estval))), length(vals))
    return lenses, vals
end
function lens_val_grid(estvals::AbstractVector{<:Union{<:AbstractVector{<:Pair{<:Any,
                                                                               <:AbstractVector}},
                                                       <:AbstractDict{<:Any,
                                                                      <:AbstractVector}}})
    lenses_vals = [lens_val_grid(estval) for estval in estvals]
    lenses = mapreduce(x -> x[1], vcat, lenses_vals)
    vals = mapreduce(x -> x[2], vcat, lenses_vals)
    return lenses, vals
end
"""
    search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                           gscv::GridSearchCrossValidation,
                           rd::ReturnsResult)

Performs grid search cross-validation for portfolio optimisation estimators. Iterates over parameter grids, applies cross-validation splits, fits and scores each configuration, and selects the optimal parameters using the provided scoring strategy.

# Arguments

  - `opt`: Portfolio optimisation estimator to be tuned.
  - `gscv`: Grid search cross-validation estimator specifying parameter grid, CV splitter, risk measure, scorer, execution strategy, and options.
  - `rd`: Returns result containing asset returns data.

# Returns

  - `SearchCrossValidationResult`: Result type containing the optimal estimator, test and train scores, parameter grid, and selected index.

# Details

  - Iterates over all parameter combinations in the grid.
  - Applies cross-validation splits to the returns data.
  - Fits the estimator for each parameter set and split.
  - Scores each configuration using the specified risk measure and scoring function.
  - Selects the optimal parameter set based on cross-validation scores.
  - Returns a result object encapsulating the optimal estimator and score matrices.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`ReturnsResult`](@ref)
  - [`GridSearchCrossValidation`](@ref)
"""
function search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                                 gscv::GridSearchCrossValidation, rd::ReturnsResult)
    p = gscv.p
    lens_grid, val_grid = lens_val_grid(p)
    cv = split(gscv.cv, rd)
    N = length(val_grid)
    M = length(cv.train_idx)
    test_scores = Matrix{eltype(rd.X)}(undef, M, N)
    train_scores = if gscv.train_score
        Matrix{eltype(rd.X)}(undef, M, N)
    else
        nothing
    end
    let opt = opt, test_scores = test_scores, train_scores = train_scores
        FLoops.@floop gscv.ex for (i, (lenses, vals)) in
                                  enumerate(zip(lens_grid, val_grid))
            local opti = opt
            for (lens, val) in zip(lenses, vals)
                opti = Accessors.set(opti, lens, val)
            end
            for j in eachindex(cv.train_idx)
                test_score, train_score = fit_and_score(opti, gscv, cv, rd, j)
                test_scores[j, i] = test_score
                if gscv.train_score
                    train_scores[j, i] = train_score
                end
            end
        end
    end
    opt_idx = gscv.scorer(test_scores)
    opt_lens = lens_grid[opt_idx]
    opt_vals = val_grid[opt_idx]
    for (lens, val) in zip(opt_lens, opt_vals)
        opt = Accessors.set(opt, lens, val)
    end
    return SearchCrossValidationResult(; opt = opt, test_scores = test_scores,
                                       train_scores = train_scores, lens_grid = lens_grid,
                                       val_grid = val_grid, idx = opt_idx)
end
"""
    search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                            gscv::GridSearchCrossValidation{<:Any, <:CombinatorialCrossValidation},
                            rd::ReturnsResult)

Grid search cross-validation over a [`CombinatorialCrossValidation`](@ref) scheme.

Unlike the contiguous schemes (which score one fold per row), combinatorial cross-validation
recombines its disjoint test groups into full-length backtest **paths**. Scoring a single
split in isolation would mix groups belonging to different paths, so this method scores
per-path instead: for each candidate the whole scheme is run through [`fit_and_predict`](@ref)
(splits fitted, groups recombined by [`sort_predictions!`](@ref) into a
[`PopulationPredictionResult`](@ref)), and [`expected_risk`](@ref) yields one score per path.
The score matrix is therefore `n_paths × n_candidates`; the scorer selects across candidates
exactly as for the other schemes. The randomised form delegates here through its grid.

`train_scores` (only when `gscv.train_score`) keeps every per-fold in-sample score rather than
collapsing to one number per path: it is a `Vector` of `n_paths` matrices, one per path, each
`folds_in_path × n_candidates`. (Test scores stay one-per-path because a path's out-of-sample
returns pool into a single series, whereas its folds train on distinct in-sample windows.)

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`fit_and_predict`](@ref)
  - [`expected_risk`](@ref)
  - [`search_cross_validation`](@ref)
"""
function search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                                 gscv::GridSearchCrossValidation{<:Any,
                                                                 <:CombinatorialCrossValidation},
                                 rd::ReturnsResult)
    lens_grid, val_grid = lens_val_grid(gscv.p)
    cv = split(gscv.cv, rd)
    N = length(val_grid)
    M = maximum(cv.path_ids)          # one score per recombined backtest path
    r = gscv.r
    sgn = ifelse(bigger_is_better(r), 1, -1)
    test_scores = Matrix{eltype(rd.X)}(undef, M, N)
    # Train scores are per fold, and each path holds a different number of folds, so they
    # are kept as one `folds × candidates` matrix per path (a Vector of matrices) rather
    # than collapsed — test scores stay one-per-path.
    train_scores = if gscv.train_score
        [Matrix{eltype(rd.X)}(undef, count(==(p), cv.path_ids), N) for p in 1:M]
    else
        nothing
    end
    for (i, (lenses, vals)) in enumerate(zip(lens_grid, val_grid))
        opti = opt
        for (lens, val) in zip(lenses, vals)
            opti = Accessors.set(opti, lens, val)
        end
        # Fold-level parallelism happens inside fit_and_predict; the candidate loop is
        # sequential to avoid nested threading.
        predictions = fit_and_predict(opti, rd, gscv.cv; ex = gscv.ex)
        test_scores[:, i] = sgn * expected_risk(r, predictions; gscv.kwargs...)
        if gscv.train_score
            for (p, path) in enumerate(predictions.pred)
                train_scores[p][:, i] = [sgn * expected_risk(r, fp.res; gscv.kwargs...)
                                         for fp in path.pred]
            end
        end
    end
    opt_idx = gscv.scorer(test_scores)
    for (lens, val) in zip(lens_grid[opt_idx], val_grid[opt_idx])
        opt = Accessors.set(opt, lens, val)
    end
    return SearchCrossValidationResult(; opt = opt, test_scores = test_scores,
                                       train_scores = train_scores, lens_grid = lens_grid,
                                       val_grid = val_grid, idx = opt_idx)
end
export search_cross_validation, GridSearchCrossValidation
