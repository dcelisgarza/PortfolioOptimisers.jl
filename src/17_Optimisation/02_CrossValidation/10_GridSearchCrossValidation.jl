"""
    lens_val_grid(estval::AbstractVector{<:Pair{<:Any, <:AbstractVector}})
    lens_val_grid(estval::AbstractDict{<:Any, <:AbstractVector})
    lens_val_grid(estvals::AbstractVector{<:Union{<:AbstractVector{<:Pair{<:Any, <:AbstractVector}},
                                                  <:AbstractDict{<:Any, <:AbstractVector}}})

Build the grid of a search, the lenses and the values of every candidate.

One parameter set gives the Cartesian product of the value vectors of its keys. A vector of parameter sets gives the concatenation of their grids, so the values of two sets never combine into one candidate.

# Mathematical definition

```math
\\begin{align}
\\Theta &= V_1 \\times V_2 \\times \\cdots \\times V_k\\,, \\\\
|\\Theta| &= \\prod_{j=1}^{k} |V_j|\\,, \\\\
\\Theta &= \\Theta^{(1)} \\Vert \\Theta^{(2)} \\Vert \\cdots \\Vert \\Theta^{(m)}\\,, \\\\
|\\Theta| &= \\sum_{l=1}^{m} |\\Theta^{(l)}|\\,.
\\end{align}
```

Where:

  - $(math_dict[:Theta_grid])
  - ``V_j``: Candidate values of the ``j``-th key of one parameter set, the value vector that the set gives the key.
  - ``k``: Count of the keys of one parameter set.
  - ``\\Theta^{(l)}``: Grid of the ``l``-th parameter set, by the first two equations.
  - ``m``: Count of the parameter sets.
  - ``\\Vert``: Concatenation of two sequences.

The first two equations hold for one parameter set, and the last two for a vector of sets. An empty ``V_j`` makes the product empty.

# Algorithm

For one parameter set, `estval`:

 1. Check the count of the grid through [`assert_search_grid_cap`](@ref), before the function builds the product.
 2. Collect the product of the value vectors into `vals`, one tuple per grid point.
 3. Parse each key into a lens through [`parse_lens`](@ref), and repeat that vector of lenses once per entry of `vals`, giving `lenses`.

For a vector of parameter sets, `estvals`:

 1. Build the grid of each set by the steps above, giving `lenses_vals`.
 2. Concatenate the lenses and the values of the sets in the order of `estvals`, giving `lenses` and `vals`.
 3. Check `length(vals)` through [`assert_search_grid_cap`](@ref).

# Arguments

  - `estval`: One parameter set, a vector of `key => values` pairs or a dictionary from key to values. A key is a property path that [`parse_lens`](@ref) reads, such as `"opt.l1"`.
  - `estvals`: A vector of parameter sets, each of the form of `estval`.

# Validation

  - Every value vector holds at least one value, through [`assert_search_grid_cap`](@ref).
  - The grid holds at most `RESOURCE_LIMITS[].max_search_grid` candidates, through [`assert_search_grid_cap`](@ref). The function checks the count of a set before it builds the product of that set. Under `estvals` it checks each set as it builds that set, and the sum of the sets after the concatenation.

# Returns

  - `lenses::AbstractVector`: One vector of lenses per grid point.
  - `vals::AbstractVector{<:Tuple}`: One tuple of values per grid point, in the order of the lenses at that point. Inside one set the first key varies fastest, and a dictionary gives its keys in its own order of iteration.

# Related

  - [`parse_lens`](@ref)
  - [`assert_search_grid_cap`](@ref)
  - [`pipeline_lens_val_grid`](@ref): the grid of a [`Pipeline`](@ref) search, whose keys can name a step.
  - [`GridSearchCrossValidation`](@ref)
  - [`search_cross_validation`](@ref)
"""
function lens_val_grid(estval::AbstractVector{<:Pair{<:Any, <:AbstractVector}})
    # Trust boundary: the grid is a Cartesian product, so cap it before `collect`
    # materialises it -- `k` parameters of `N` values are `N^k` candidates and `N^k` fits.
    assert_search_grid_cap(estval)
    vals = vec(collect(Iterators.product(map(x -> x[2], estval)...)))
    lenses = fill(map(x -> parse_lens(x[1]), estval), length(vals))
    return lenses, vals
end
function lens_val_grid(estval::AbstractDict{<:Any, <:AbstractVector})
    assert_search_grid_cap(estval)
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
    # Concatenated grids are a sum of products: each one is capped as it is built, this
    # caps what they add up to.
    assert_search_grid_cap(length(vals),
                           "the sum of the $(length(estvals)) concatenated parameter sets")
    return lenses, vals
end
"""
    search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                            gscv::GridSearchCrossValidation, rd::ReturnsResult)

Score every candidate of the grid of `gscv` on every fold of `gscv.cv`, and return the winner.

Each candidate runs through [`fit_and_predict`](@ref), the fold loop of every cross-validation entry point, so the candidate runs the scheme it declares. A walk-forward threads the weights of the previous fold through the `pws` of the scheme, and a [`TimeDependent`](@ref) schedule resolves per fold. Under an Online Scheme, every candidate warms up cold on the first training window and steps through the folds, and no state passes from one candidate to the next. The online search then gives the score matrix of the batch search over the expanding walk-forward. The two matrices agree to the tolerance of the moment estimators and of the solver, and the two searches select the same candidate.

# Mathematical definition

```math
\\begin{align}
S_{fi} &= s \\, \\mathcal{R}\\left(\\hat{P}_{f}(\\theta_i)\\right)\\,, \\quad f = 1, \\ldots, F\\,, \\quad \\theta_i \\in \\Theta\\,, \\\\
\\mathcal{C} &= \\left\\{ i : S_{fi} \\text{ is finite for every } f \\right\\}\\,, \\\\
i^{\\star} &= c_{\\sigma\\left(\\mathbf{S}_{:,\\,\\mathcal{C}}\\right)}\\,.
\\end{align}
```

Where:

  - ``S_{fi}``: Test score of candidate ``i`` on fold ``f``, the entry of the score matrix ``\\mathbf{S}``, ``F \\times |\\Theta|``.
  - ``\\hat{P}_{f}(\\theta_i)``: Prediction of fold ``f``, the returns over the test window of the fold of the weights that ``\\theta_i`` fits on its training window.
  - ``F``: Count of the folds that `split` enumerates.
  - $(math_dict[:theta_i_cand])
  - $(math_dict[:Theta_grid])
  - $(math_dict[:s_orient_search])
  - $(math_dict[:R_search])
  - $(math_dict[:C_finite_cand])
  - ``\\mathbf{S}_{:,\\,\\mathcal{C}}``: The columns of ``\\mathbf{S}`` at the finite candidates, in grid order.
  - ``c_j``: The ``j``-th smallest entry of ``\\mathcal{C}``.
  - $(math_dict[:sigma_scorer])
  - $(math_dict[:i_star_cand])

A candidate that failed one fold is never selected, whatever the scorer computes, because the scorer never reads its column.

# Algorithm

 1. Refuse an estimator that carries a partial-fit state, through [`assert_search_entry`](@ref).
 2. Build `lens_grid` and `val_grid` through [`lens_val_grid`](@ref).
 3. Run the entry checks of [`cross_val_predict`](@ref) on every candidate through [`assert_search_candidates`](@ref), before any candidate is scored.
 4. Fix the folds through [`pin_draw`](@ref), giving `scheme`. A scheme whose `split` draws at random then scores every candidate on the same folds.
 5. Split `rd` by `scheme`, giving `cv`.
 6. Read the rows of the folds through [`score_rows`](@ref), giving `rows`.
 7. Set `sgn` to ``s``.
 8. Allocate `test_scores`, of size `M × N` for `M` folds and `N` candidates, with the element type of `rd.X`. When `gscv.train_score` is `true`, allocate `train_scores` of the same size.
 9. When `gscv.train_score` is `true`, view the training returns of each fold through [`fold_train_returns`](@ref), giving `train_X`.
10. For each candidate `i`, in parallel over `gscv.ex`, do steps 11 to 13.
11. Build `opti` through [`search_candidate`](@ref).
12. Fit and predict `opti` over `scheme` through [`fit_and_predict`](@ref), with the folds in sequence, giving `predictions`.
13. Write column `i` of `test_scores` and `train_scores` through [`write_candidate_scores!`](@ref).
14. Select `opt_idx`, the position ``i^{\\star}``, through [`finite_candidate_index`](@ref).
15. Build the selected candidate `opt` through [`search_candidate`](@ref).
16. Return a [`SearchCrossValidationResult`](@ref).

# Arguments

  - `opt`: The estimator to tune. The search reads it as configuration alone.
  - `gscv`: The grid search. It gives the grid `p`, the scheme `cv`, the risk measure `r`, the `scorer`, the executor `ex`, the `train_score` flag and the keyword arguments `kwargs` of [`expected_risk`](@ref).
  - $(arg_dict[:rd])

# Validation

  - `opt` carries no partial-fit state, through [`assert_search_entry`](@ref). Under an Online Scheme, `opt` also passes every check of [`assert_online_entry`](@ref).
  - Every candidate passes the entry checks of [`cross_val_predict`](@ref), through [`assert_search_candidates`](@ref).
  - The grid is not empty, and it holds at most `RESOURCE_LIMITS[].max_search_grid` candidates, through [`lens_val_grid`](@ref).
  - At least one candidate finishes every fold. Otherwise [`finite_candidate_index`](@ref) throws an `IsNonFiniteError`.

# Returns

  - `res::SearchCrossValidationResult`: The field `test_scores` is ``\\mathbf{S}``. It is the raw matrix, with `NaN` at a failed fold, so its columns agree with the grid and a reader sees which fold failed. Row ``f`` is fold ``f`` in the order of `split`. Under a [`MultipleRandomised`](@ref), [`score_rows`](@ref) puts the scores of each path back onto the rows of the split. The field `train_scores` holds ``s \\, \\mathcal{R}`` of each fitted result over its own training window, in a matrix of the same size, or `nothing` when `gscv.train_score` is `false`. The field `opt` is candidate ``i^{\\star}``, and `idx` is ``i^{\\star}``.

# Related

  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref): its search samples a grid and runs this method on it.
  - [`fit_and_predict`](@ref)
  - [`finite_candidate_index`](@ref)
  - [`write_candidate_scores!`](@ref)
  - [`score_rows`](@ref)
  - [`pin_draw`](@ref)
  - [`assert_search_entry`](@ref)
  - [`assert_search_candidates`](@ref)
  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`ReturnsResult`](@ref)
"""
function search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                                 gscv::GridSearchCrossValidation, rd::ReturnsResult)
    assert_search_entry(opt, gscv.cv)
    lens_grid, val_grid = lens_val_grid(gscv.p)
    assert_search_candidates(opt, lens_grid, val_grid)
    scheme = pin_draw(gscv.cv)
    cv = split(scheme, rd)
    rows = score_rows(cv)
    N = length(val_grid)
    M = length(cv.train_idx)
    r = gscv.r
    sgn = ifelse(bigger_is_better(r), 1, -1)
    test_scores = Matrix{eltype(rd.X)}(undef, M, N)
    train_scores = if gscv.train_score
        Matrix{eltype(rd.X)}(undef, M, N)
    else
        nothing
    end
    # One view per fold, built once for the whole search rather than per candidate. It is
    # read only for a result that carries no carrier of its own; see
    # [`candidate_train_score`](@ref).
    train_X = if gscv.train_score
        [fold_train_returns(cv, rd, k) for k in eachindex(cv.train_idx)]
    else
        nothing
    end
    let opt = opt, test_scores = test_scores, train_scores = train_scores, train_X = train_X
        FLoops.@floop gscv.ex for (i, (lenses, vals)) in
                                  enumerate(zip(lens_grid, val_grid))
            local opti = search_candidate(opt, lenses, vals)
            # Candidates run in parallel over `gscv.ex`; the folds inside one run in
            # sequence, through the same loop every other entry point runs.
            local predictions = fit_and_predict(opti, rd, scheme;
                                                ex = FLoops.SequentialEx())
            write_candidate_scores!(test_scores, train_scores, i, predictions, rows,
                                    train_X, r, sgn, gscv.kwargs)
        end
    end
    opt_idx = finite_candidate_index(gscv.scorer, test_scores)
    opt = search_candidate(opt, lens_grid[opt_idx], val_grid[opt_idx])
    return SearchCrossValidationResult(; opt = opt, test_scores = test_scores,
                                       train_scores = train_scores, lens_grid = lens_grid,
                                       val_grid = val_grid, idx = opt_idx)
end
"""
    search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                            gscv::GridSearchCrossValidation{<:Any, <:CombinatorialCrossValidation},
                            rd::ReturnsResult)

Tune `opt` over the grid of `gscv` under a [`CombinatorialCrossValidation`](@ref), with one score per candidate and backtest path.

A combinatorial scheme recombines its disjoint test groups into full backtest paths, and a split scored alone mixes the groups of different paths. So this method scores each path. For each candidate, the scheme runs through [`fit_and_predict`](@ref), [`sort_predictions!`](@ref) recombines the groups into a [`PopulationPredictionResult`](@ref), and [`expected_risk`](@ref) gives one score per path. The candidates run in sequence, and the fold loop of each candidate runs with the executor `gscv.ex`.

# Mathematical definition

```math
\\begin{align}
S_{pi} &= s \\, \\mathcal{R}\\left(\\hat{P}_{p}(\\theta_i)\\right)\\,, \\quad p = 1, \\ldots, n_{p}\\,, \\quad \\theta_i \\in \\Theta\\,, \\\\
\\mathcal{C} &= \\left\\{ i : S_{pi} \\text{ is finite for every } p \\right\\}\\,, \\\\
i^{\\star} &= c_{\\sigma\\left(\\mathbf{S}_{:,\\,\\mathcal{C}}\\right)}\\,.
\\end{align}
```

Where:

  - ``S_{pi}``: Test score of candidate ``i`` on path ``p``, the entry of the score matrix ``\\mathbf{S}``, ``n_{p} \\times |\\Theta|``.
  - ``\\hat{P}_{p}(\\theta_i)``: Path ``p`` of candidate ``i``, the predictions of the test groups that the path holds, pooled into one series. Each group is predicted by the weights that ``\\theta_i`` fits on the training window of the fold that tests the group.
  - ``n_{p}``: Count of the paths, the greatest entry of `path_ids` of the split.
  - $(math_dict[:theta_i_cand])
  - $(math_dict[:Theta_grid])
  - $(math_dict[:s_orient_search])
  - $(math_dict[:R_search])
  - $(math_dict[:C_finite_cand])
  - ``\\mathbf{S}_{:,\\,\\mathcal{C}}``: The columns of ``\\mathbf{S}`` at the finite candidates, in grid order.
  - ``c_j``: The ``j``-th smallest entry of ``\\mathcal{C}``.
  - $(math_dict[:sigma_scorer])
  - $(math_dict[:i_star_cand])

A candidate that failed one path is never selected, whatever the scorer computes, because the scorer never reads its column.

# Algorithm

 1. Refuse an estimator that carries a partial-fit state, through [`assert_search_entry`](@ref).
 2. Build `lens_grid` and `val_grid` through [`lens_val_grid`](@ref).
 3. Run the entry checks of [`cross_val_predict`](@ref) on every candidate through [`assert_search_candidates`](@ref), before any candidate is scored.
 4. Split `rd` by `gscv.cv`, giving `cv`.
 5. Set `M`, the count of the paths, to the greatest entry of `cv.path_ids`.
 6. Set `sgn` to ``s``.
 7. Allocate `test_scores`, of size `M × N` for `N` candidates, with the element type of `rd.X`. When `gscv.train_score` is `true`, allocate `train_scores`, one matrix per path, with one row per fold of the path and one column per candidate.
 8. View the training returns of each fold of each path through [`fold_train_returns`](@ref), giving `path_X`. The entries of a path in `cv.path_ids` are read in column order, and the column is the fold, which is the order of the folds in the path that [`sort_predictions!`](@ref) gives.
 9. For each candidate `i`, in sequence, do steps 10 to 13.
10. Build `opti` through [`search_candidate`](@ref).
11. Fit and predict `opti` over `gscv.cv` through [`fit_and_predict`](@ref), with the executor `gscv.ex`, giving `predictions`, one path per member.
12. Write ``s \\, \\mathcal{R}`` of each path of `predictions` into column `i` of `test_scores`.
13. When `gscv.train_score` is `true`, write the train score of each fold of each path into column `i` of the matrix of that path, through [`candidate_train_score`](@ref) with the view of `path_X`.
14. Select `opt_idx`, the position ``i^{\\star}``, through [`finite_candidate_index`](@ref).
15. Build the selected candidate `opt` through [`search_candidate`](@ref).
16. Return a [`SearchCrossValidationResult`](@ref).

# Arguments

  - `opt`: The estimator to tune. The search reads it as configuration alone.
  - `gscv`: The grid search over a [`CombinatorialCrossValidation`](@ref). It gives the grid `p`, the scheme `cv`, the risk measure `r`, the `scorer`, the executor `ex`, the `train_score` flag and the keyword arguments `kwargs` of [`expected_risk`](@ref).
  - $(arg_dict[:rd])

# Validation

  - `opt` carries no partial-fit state, through [`assert_search_entry`](@ref).
  - Every candidate passes the entry checks of [`cross_val_predict`](@ref), through [`assert_search_candidates`](@ref).
  - The grid is not empty, and it holds at most `RESOURCE_LIMITS[].max_search_grid` candidates, through [`lens_val_grid`](@ref).
  - At least one candidate finishes every path. Otherwise [`finite_candidate_index`](@ref) throws an `IsNonFiniteError`.

# Returns

  - `res::SearchCrossValidationResult`: The field `test_scores` is ``\\mathbf{S}``, the raw matrix, with `NaN` at a failed path. The field `train_scores` is `nothing` when `gscv.train_score` is `false`. Otherwise it is a vector of ``n_{p}`` matrices, one per path, and each matrix holds one row per fold of the path and one column per candidate. The method keeps the train scores per fold, because the folds of one path train on different windows. The test returns of one path pool into one series, so the test score is one per path. The field `opt` is candidate ``i^{\\star}``, and `idx` is ``i^{\\star}``.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref): its search samples a grid and runs this method on it under a combinatorial scheme.
  - [`fit_and_predict`](@ref)
  - [`sort_predictions!`](@ref)
  - [`expected_risk`](@ref)
  - [`finite_candidate_index`](@ref)
  - [`candidate_train_score`](@ref)
  - [`fold_train_returns`](@ref)
"""
function search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                                 gscv::GridSearchCrossValidation{<:Any,
                                                                 <:CombinatorialCrossValidation},
                                 rd::ReturnsResult)
    assert_search_entry(opt, gscv.cv)
    lens_grid, val_grid = lens_val_grid(gscv.p)
    assert_search_candidates(opt, lens_grid, val_grid)
    cv = split(gscv.cv, rd)
    N = length(val_grid)
    M = maximum(cv.path_ids)          # one score per recombined backtest path
    r = gscv.r
    sgn = ifelse(bigger_is_better(r), 1, -1)
    test_scores = Matrix{eltype(rd.X)}(undef, M, N)
    # Train scores are per fold, and each path holds a different number of folds, so they
    # are kept as one `folds × candidates` matrix per path (a Vector of matrices) rather
    # than collapsed. Test scores stay one per path.
    train_scores = if gscv.train_score
        [Matrix{eltype(rd.X)}(undef, count(==(p), cv.path_ids), N) for p in 1:M]
    else
        nothing
    end
    # One training view per fold of each path. A combinatorial `path_ids` is a matrix, one
    # row per test block of a fold, so a path's entries are Cartesian and the fold is the
    # **column**. The folds of a path arrive in that order, measured against `res.pr.X` of
    # each prediction. The views are lazy, so they are built whether or not a train score is
    # asked for, and read only for a result that carries no carrier of its own; see
    # [`candidate_train_score`](@ref).
    path_X = [[fold_train_returns(cv, rd, I[2]) for I in findall(==(p), cv.path_ids)]
              for p in 1:M]
    for (i, (lenses, vals)) in enumerate(zip(lens_grid, val_grid))
        opti = search_candidate(opt, lenses, vals)
        # Fold-level parallelism happens inside fit_and_predict; the candidate loop is
        # sequential to avoid nested threading.
        predictions = fit_and_predict(opti, rd, gscv.cv; ex = gscv.ex)
        test_scores[:, i] = sgn * expected_risk(r, predictions; gscv.kwargs...)
        if gscv.train_score
            for (p, path) in enumerate(predictions.pred)
                # A two-argument `map`, not a comprehension that destructures a pair: over
                # the method's declared signature JET cannot see through that destructuring.
                train_scores[p][:, i] = map(path.pred, path_X[p]) do fp, X
                    return sgn * candidate_train_score(r, fp.res, X, gscv.kwargs)
                end
            end
        end
    end
    opt_idx = finite_candidate_index(gscv.scorer, test_scores)
    opt = search_candidate(opt, lens_grid[opt_idx], val_grid[opt_idx])
    return SearchCrossValidationResult(; opt = opt, test_scores = test_scores,
                                       train_scores = train_scores, lens_grid = lens_grid,
                                       val_grid = val_grid, idx = opt_idx)
end
export search_cross_validation, GridSearchCrossValidation
