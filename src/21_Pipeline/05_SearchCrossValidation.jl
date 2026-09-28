"""
    pipeline_data_view(pr::AbstractPricesResult, idx, idx2 = :)
    pipeline_data_view(rd::AbstractReturnsResult, idx, idx2 = :)

Return the view of price or returns data at the observations `idx` and the assets `idx2`.

The two methods give one call for the two input levels of a [`Pipeline`](@ref), so the code that cuts a window from the input of a Pipeline does not read the level of the input.

# Arguments

  - `pr`: The price data, an [`AbstractPricesResult`](@ref).
  - `rd`: The returns data, an [`AbstractReturnsResult`](@ref).
  - `idx`: The rows of the observations to keep.
  - `idx2`: The columns of the assets to keep. The default `:` keeps every asset.

# Returns

  - `data`: The view of the input at the same level, from [`port_opt_view`](@ref).

# Related

  - [`port_opt_view`](@ref)
  - [`pipeline_asset_view`](@ref): the view at every observation and a subset of the assets.
  - [`PipelineBufferState`](@ref): its buffer keeps the last rows of the input through this view.
  - [`pipeline_fold_fit`](@ref): the fit of one fold reads its training window through this view.
"""
pipeline_data_view(pr::AbstractPricesResult, idx, idx2 = :) = port_opt_view(pr, idx, idx2)
pipeline_data_view(rd::AbstractReturnsResult, idx, idx2 = :) = port_opt_view(rd, idx, idx2)
"""
    pipeline_asset_view(data::AbstractReturnsResult, cols)
    pipeline_asset_view(data::AbstractPricesResult, cols)

Return the view of price or returns data at every observation and the assets `cols`.

The two levels take the assets at different positions in [`port_opt_view`](@ref). The returns level takes them second, `port_opt_view(data, cols)`. The price level takes the observations second and the assets third, `port_opt_view(data, :, cols)`. This function gives one call for the two levels, so [`pipeline_path_fit_and_predict`](@ref) does not read the level of its data.

# Arguments

  - `data`: The price data, an [`AbstractPricesResult`](@ref), or the returns data, an [`AbstractReturnsResult`](@ref).
  - `cols`: The columns of the assets to keep.

# Returns

  - `data`: The view of `data` at the same level.

# Related

  - [`pipeline_data_view`](@ref): the view at a window of the observations.
  - [`pipeline_path_fit_and_predict`](@ref)
  - [`MultipleRandomised`](@ref): each path of the scheme fits a Pipeline on a subset of the assets, through this view.
"""
pipeline_asset_view(data::AbstractReturnsResult, cols) = port_opt_view(data, cols)
pipeline_asset_view(data::AbstractPricesResult, cols) = port_opt_view(data, :, cols)
"""
    cv_data_eltype(rd::AbstractReturnsResult)
    cv_data_eltype(pr::AbstractPricesResult)

Return the element type of the score matrices of a Pipeline search over the data `rd` or `pr`.

The type comes from the data. A score is a fraction, and a failed fold scores `NaN`, so [`float_if_integer`](@ref) changes an integer type to a floating point type. Every other type stays as it is.

# Arguments

  - `rd`: The returns data, an [`AbstractReturnsResult`](@ref).
  - `pr`: The price data, an [`AbstractPricesResult`](@ref).

# Returns

  - `T::Type`: The element type of `rd.X`, or of the values of the time array `pr.X`, through [`float_if_integer`](@ref).

# Related

  - [`float_if_integer`](@ref)
  - [`search_cross_validation`](@ref)
"""
cv_data_eltype(rd::AbstractReturnsResult) = float_if_integer(eltype(rd.X))
cv_data_eltype(pr::AbstractPricesResult) = float_if_integer(eltype(TimeSeries.values(pr.X)))
"""
    is_pipeline_raw_path(key::AbstractString) -> Bool

Return `true` when `key` is a property path whose root is the `steps` field of a [`Pipeline`](@ref), such as `"steps[1]"` or `"steps[2].fill"`.

[`pipeline_lens`](@ref) calls this function on a key whose first segment is not a step name. A `Pipeline` has two fields, `names` and `steps`, and only `steps` holds the estimators that a grid tunes. So the function admits the one root `steps`. It returns `false` for the path `"names[1]"`, and for a typo such as `"gapfill"`.

# Algorithm

 1. Find `i`, the position of the first `.` or `[` in `key`.
 2. Return `false` when `key` holds neither character.
 3. Return `true` when the text before position `i` is `steps`, and `false` otherwise.

# Arguments

  - `key`: The tuning key, as the grid gives it.

# Returns

  - `flag::Bool`: `true` when the root of `key` is `steps`.

# Related

  - [`pipeline_lens`](@ref)
  - [`parse_lens`](@ref)
"""
function is_pipeline_raw_path(key::AbstractString)
    i = findfirst(c -> c == '.' || c == '[', key)
    return !isnothing(i) && SubString(key, firstindex(key), prevind(key, i)) == "steps"
end
"""
    pipeline_lens(pipe::Pipeline, key::AbstractString)
    pipeline_lens(pipe::Pipeline, key::Symbol)
    pipeline_lens(pipe::Pipeline, key::Integer)
    pipeline_lens(pipe::Pipeline, key::Expr)
    pipeline_lens(pipe::Pipeline, key)

Return the Accessors.jl lens on `pipe` that a tuning key of a Pipeline search names.

A key names a step by its name or by its position, or it gives a property path whose root is the `steps` field. A key that names a step addresses the whole step, so a grid value can replace a whole estimator. A `String` key can add a property path after the step name, such as `"gap_fill.fill"`.

The function refuses every other key. A lens with a different root sets a field of the `Pipeline` itself, and that tunes no step. The typo `"gapfill"` and the path `"names[1]"`, which addresses the table of step names, are two such keys.

# Algorithm

For a `String` key:

 1. Split `key` at its first `.`, giving `parts`.
 2. Find `i`, the position of `parts[1]` in `pipe.names`.
 3. When `parts[1]` is not a step name, refuse `key` unless [`is_pipeline_raw_path`](@ref) admits it, and return the lens of `key` from [`parse_lens`](@ref).
 4. Build `step`, the lens on entry `i` of `pipe.steps`.
 5. When `key` holds no `.`, return `step`. Otherwise return the lens of `parts[2]` from [`parse_lens`](@ref), composed after `step`.

For a `Symbol` key:

 1. Find `i`, the position of `key` in `pipe.names`.
 2. Refuse `key` when it is not a step name, and return the lens on entry `i` of `pipe.steps` otherwise.

For an `Integer` key, check that `key` is a step position, and return the lens on entry `key` of `pipe.steps`.

For an `Expr` key:

 1. Build `lens` through [`parse_lens`](@ref).
 2. Find the root of `key`, the first argument of the expression at each level down to a `Symbol`.
 3. Refuse `key` when its root is not `steps`, and return `lens` otherwise.

For a lens that the caller builds, return `key` through [`parse_lens`](@ref) unchanged.

# Arguments

  - `pipe`: The pipeline that the search tunes.
  - `key`: The tuning key, a [`GSCVKey`](@ref).

# Validation

  - A `String` key starts with a step name, or [`is_pipeline_raw_path`](@ref) admits it. Otherwise an `ArgumentError` names the key and suggests the closest step name.
  - A `Symbol` key is a step name. Julia reads a dotted `Symbol` as one name, so the function refuses it, and the `ArgumentError` asks for a `String` path.
  - An `Integer` key is in `1:length(pipe.steps)`.
  - The root of an `Expr` key is `steps`.
  - A key that reaches [`parse_lens`](@ref) passes its caps on length and depth.

# Returns

  - `lens`: The Accessors.jl lens on `pipe`.

# Related

  - [`parse_lens`](@ref)
  - [`is_pipeline_raw_path`](@ref)
  - [`pipeline_lens_val_grid`](@ref)
  - [`Pipeline`](@ref)
  - [`search_cross_validation`](@ref)
"""
function pipeline_lens(pipe::Pipeline, key::AbstractString)
    parts = split(key, '.'; limit = 2)
    i = findfirst(==(parts[1]), pipe.names)
    if isnothing(i)
        # A key that misses the step-name table is a lens path only when it is rooted at
        # `steps`, the one `Pipeline` field that holds tunable estimators. Every other root
        # is refused rather than read as a property access on the pipeline struct, where a
        # path into the step-name table (`"names[1]"`) is written into on every fold. An
        # admitted path still falls through to `parse_lens`, which is structurally capped.
        @argcheck(is_pipeline_raw_path(key),
                  ArgumentError("`$(key)` is not a step name among the $(length(pipe.names)) named pipeline steps, nor a property path rooted at `steps`" *
                                did_you_mean(key, pipe.names)))
        return parse_lens(key)
    end
    step = Accessors.IndexLens((i,)) ∘ Accessors.PropertyLens(:steps)
    return length(parts) == 1 ? step : parse_lens(parts[2]) ∘ step
end
function pipeline_lens(pipe::Pipeline, key::Symbol)
    ks = string(key)
    i = findfirst(==(ks), pipe.names)
    if isnothing(i)
        # A symbol is one property name, never a path: `parse_lens` would turn
        # `Symbol("steps[1].fill")` into a lens on a field of that literal name, which no
        # `Pipeline` has. So a symbol that misses the step-name table is refused.
        throw(ArgumentError("`$(key)` is not a step name among the $(length(pipe.names)) named pipeline steps. A `Symbol` key names a whole step, so write a property path as a `String`, such as `\"steps[1].fill\"`." *
                            did_you_mean(ks, pipe.names)))
    end
    return Accessors.IndexLens((i,)) ∘ Accessors.PropertyLens(:steps)
end
function pipeline_lens(pipe::Pipeline, key::Integer)
    @argcheck(1 <= key <= length(pipe.steps),
              ArgumentError("step position $key is out of bounds for a pipeline with $(length(pipe.steps)) steps"))
    return Accessors.IndexLens((Int(key),)) ∘ Accessors.PropertyLens(:steps)
end
function pipeline_lens(::Pipeline, key::Expr)
    # `parse_lens` caps the depth first, so the walk to the root below is bounded.
    lens = parse_lens(key)
    root = key
    while root isa Expr
        root = root.args[1]
    end
    @argcheck(root === :steps,
              ArgumentError("`$(key)` is not a property path rooted at `steps`, the one field of a `Pipeline` that holds the estimators a grid tunes. Name a step by a `String`, a `Symbol` or an `Integer` key, or root the path at `steps`, such as `:(steps[1].fill)`."))
    return lens
end
function pipeline_lens(::Pipeline, key)
    return parse_lens(key)
end
"""
    pipeline_lens_val_grid(pipe::Pipeline,
                           estval::AbstractVector{<:Pair{<:Any, <:AbstractVector}})
    pipeline_lens_val_grid(pipe::Pipeline, estval::AbstractDict{<:Any, <:AbstractVector})
    pipeline_lens_val_grid(pipe::Pipeline,
                           estvals::AbstractVector{<:Union{<:AbstractVector{<:Pair{<:Any, <:AbstractVector}},
                                                           <:AbstractDict{<:Any, <:AbstractVector}}})

Build the grid of a Pipeline search, the lenses and the values of every candidate.

The grid is the grid of [`lens_val_grid`](@ref). The difference is the lenses: [`pipeline_lens`](@ref) resolves each key on `pipe`, so a key can name a step by its name or by its position.

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
  - $(math_dict[:V_j_grid])
  - $(math_dict[:k_grid_keys])
  - $(math_dict[:Theta_l_grid])
  - $(math_dict[:m_grid_sets])
  - $(math_dict[:Vert_concat])

The first two equations hold for one parameter set, and the last two for a vector of sets.

# Algorithm

For one parameter set, `estval`:

 1. Check the count of the grid through [`assert_search_grid_cap`](@ref), before the function builds the product.
 2. Collect the product of the value vectors into `vals`, one tuple per grid point.
 3. Resolve each key into a lens on `pipe` through [`pipeline_lens`](@ref), and repeat that vector of lenses once per entry of `vals`, giving `lenses`.

For a vector of parameter sets, `estvals`:

 1. Build the grid of each set by the steps above, giving `lenses_vals`.
 2. Concatenate the lenses and the values of the sets in the order of `estvals`, giving `lenses` and `vals`.
 3. Check `length(vals)` through [`assert_search_grid_cap`](@ref).

# Arguments

  - `pipe`: The pipeline that the search tunes.
  - `estval`: One parameter set, a vector of `key => values` pairs or a dictionary from key to values. A key is a tuning key that [`pipeline_lens`](@ref) reads, such as `"gap_fill.fill"`.
  - `estvals`: A vector of parameter sets, each of the form of `estval`.

# Validation

  - Every value vector holds at least one value, through [`assert_search_grid_cap`](@ref).
  - The grid holds at most `RESOURCE_LIMITS[].max_search_grid` candidates, through [`assert_search_grid_cap`](@ref). The function checks the count of a set before it builds the product of that set. Under `estvals` it checks each set as it builds that set, and the sum of the sets after the concatenation.
  - Every key names a step or a path rooted at `steps`, through [`pipeline_lens`](@ref).

# Returns

  - `lenses::AbstractVector`: One vector of lenses per grid point.
  - `vals::AbstractVector{<:Tuple}`: One tuple of values per grid point, in the order of the lenses at that point. Inside one set the first key varies fastest, and a dictionary gives its keys in its own order of iteration.

# Related

  - [`lens_val_grid`](@ref): the grid of a search over an optimiser.
  - [`pipeline_lens`](@ref)
  - [`assert_search_grid_cap`](@ref)
  - [`search_cross_validation`](@ref)
"""
function pipeline_lens_val_grid(pipe::Pipeline,
                                estval::AbstractVector{<:Pair{<:Any, <:AbstractVector}})
    # Same product sink as `lens_val_grid`, same cap, same place -- before the `collect`.
    assert_search_grid_cap(estval)
    vals = vec(collect(Iterators.product(map(x -> x[2], estval)...)))
    lenses = fill(map(x -> pipeline_lens(pipe, x[1]), estval), length(vals))
    return lenses, vals
end
function pipeline_lens_val_grid(pipe::Pipeline,
                                estval::AbstractDict{<:Any, <:AbstractVector})
    assert_search_grid_cap(estval)
    vals = vec(collect(Iterators.product(values(estval)...)))
    lenses = fill(map(x -> pipeline_lens(pipe, x), collect(keys(estval))), length(vals))
    return lenses, vals
end
function pipeline_lens_val_grid(pipe::Pipeline,
                                estvals::AbstractVector{<:Union{<:AbstractVector{<:Pair{<:Any,
                                                                                        <:AbstractVector}},
                                                                <:AbstractDict{<:Any,
                                                                               <:AbstractVector}}})
    lenses_vals = [pipeline_lens_val_grid(pipe, estval) for estval in estvals]
    lenses = mapreduce(x -> x[1], vcat, lenses_vals)
    vals = mapreduce(x -> x[2], vcat, lenses_vals)
    assert_search_grid_cap(length(vals),
                           "the sum of the $(length(estvals)) concatenated parameter sets")
    return lenses, vals
end
"""
    assert_search_candidates(pipe::Pipeline, lens_grid, val_grid) -> Nothing

Refuse a Pipeline search in which a candidate holds a [`TrainTestSplit`](@ref), before the search scores any candidate.

A lens can write a `TrainTestSplit` into a step, so the function checks each candidate and not `pipe` alone. It also checks `pipe` when the grid leaves the split in place. The check runs before the candidates run in parallel, so the caller gets the `ArgumentError` and not a `TaskFailedException` that holds it.

# Algorithm

For each grid point, build the candidate through [`search_candidate`](@ref), and check it through [`assert_no_holdout`](@ref).

# Arguments

  - `pipe`: The pipeline that the search tunes.
  - `lens_grid`: The lenses of each grid point.
  - `val_grid`: The values of each grid point.

# Validation

  - No candidate holds a `TrainTestSplit`, through [`assert_no_holdout`](@ref).

# Returns

  - `nothing`.

# Related

  - [`assert_no_holdout`](@ref)
  - [`search_candidate`](@ref)
  - [`search_cross_validation`](@ref)
"""
function assert_search_candidates(pipe::Pipeline, lens_grid, val_grid)::Nothing
    for (lenses, vals) in zip(lens_grid, val_grid)
        assert_no_holdout(search_candidate(pipe, lenses, vals))
    end
    return nothing
end
"""
    search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation, data::Prices_RR)
    search_cross_validation(pipe::Pipeline, rscv::RandomisedSearchCrossValidation,
                            data::Prices_RR)

Score every candidate Pipeline of the grid on every fold of the scheme, and return the winner.

Each candidate runs through [`cross_val_predict`](@ref), the fold loop of every cross-validation entry point. Every fold fits the whole workflow on its training window and predicts its test window, so a preprocessing step learns from the training window alone. A walk-forward threads the weights of the previous fold through the `pws` of the scheme. A [`TimeDependent`](@ref) schedule resolves per fold, and a key that names its step replaces the whole schedule.

Under an Online Scheme, every candidate warms up once on the first training window, and [`partial_fit!`](@ref) folds each later block of rows into it. The search then gives the score matrix of the batch search over the expanding walk-forward with the same folds, to the tolerance of the estimators and of the solver. The two searches select the same candidate. The randomised form samples a grid from `rscv.p` and runs the grid search on it.

# Mathematical definition

```math
\\begin{align}
S_{fi} &= s \\, \\mathcal{R}\\left(\\hat{P}_{f}(\\theta_i)\\right)\\,, \\quad f = 1, \\ldots, F\\,, \\quad \\theta_i \\in \\Theta\\,, \\\\
\\mathcal{C} &= \\left\\{ i : S_{fi} \\text{ is finite for every } f \\right\\}\\,, \\\\
i^{\\star} &= c_{\\sigma\\left(\\mathbf{S}_{:,\\,\\mathcal{C}}\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:S_fi_search])
  - $(math_dict[:P_f_search])
  - $(math_dict[:F_folds_search])
  - $(math_dict[:theta_i_cand])
  - $(math_dict[:Theta_grid])
  - $(math_dict[:s_orient_search])
  - $(math_dict[:R_search])
  - $(math_dict[:C_finite_cand])
  - $(math_dict[:S_C_search])
  - $(math_dict[:c_j_search])
  - $(math_dict[:sigma_scorer])
  - $(math_dict[:i_star_cand])

The scorer never reads the column of a candidate that failed a fold, so the search never selects that candidate.

# Algorithm

For `gscv`:

 1. Refuse a pipeline that carries a partial-fit state, through [`assert_search_entry`](@ref).
 2. Build `lens_grid` and `val_grid` through [`pipeline_lens_val_grid`](@ref).
 3. Refuse a candidate that holds a [`TrainTestSplit`](@ref), through [`assert_search_candidates`](@ref).
 4. Fix the folds through [`pin_draw`](@ref), giving `scheme`. A scheme whose `split` draws at random then scores every candidate on the same folds.
 5. Split `data` by `scheme`, giving `cv`.
 6. Read the rows of the folds through [`score_rows`](@ref), giving `rows`.
 7. Set `sgn` to ``s``.
 8. Allocate `test_scores`, of size `M × N` for `M` folds and `N` candidates, with the element type from [`cv_data_eltype`](@ref). When `gscv.train_score` is `true`, allocate `train_scores` of the same size.
 9. For each candidate `i`, in parallel over `gscv.ex`, do steps 10 to 12.
10. Build `pipei` through [`search_candidate`](@ref).
11. Run `pipei` over `scheme` through [`cross_val_predict`](@ref), with the folds in sequence, giving `predictions`.
12. Write column `i` of `test_scores` and `train_scores` through [`write_candidate_scores!`](@ref), with no view of the training returns.
13. Select `opt_idx`, the position ``i^{\\star}``, through [`finite_candidate_index`](@ref).
14. Build the selected candidate `pipe` through [`search_candidate`](@ref).
15. Return a [`SearchCrossValidationResult`](@ref).

For `rscv`:

 1. Resolve the random number generator from `rscv.rng` and `rscv.seed` through [`resolve_rng`](@ref).
 2. Sample the grid through [`make_p_grid`](@ref), with at most `rscv.n_iter` values per key.
 3. Build a [`GridSearchCrossValidation`](@ref) with that grid and the other fields of `rscv`, and run the search of that grid. Under a [`CombinatorialCrossValidation`](@ref) this is the combinatorial method.

# Arguments

  - `pipe`: The pipeline to tune. The search reads it as configuration alone.
  - `gscv`: The grid search. It gives the grid `p`, the scheme `cv`, the risk measure `r`, the `scorer`, the executor `ex`, the `train_score` flag and the keyword arguments `kwargs` of [`expected_risk`](@ref).
  - `rscv`: The randomised search. It gives the fields of `gscv`, the distributions or vectors `p` to sample, the sample count `n_iter`, and `rng` and `seed`.
  - `data`: The price or returns data, a [`Prices_RR`](@ref), at the input level of `pipe`.

# Validation

  - `pipe` carries no partial-fit state, through [`assert_search_entry`](@ref). Under an Online Scheme, `pipe` also passes every check of [`assert_online_entry`](@ref).
  - No candidate holds a [`TrainTestSplit`](@ref), through [`assert_search_candidates`](@ref).
  - The grid is not empty, it holds at most `RESOURCE_LIMITS[].max_search_grid` candidates, and every key names a step or a path rooted at `steps`, through [`pipeline_lens_val_grid`](@ref).
  - A [`TimeDependent`](@ref) schedule of a candidate holds one entry per fold. The fold loop checks each candidate as it runs it.
  - At least one candidate finishes every fold. Otherwise [`finite_candidate_index`](@ref) throws an `IsNonFiniteError`.

# Returns

  - `res::SearchCrossValidationResult`: The field `test_scores` is ``\\mathbf{S}``, the raw matrix, with `NaN` at a failed fold. Row ``f`` is fold ``f`` in the order of `split`, and under a [`MultipleRandomised`](@ref), [`score_rows`](@ref) puts the scores of each path back onto the rows of the split. The field `train_scores` is `nothing` when `gscv.train_score` is `false`. Otherwise it holds ``s \\, \\mathcal{R}`` of each fitted result through its prior result, in a matrix of the same size. The field `opt` is candidate ``i^{\\star}``, and `idx` is ``i^{\\star}``.

# Related

  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`pipeline_lens`](@ref)
  - [`cross_val_predict`](@ref)
  - [`write_candidate_scores!`](@ref)
  - [`candidate_train_score`](@ref): the train score reads the prior result, because the steps of a Pipeline change the data of each fold.
  - [`score_rows`](@ref)
  - [`pin_draw`](@ref)
  - [`finite_candidate_index`](@ref)
  - [`assert_search_entry`](@ref)
  - [`Prices_RR`](@ref)
"""
function search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation,
                                 data::Prices_RR)
    assert_search_entry(pipe, gscv.cv)
    lens_grid, val_grid = pipeline_lens_val_grid(pipe, gscv.p)
    assert_search_candidates(pipe, lens_grid, val_grid)
    scheme = pin_draw(gscv.cv)
    cv = split(scheme, data)
    rows = score_rows(cv)
    N = length(val_grid)
    M = length(cv.train_idx)
    r = gscv.r
    sgn = ifelse(bigger_is_better(r), 1, -1)
    test_scores = Matrix{cv_data_eltype(data)}(undef, M, N)
    train_scores = if gscv.train_score
        Matrix{cv_data_eltype(data)}(undef, M, N)
    else
        nothing
    end
    let pipe = pipe, test_scores = test_scores, train_scores = train_scores
        FLoops.@floop gscv.ex for (i, (lenses, vals)) in
                                  enumerate(zip(lens_grid, val_grid))
            local pipei = search_candidate(pipe, lenses, vals)
            # Candidates run in parallel over `gscv.ex`; the folds inside one run in
            # sequence, through the same loop every other entry point runs.
            local predictions = cross_val_predict(pipei, data, scheme;
                                                  ex = FLoops.SequentialEx())
            # The window is `nothing` here: a Pipeline's steps transform the data per
            # fold, so the returns the optimiser was handed are not the returns the split
            # names. See [`candidate_train_score`](@ref), whose `Nothing` arm scores
            # through the prior result.
            write_candidate_scores!(test_scores, train_scores, i, predictions, rows,
                                    nothing, r, sgn, gscv.kwargs)
        end
    end
    opt_idx = finite_candidate_index(gscv.scorer, test_scores)
    pipe = search_candidate(pipe, lens_grid[opt_idx], val_grid[opt_idx])
    return SearchCrossValidationResult(; opt = pipe, test_scores = test_scores,
                                       train_scores = train_scores, lens_grid = lens_grid,
                                       val_grid = val_grid, idx = opt_idx)
end
"""
    search_cross_validation(pipe::Pipeline,
                            gscv::GridSearchCrossValidation{<:Any, <:CombinatorialCrossValidation},
                            data::Prices_RR)

Tune a [`Pipeline`](@ref) over the grid of `gscv` under a [`CombinatorialCrossValidation`](@ref), with one score per candidate and backtest path.

A combinatorial scheme recombines its disjoint test groups into full backtest paths, and a split scored alone mixes the groups of different paths. So this method scores each path. For each candidate, [`cross_val_predict`](@ref) runs the scheme, [`sort_predictions`](@ref) recombines the groups into a [`PopulationPredictionResult`](@ref), and [`expected_risk`](@ref) gives one score per path. The candidates run in sequence, and the fold loop of each candidate runs with the executor `gscv.ex`.

The method takes price data too. The training rows of a split then have gaps where the test groups are, so [`PricesToReturns`](@ref) makes one return across each gap. The test groups are contiguous, so the predictions have no such return. A [`MultipleRandomised`](@ref) scheme keeps the training rows contiguous at the price level.

# Mathematical definition

```math
\\begin{align}
S_{pi} &= s \\, \\mathcal{R}\\left(\\hat{P}_{p}(\\theta_i)\\right)\\,, \\quad p = 1, \\ldots, n_{p}\\,, \\quad \\theta_i \\in \\Theta\\,, \\\\
\\mathcal{C} &= \\left\\{ i : S_{pi} \\text{ is finite for every } p \\right\\}\\,, \\\\
i^{\\star} &= c_{\\sigma\\left(\\mathbf{S}_{:,\\,\\mathcal{C}}\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:S_pi_search])
  - $(math_dict[:P_p_search])
  - $(math_dict[:n_p_search])
  - $(math_dict[:theta_i_cand])
  - $(math_dict[:Theta_grid])
  - $(math_dict[:s_orient_search])
  - $(math_dict[:R_search])
  - $(math_dict[:C_finite_cand])
  - $(math_dict[:S_C_search])
  - $(math_dict[:c_j_search])
  - $(math_dict[:sigma_scorer])
  - $(math_dict[:i_star_cand])

The scorer never reads the column of a candidate that failed a path, so the search never selects that candidate.

# Algorithm

 1. Refuse a pipeline that carries a partial-fit state, through [`assert_search_entry`](@ref).
 2. Build `lens_grid` and `val_grid` through [`pipeline_lens_val_grid`](@ref).
 3. Refuse a candidate that holds a [`TrainTestSplit`](@ref), through [`assert_search_candidates`](@ref).
 4. Split `data` by `gscv.cv`, giving `cv`.
 5. Set `M`, the count of the paths, to the greatest entry of `cv.path_ids`.
 6. Set `sgn` to ``s``.
 7. Allocate `test_scores`, of size `M × N` for `N` candidates, with the element type from [`cv_data_eltype`](@ref). When `gscv.train_score` is `true`, allocate `train_scores`, one matrix per path, with one row per fold of the path and one column per candidate.
 8. For each candidate `i`, in sequence, do steps 9 to 12.
 9. Build `pipei` through [`search_candidate`](@ref).
10. Run `pipei` over `gscv.cv` through [`cross_val_predict`](@ref), with the executor `gscv.ex`, giving `predictions`, one path per member.
11. Write ``s \\, \\mathcal{R}`` of each path of `predictions` into column `i` of `test_scores`.
12. When `gscv.train_score` is `true`, write the train score of each fold of each path into column `i` of the matrix of that path, through [`candidate_train_score`](@ref) with no view of the training returns.
13. Select `opt_idx`, the position ``i^{\\star}``, through [`finite_candidate_index`](@ref).
14. Build the selected candidate `pipe` through [`search_candidate`](@ref).
15. Return a [`SearchCrossValidationResult`](@ref).

# Arguments

  - `pipe`: The pipeline to tune. The search reads it as configuration alone.
  - `gscv`: The grid search over a [`CombinatorialCrossValidation`](@ref). It gives the grid `p`, the scheme `cv`, the risk measure `r`, the `scorer`, the executor `ex`, the `train_score` flag and the keyword arguments `kwargs` of [`expected_risk`](@ref).
  - `data`: The price or returns data, a [`Prices_RR`](@ref), at the input level of `pipe`.

# Validation

  - `pipe` carries no partial-fit state, through [`assert_search_entry`](@ref).
  - No candidate holds a [`TrainTestSplit`](@ref), through [`assert_search_candidates`](@ref).
  - The grid is not empty, it holds at most `RESOURCE_LIMITS[].max_search_grid` candidates, and every key names a step or a path rooted at `steps`, through [`pipeline_lens_val_grid`](@ref).
  - At least one candidate finishes every path. Otherwise [`finite_candidate_index`](@ref) throws an `IsNonFiniteError`.

# Returns

  - `res::SearchCrossValidationResult`: The field `test_scores` is ``\\mathbf{S}``, the raw matrix, with `NaN` at a failed path. The field `train_scores` is `nothing` when `gscv.train_score` is `false`. Otherwise it is a vector of ``n_{p}`` matrices, one per path, and each matrix holds one row per fold of the path and one column per candidate. Each entry is ``s \\, \\mathcal{R}`` of the fitted result of the fold, through its prior result. The folds of one path train on different windows, so the train scores stay per fold. The field `opt` is candidate ``i^{\\star}``, and `idx` is ``i^{\\star}``.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref): its search samples a grid and runs this method on it under a combinatorial scheme.
  - [`cross_val_predict`](@ref)
  - [`sort_predictions`](@ref)
  - [`expected_risk`](@ref)
  - [`finite_candidate_index`](@ref)
  - [`candidate_train_score`](@ref)
"""
function search_cross_validation(pipe::Pipeline,
                                 gscv::GridSearchCrossValidation{<:Any,
                                                                 <:CombinatorialCrossValidation},
                                 data::Prices_RR)
    assert_search_entry(pipe, gscv.cv)
    lens_grid, val_grid = pipeline_lens_val_grid(pipe, gscv.p)
    assert_search_candidates(pipe, lens_grid, val_grid)
    cv = split(gscv.cv, data)
    N = length(val_grid)
    M = maximum(cv.path_ids)          # one score per recombined backtest path
    r = gscv.r
    sgn = ifelse(bigger_is_better(r), 1, -1)
    test_scores = Matrix{cv_data_eltype(data)}(undef, M, N)
    # Train scores are per fold, and each path holds a different number of folds, so they
    # are kept as one `folds × candidates` matrix per path (a Vector of matrices) rather
    # than collapsed. Test scores stay one per path.
    train_scores = if gscv.train_score
        [Matrix{cv_data_eltype(data)}(undef, count(==(p), cv.path_ids), N) for p in 1:M]
    else
        nothing
    end
    for (i, (lenses, vals)) in enumerate(zip(lens_grid, val_grid))
        pipei = search_candidate(pipe, lenses, vals)
        # cross_val_predict fits every split and recombines groups into paths (handling any
        # time-dependent schedules); fold-level parallelism lives inside it.
        predictions = cross_val_predict(pipei, data, gscv.cv; ex = gscv.ex)
        test_scores[:, i] = sgn * expected_risk(r, predictions; gscv.kwargs...)
        if gscv.train_score
            for (p, path) in enumerate(predictions.pred)
                for (j, fp) in enumerate(path.pred)
                    train_scores[p][j, i] = sgn * candidate_train_score(r, fp.res, nothing,
                                                                        gscv.kwargs)
                end
            end
        end
    end
    opt_idx = finite_candidate_index(gscv.scorer, test_scores)
    pipe = search_candidate(pipe, lens_grid[opt_idx], val_grid[opt_idx])
    return SearchCrossValidationResult(; opt = pipe, test_scores = test_scores,
                                       train_scores = train_scores, lens_grid = lens_grid,
                                       val_grid = val_grid, idx = opt_idx)
end
function search_cross_validation(pipe::Pipeline, rscv::RandomisedSearchCrossValidation,
                                 data::Prices_RR)
    rng = resolve_rng(rscv.rng, rscv.seed)
    return search_cross_validation(pipe,
                                   GridSearchCrossValidation(make_p_grid(rscv.p,
                                                                         rscv.n_iter, rng);
                                                             cv = rscv.cv, r = rscv.r,
                                                             scorer = rscv.scorer,
                                                             ex = rscv.ex,
                                                             train_score = rscv.train_score,
                                                             kwargs = rscv.kwargs), data)
end
