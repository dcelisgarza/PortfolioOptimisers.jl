"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the observation-window view of price- or returns-level data used by pipeline search cross-validation.

# Arguments

  - `data`: The input data ([`AbstractPricesResult`](@ref) or [`AbstractReturnsResult`](@ref)).
  - `idx`: Observation window into the rows of `data`.

# Returns

  - `data′`: The windowed data at the same level.

# Related

  - [`port_opt_view`](@ref)
  - [`port_opt_view`](@ref)
  - [`fit_and_score`](@ref)
"""
pipeline_data_view(pr::AbstractPricesResult, idx, idx2 = :) = port_opt_view(pr, idx, idx2)
pipeline_data_view(rd::AbstractReturnsResult, idx, idx2 = :) = port_opt_view(rd, idx, idx2)
"""
    pipeline_asset_view(data::AbstractReturnsResult, cols)
    pipeline_asset_view(data::AbstractPricesResult, cols)

Return the asset-subset view of price- or returns-level `data` for a [`MultipleRandomised`](@ref)
resampling path — all observations, only the columns `cols`.

The two levels index assets through different `port_opt_view` arities: returns take the
two-argument asset form `port_opt_view(rd, cols)`, prices the observation-then-asset form
`port_opt_view(pr, :, cols)`. This wrapper hides that asymmetry so
[`pipeline_path_fit_and_predict`](@ref) stays level-agnostic.

# Related

  - [`pipeline_data_view`](@ref)
  - [`pipeline_path_fit_and_predict`](@ref)
"""
pipeline_asset_view(data::AbstractReturnsResult, cols) = port_opt_view(data, cols)
pipeline_asset_view(data::AbstractPricesResult, cols) = port_opt_view(data, :, cols)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the element type search-CV score matrices use for the given data level.

# Related

  - [`search_cross_validation`](@ref)
"""
cv_data_eltype(rd::AbstractReturnsResult) = eltype(rd.X)
cv_data_eltype(pr::AbstractPricesResult) = eltype(TimeSeries.values(pr.X))
"""
    pipeline_lens(pipe::Pipeline, key) -> lens

Resolve a tuning key into an Accessors.jl lens on a [`Pipeline`](@ref).

A leading step name resolves to the step's position (name → index → property path): `"impute.stat"` targets the `stat` field of the step named `"impute"`, and a bare step name (`"impute"`, `:impute`) or an integer position targets the whole step — swapping entire estimators as grid values needs no extra syntax. Keys whose leading segment is not a step name fall through to [`parse_lens`](@ref), so raw property paths (`"steps[2].stat"`) and prebuilt lenses keep working.

# Arguments

  - `pipe`: The pipeline being tuned.
  - `key`: A [`GSCVKey`](@ref): step name with optional trailing property path, integer step position, raw property path, `Expr`/`Symbol`, or a prebuilt lens.

# Returns

  - `lens`: A composed Accessors.jl lens rooted at the pipeline.

# Related

  - [`parse_lens`](@ref)
  - [`Pipeline`](@ref)
  - [`search_cross_validation`](@ref)
"""
function pipeline_lens(pipe::Pipeline, key::AbstractString)
    parts = split(key, '.'; limit = 2)
    i = findfirst(==(parts[1]), pipe.names)
    if isnothing(i)
        return parse_lens(key)
    end
    step = Accessors.IndexLens((i,)) ∘ Accessors.PropertyLens(:steps)
    return length(parts) == 1 ? step : parse_lens(parts[2]) ∘ step
end
function pipeline_lens(pipe::Pipeline, key::Symbol)
    ks = string(key)
    i = findfirst(==(ks), pipe.names)
    if isnothing(i)
        # A bare (undotted) symbol that misses the step-name table is a typo, not a
        # lens path — fail closed rather than silently reinterpreting it as a property
        # access on the pipeline struct. Genuinely dotted symbols still fall through to
        # `parse_lens`, which is structurally capped.
        @argcheck(occursin('.', ks),
                  ArgumentError("`$(key)` is not a step name among the $(length(pipe.names)) named pipeline steps" *
                                did_you_mean(ks, pipe.names)))
        return parse_lens(key)
    end
    return Accessors.IndexLens((i,)) ∘ Accessors.PropertyLens(:steps)
end
function pipeline_lens(pipe::Pipeline, key::Integer)
    @argcheck(1 <= key <= length(pipe.steps),
              ArgumentError("step position $key is out of bounds for a pipeline with $(length(pipe.steps)) steps"))
    return Accessors.IndexLens((Int(key),)) ∘ Accessors.PropertyLens(:steps)
end
function pipeline_lens(::Pipeline, key)
    return parse_lens(key)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the (lens, value) grid for tuning a [`Pipeline`](@ref) — the pipeline-aware counterpart of [`lens_val_grid`](@ref), resolving keys through [`pipeline_lens`](@ref) so step names and positions address steps.

# Arguments

  - `pipe`: The pipeline being tuned.
  - `estval`: The parameter grid: `key => values` pairs, a dict, or a vector of either (independent grids concatenated).

# Returns

  - `(lenses, vals)`: Per-candidate lens vectors and value tuples.

# Related

  - [`pipeline_lens`](@ref)
  - [`search_cross_validation`](@ref)
"""
function pipeline_lens_val_grid(pipe::Pipeline,
                                estval::AbstractVector{<:Pair{<:Any, <:AbstractVector}})
    vals = vec(collect(Iterators.product(map(x -> x[2], estval)...)))
    lenses = fill(map(x -> pipeline_lens(pipe, x[1]), estval), length(vals))
    return lenses, vals
end
function pipeline_lens_val_grid(pipe::Pipeline,
                                estval::AbstractDict{<:Any, <:AbstractVector})
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
    return lenses, vals
end
"""
    fit_and_score(pipe::Pipeline,
                       scv::Union{<:GridSearchCrossValidation{<:Any, <:Any},
                                  <:RandomisedSearchCrossValidation{<:Any, <:Any}},
                       cv::CrossValidationResult, rd::Prices_RR, i::Integer)

Fit a [`Pipeline`](@ref) on the training window and score it on the test window for search cross-validation.

The whole workflow is fitted per fold: stateful preprocessing (universe, imputation parameters) is learned on the training window only, and [`predict`](@ref) replays it on the test window before the score is computed — no test information leaks into the preprocessing, which is the point of the pipeline (ADR 0028).

# Arguments

  - `pipe`: The pipeline candidate.
  - `scv`: The search cross-validation estimator carrying the risk measure, options, and train-score flag.
  - `data`: Price- or returns-level input data.
  - `train_idx`: Observation indices of the training window.
  - `test_idx`: Observation indices of the test window.

# Returns

  - `(test_score, train_score)`: Signed scores; `train_score` is `nothing` unless requested.

# Related

  - [`search_cross_validation`](@ref)
  - [`expected_risk`](@ref)
  - [`predict(res::PipelineResult, data::AbstractPricesResult, window)`](@ref)
"""
function fit_and_score(pipe::Pipeline,
                       scv::Union{<:GridSearchCrossValidation{<:Any, <:Any},
                                  <:RandomisedSearchCrossValidation{<:Any, <:Any}},
                       cv::CrossValidationResult, rd::Prices_RR, i::Integer)
    assert_no_holdout(pipe)
    prediction = fit_and_predict(pipe, rd; train_idx = cv.train_idx[i],
                                 test_idx = cv.test_idx[i])
    r = scv.r
    sign = ifelse(bigger_is_better(r), 1, -1)
    test_score = sign * expected_risk(r, prediction; scv.kwargs...)
    train_score = if scv.train_score
        sign * expected_risk(r, prediction.res; scv.kwargs...)
    else
        nothing
    end
    return test_score, train_score
end
function fit_and_score(pipe::Pipeline,
                       scv::Union{<:GridSearchCrossValidation{<:Any, <:MultipleRandomised},
                                  <:RandomisedSearchCrossValidation{<:Any,
                                                                    <:MultipleRandomised}},
                       cv::MultipleRandomisedResult, rd::Prices_RR, i::Integer)
    assert_no_holdout(pipe)
    prediction = fit_and_predict(pipe, rd; train_idx = cv.train_idx[i],
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
    search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation, data::Prices_RR)
    search_cross_validation(pipe::Pipeline, rscv::RandomisedSearchCrossValidation, data::Prices_RR)

Tune a [`Pipeline`](@ref) by grid (or randomised) search cross-validation on price- or returns-level input data.

The input is split into contiguous observation windows by `gscv.cv` (price-level splits keep stateful preprocessing inside the fold); for each candidate the lens grid is applied to the pipeline (keys resolved by [`pipeline_lens`](@ref), so step names, step positions, and raw property paths all address steps), the whole workflow is fitted on the training window and scored on the test window via [`fit_and_score`](@ref), and the scorer picks the winner. The randomised form samples the grid and delegates, exactly as for plain optimisers.

[`TimeDependent`](@ref) schedules resolve against the *tuning* folds: when a candidate is time-dependent, its schedules are sized to the tuning scheme's fold count (asserted per candidate — a grid value may swap a whole schedule in or out), and tuning fold `j` swaps in entry `j` via the pipeline-level [`update_time_dependent_estimator`](@ref) before [`fit_and_score`](@ref) runs. Lenses need no schedule-specific semantics: naming the step swaps the whole schedule as a grid value, and raw property paths address entries.

# Arguments

  - `pipe`: The pipeline to tune.
  - `gscv`/`rscv`: The search cross-validation estimator.
  - `data`: Price- or returns-level input data ([`Prices_RR`](@ref)).

# Returns

  - `res::SearchCrossValidationResult`: The tuned pipeline (`res.opt`), score matrices, lens/value grids, and selected index.

# Related

  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`pipeline_lens`](@ref)
  - [`fit_and_score`](@ref)
"""
function search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation,
                                 data::Prices_RR)
    assert_no_holdout(pipe)
    lens_grid, val_grid = pipeline_lens_val_grid(pipe, gscv.p)
    cv = split(gscv.cv, data)
    N = length(val_grid)
    M = length(cv.train_idx)
    test_scores = Matrix{cv_data_eltype(data)}(undef, M, N)
    train_scores = if gscv.train_score
        Matrix{cv_data_eltype(data)}(undef, M, N)
    else
        nothing
    end
    let pipe = pipe, test_scores = test_scores, train_scores = train_scores
        FLoops.@floop gscv.ex for (i, (lenses, vals)) in
                                  enumerate(zip(lens_grid, val_grid))
            local pipei = pipe
            for (lens, val) in zip(lenses, vals)
                pipei = Accessors.set(pipei, lens, val)
            end
            local td_flag = is_time_dependent(pipei)
            if td_flag
                assert_time_dependent_fold_count(pipei, M)
            end
            for j in eachindex(cv.train_idx)
                local pipej = pipei
                if td_flag
                    pipej = update_time_dependent_estimator(pipei,
                                                            TimeDependentContext(; i = j,
                                                                                 n = M,
                                                                                 rd = data,
                                                                                 train_idx = cv.train_idx,
                                                                                 test_idx = cv.test_idx))
                end
                test_score, train_score = fit_and_score(pipej, gscv, cv, data, j)
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
        pipe = Accessors.set(pipe, lens, val)
    end
    return SearchCrossValidationResult(; opt = pipe, test_scores = test_scores,
                                       train_scores = train_scores, lens_grid = lens_grid,
                                       val_grid = val_grid, idx = opt_idx)
end
"""
    search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation{<:Any, <:CombinatorialCrossValidation}, data::AbstractReturnsResult)

Grid search cross-validation of a [`Pipeline`](@ref) over a [`CombinatorialCrossValidation`](@ref) scheme.

Combinatorial recombines its disjoint test groups into full-length backtest **paths**, so — like the plain-optimiser combinatorial method — scoring is per-path, not per-split: scoring a split in isolation would mix groups belonging to different paths. For each candidate the whole workflow runs through [`cross_val_predict`](@ref) (splits fitted, groups recombined by [`sort_predictions!`](@ref) into a [`PopulationPredictionResult`](@ref)), and [`expected_risk`](@ref) yields one score per path; the score matrix is therefore `n_paths × n_candidates` and the scorer selects across candidates as usual.

`train_scores` (only when `gscv.train_score`) keeps every per-fold in-sample score: a `Vector` of `n_paths` matrices, one per path, each `folds_in_path × n_candidates` (test scores stay one-per-path because a path's out-of-sample returns pool into one series, while its folds train on distinct in-sample windows).

Combinatorial runs at both levels here. At the **price level** a split's training rows are non-contiguous (gaps where the held-out groups sit), so the fold's [`PricesToReturns`](@ref) produces one spurious return per gap boundary — an accepted approximation in exchange for the combinatorial paths (see [`cross_val_predict(pipe::Pipeline, data::Prices_RR, cv::CombinatorialCrossValidation)`](@ref)). The randomised form delegates here through its grid.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`cross_val_predict`](@ref)
  - [`expected_risk`](@ref)
  - [`search_cross_validation`](@ref)
"""
function search_cross_validation(pipe::Pipeline,
                                 gscv::GridSearchCrossValidation{<:Any,
                                                                 <:CombinatorialCrossValidation},
                                 data::Prices_RR)
    assert_no_holdout(pipe)
    lens_grid, val_grid = pipeline_lens_val_grid(pipe, gscv.p)
    cv = split(gscv.cv, data)
    N = length(val_grid)
    M = maximum(cv.path_ids)          # one score per recombined backtest path
    r = gscv.r
    sgn = ifelse(bigger_is_better(r), 1, -1)
    test_scores = Matrix{cv_data_eltype(data)}(undef, M, N)
    # Train scores are per fold, and each path holds a different number of folds, so they
    # are kept as one `folds × candidates` matrix per path (a Vector of matrices) rather
    # than collapsed — test scores stay one-per-path.
    train_scores = if gscv.train_score
        [Matrix{cv_data_eltype(data)}(undef, count(==(p), cv.path_ids), N) for p in 1:M]
    else
        nothing
    end
    for (i, (lenses, vals)) in enumerate(zip(lens_grid, val_grid))
        pipei = pipe
        for (lens, val) in zip(lenses, vals)
            pipei = Accessors.set(pipei, lens, val)
        end
        # cross_val_predict fits every split and recombines groups into paths (handling any
        # time-dependent schedules); fold-level parallelism lives inside it.
        predictions = cross_val_predict(pipei, data, gscv.cv; ex = gscv.ex)
        test_scores[:, i] = sgn * expected_risk(r, predictions; gscv.kwargs...)
        if gscv.train_score
            for (p, path) in enumerate(predictions.pred)
                for (j, fp) in enumerate(path.pred)
                    train_scores[p][j, i] = sgn * expected_risk(r, fp.res; gscv.kwargs...)
                end
            end
        end
    end
    opt_idx = gscv.scorer(test_scores)
    for (lens, val) in zip(lens_grid[opt_idx], val_grid[opt_idx])
        pipe = Accessors.set(pipe, lens, val)
    end
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
