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
  - [`cross_val_predict`](@ref)
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
    is_pipeline_raw_path(key::AbstractString) -> Bool

Return `true` when `key` is a raw property path rooted at the `steps` field of a [`Pipeline`](@ref) — `"steps[1]"`, `"steps[2].fill"`.

The predicate the `AbstractString` arm of [`pipeline_lens`](@ref) uses to separate a *raw property path* from a *typo* when the leading segment misses the step-name table. It names the one root a path may address instead of rejecting one shape. A `Pipeline` holds two fields, `names` and `steps`, and only `steps` holds the estimators a grid tunes, so `"names[1]"` is refused with `"gapfill"`. An admitted key falls through to [`parse_lens`](@ref).

The `Symbol` arm tests for a dot as well, because a symbol key is not run through `Meta.parse`: an index in a symbol is a character in a property name, not a lens path.

# Related

  - [`pipeline_lens`](@ref)
  - [`parse_lens`](@ref)
"""
function is_pipeline_raw_path(key::AbstractString)
    i = findfirst(c -> c == '.' || c == '[', key)
    return !isnothing(i) && SubString(key, firstindex(key), prevind(key, i)) == "steps"
end
"""
    pipeline_lens(pipe::Pipeline, key) -> lens

Resolve a tuning key into an Accessors.jl lens on a [`Pipeline`](@ref).

A leading step name resolves to the step's position (name → index → property path): `"gap_fill.fill"` targets the `fill` field of the step named `"gap_fill"`, and a bare step name (`"gap_fill"`, `:gap_fill`) or an integer position targets the whole step — swapping entire estimators as grid values needs no extra syntax. A key whose leading segment is not a step name falls through to [`parse_lens`](@ref) only when it is rooted at `steps`, so raw property paths (`"steps[2].fill"`, `"steps[2]"`) and prebuilt lenses keep working.

A key that misses the step-name table and is not a path rooted at `steps` (see [`is_pipeline_raw_path`](@ref)) is rejected instead — `"gapfill"` is a typo, not a path, and `"names[1]"` addresses the step-name table, so reinterpreting either as a property access on the `Pipeline` struct tunes nothing at best and writes into a real field at worst. The `Symbol` arm fails closed on the same rule, and tests for a dot as well, because a symbol key never reaches `Meta.parse`.

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
        # A key that misses the step-name table is a lens path only when it is rooted at
        # `steps`, the one `Pipeline` field that holds tunable estimators — fail closed on
        # every other root rather than silently reinterpreting the key as a property access
        # on the pipeline struct, where a path into the step-name table (`"names[1]"`) is
        # written into on every fold. An admitted path still falls through to `parse_lens`,
        # which is structurally capped. Mirrors the `Symbol` arm below.
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
        # A symbol that misses the step-name table is a lens path only when it is dotted
        # and rooted at `steps` — fail closed rather than silently reinterpreting it as a
        # property access on the pipeline struct. An admitted symbol still falls through to
        # `parse_lens`, which is structurally capped. The dot is tested here and not in
        # `is_pipeline_raw_path` because a `Symbol` is never run through `Meta.parse`: an
        # index in one is a character in a property name, not a lens path.
        @argcheck(occursin('.', ks) && is_pipeline_raw_path(ks),
                  ArgumentError("`$(key)` is not a step name among the $(length(pipe.names)) named pipeline steps, nor a property path rooted at `steps`" *
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

# Validation

  - The candidate count must not exceed `RESOURCE_LIMITS[].max_search_grid`, asserted by [`assert_search_grid_cap`](@ref) before the product is materialised.

# Returns

  - `(lenses, vals)`: Per-candidate lens vectors and value tuples.

# Related

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
    search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation, data::Prices_RR)
    search_cross_validation(pipe::Pipeline, rscv::RandomisedSearchCrossValidation, data::Prices_RR)

Tune a [`Pipeline`](@ref) by grid (or randomised) search cross-validation on price- or returns-level input data.

The input is split into contiguous observation windows by `gscv.cv` (price-level splits keep stateful preprocessing inside the fold); for each candidate the lens grid is applied to the pipeline (keys resolved by [`pipeline_lens`](@ref), so step names, step positions, and raw property paths all address steps), and the candidate is scored through [`cross_val_predict`](@ref)`(pipe_i, data, gscv.cv; ex = SequentialEx())`, the one fold loop every cross-validation entry point runs. So the candidate runs the scheme it declared: every fold fits the whole workflow on its training window and scores it on its test window, a walk-forward threads the previous fold's weights through the scheme's `pws`, and a [`TimeDependent`](@ref) schedule resolves per fold against the fold's [`TimeDependentContext`](@ref), sized to the scheme's fold count and asserted per candidate, because a grid value may swap a whole schedule in or out. Lenses need no schedule-specific semantics: naming the step swaps the whole schedule as a grid value, and raw property paths address entries. Candidates run in parallel over `gscv.ex`, the folds inside one in sequence. One row per fold, in `split`'s order, through [`write_candidate_scores!`](@ref) and [`score_rows`](@ref); a scheme whose `split` draws at random is fixed once through [`pin_draw`](@ref). The scorer picks the winner among the candidates that finished every fold, through [`finite_candidate_index`](@ref), so a candidate that failed a fold never wins (ADR 0120). The randomised form samples the grid and delegates, exactly as for plain optimisers.

A scheme that declares a Fold Fit runs every candidate through the Pipeline's online step (ADR 0142): each candidate is warmed up once and folded fold by fold through [`partial_fit!`](@ref), and read out through `fit(pipe)` where a refit would have run, so the search picks the candidate the batch search picks over the same steps. A warm pipeline is refused once, before the grid, through [`assert_search_entry`](@ref); the refit route `Online(pipe)` is not a search root, because the grid's lenses address the pipeline's steps and not a wrapper's.

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
  - [`cross_val_predict`](@ref)
  - [`write_candidate_scores!`](@ref)
  - [`score_rows`](@ref)
  - [`pin_draw`](@ref)
  - [`finite_candidate_index`](@ref)
  - [`assert_search_entry`](@ref)
"""
function search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation,
                                 data::Prices_RR)
    assert_no_holdout(pipe)
    assert_search_entry(pipe, gscv.cv)
    lens_grid, val_grid = pipeline_lens_val_grid(pipe, gscv.p)
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
            local pipei = pipe
            for (lens, val) in zip(lenses, vals)
                pipei = Accessors.set(pipei, lens, val)
            end
            # Candidates run in parallel over `gscv.ex`; the folds inside one run in
            # sequence, through the same loop every other entry point runs.
            local predictions = cross_val_predict(pipei, data, scheme;
                                                  ex = FLoops.SequentialEx())
            write_candidate_scores!(test_scores, train_scores, i, predictions, rows, r, sgn,
                                    gscv.kwargs)
        end
    end
    opt_idx = finite_candidate_index(gscv.scorer, test_scores)
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

Combinatorial recombines its disjoint test groups into full-length backtest **paths**, so — like the plain-optimiser combinatorial method — scoring is per-path, not per-split: scoring a split in isolation would mix groups belonging to different paths. For each candidate the whole workflow runs through [`cross_val_predict`](@ref) (splits fitted, groups recombined by [`sort_predictions!`](@ref) into a [`PopulationPredictionResult`](@ref)), and [`expected_risk`](@ref) yields one score per path; the score matrix is therefore `n_paths × n_candidates` and the scorer selects across candidates as usual, through [`finite_candidate_index`](@ref), so a candidate that failed a path never wins (ADR 0120).

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
    opt_idx = finite_candidate_index(gscv.scorer, test_scores)
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
