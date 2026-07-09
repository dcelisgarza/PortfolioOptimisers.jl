"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the observation-window view of price- or returns-level data used by pipeline search cross-validation.

# Arguments

  - `data`: The input data ([`AbstractPricesResult`](@ref) or [`AbstractReturnsResult`](@ref)).
  - `idx`: Observation window into the rows of `data`.

# Returns

  - `data′`: The windowed data at the same level.

# Related

  - [`prices_view`](@ref)
  - [`returns_result_view`](@ref)
  - [`fit_and_score`](@ref)
"""
pipeline_data_view(pr::AbstractPricesResult, idx) = prices_view(pr, idx)
pipeline_data_view(rd::AbstractReturnsResult, idx) = returns_result_view(rd, idx, :)
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
    i = findfirst(==(string(key)), pipe.names)
    if isnothing(i)
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
    fit_and_score(pipe::Pipeline, scv::AbstractSearchCrossValidationEstimator, data, train_idx::VecInt, test_idx::VecInt)

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
function fit_and_score(pipe::Pipeline, scv::AbstractSearchCrossValidationEstimator,
                       data::Rd_Pr, train_idx::VecInt, test_idx::VecInt)
    res = StatsAPI.fit(pipe, pipeline_data_view(data, train_idx))
    test_pred = StatsAPI.predict(res, data, test_idx)
    r = scv.r
    sign = ifelse(bigger_is_better(r), 1, -1)
    test_score = sign * expected_risk(r, test_pred; scv.kwargs...)
    train_score = if scv.train_score
        sign * expected_risk(r, res.ctx.opt; scv.kwargs...)
    else
        nothing
    end
    return test_score, train_score
end
"""
    search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation, data::Rd_Pr)
    search_cross_validation(pipe::Pipeline, rscv::RandomisedSearchCrossValidation, data::Rd_Pr)

Tune a [`Pipeline`](@ref) by grid (or randomised) search cross-validation on price- or returns-level input data.

The input is split into contiguous observation windows by `gscv.cv` (price-level splits keep stateful preprocessing inside the fold); for each candidate the lens grid is applied to the pipeline (keys resolved by [`pipeline_lens`](@ref), so step names, step positions, and raw property paths all address steps), the whole workflow is fitted on the training window and scored on the test window via [`fit_and_score`](@ref), and the scorer picks the winner. The randomised form samples the grid and delegates, exactly as for plain optimisers.

# Arguments

  - `pipe`: The pipeline to tune.
  - `gscv`/`rscv`: The search cross-validation estimator.
  - `data`: Price- or returns-level input data ([`Rd_Pr`](@ref)).

# Returns

  - `res::SearchCrossValidationResult`: The tuned pipeline (`res.opt`), score matrices, lens/value grids, and selected index.

# Related

  - [`GridSearchCrossValidation`](@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`pipeline_lens`](@ref)
  - [`fit_and_score`](@ref)
"""
function search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation,
                                 data::Rd_Pr)
    lens_grid, val_grid = pipeline_lens_val_grid(pipe, gscv.p)
    cv = split(gscv.cv, data)
    @argcheck(isa(cv.test_idx[1], VecInt),
              ArgumentError("grid search cross-validation requires non-combinatorial (VecInt) test indices, but got $(typeof(cv.test_idx[1]))"))
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
            for (j, (train_idx, test_idx)) in enumerate(zip(cv.train_idx, cv.test_idx))
                test_score, train_score = fit_and_score(pipei, gscv, data, train_idx,
                                                        test_idx)
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
function search_cross_validation(pipe::Pipeline, rscv::RandomisedSearchCrossValidation,
                                 data::Rd_Pr)
    if !isnothing(rscv.seed)
        Random.seed!(rscv.rng, rscv.seed)
    end
    return search_cross_validation(pipe,
                                   GridSearchCrossValidation(make_p_grid(rscv.p,
                                                                         rscv.n_iter,
                                                                         rscv.rng);
                                                             cv = rscv.cv, r = rscv.r,
                                                             scorer = rscv.scorer,
                                                             ex = rscv.ex,
                                                             train_score = rscv.train_score,
                                                             kwargs = rscv.kwargs), data)
end
