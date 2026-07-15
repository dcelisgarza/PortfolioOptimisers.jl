"""
    make_p_grid(p, n_iter, rng, ...)

Build a randomised parameter grid from a specification.

Samples `n_iter` random parameter combinations from the provided distributions or ranges, for use in randomised search cross-validation.

# Arguments

  - `p`: Parameter specification (pair, vector of pairs, or dictionary).
  - `n_iter`: Number of random parameter combinations to sample.
  - `rng`: Random number generator.
  - Additional arguments.

# Returns

  - Vector of sampled parameter combinations.

# Related

  - [`RandomisedSearchCrossValidation`](@ref)
  - [`parse_lens`](@ref)
"""
function make_p_grid(p::Pair, n_iter::Integer, rng::Random.AbstractRNG,
                     replace::Bool = false)
    vals = if isa(p[2], Distributions.Distribution)
        rand(rng, p[2], n_iter)
    else
        StatsBase.sample(rng, p[2], min(length(p[2]), n_iter); replace = replace)
    end
    return Pair(p[1], vals)
end
function make_p_grid(ps::AbstractVector{<:Pair}, n_iter::Integer, rng::Random.AbstractRNG)
    replace = any(x -> isa(x[2], Distributions.Distribution), ps)
    return concrete_typed_array([make_p_grid(p, n_iter, rng, replace) for p in ps])
end
function make_p_grid(ps::AbstractDict, n_iter::Integer, rng::Random.AbstractRNG)
    replace = any(x -> isa(x[2], Distributions.Distribution), ps)
    return Dict(make_p_grid(key => val, n_iter, rng, replace) for (key, val) in ps)
end
function make_p_grid(pss::AbstractVector{<:Union{<:AbstractDict, <:AbstractVector{<:Pair}}},
                     n_iter::Integer, rng::Random.AbstractRNG)
    return concrete_typed_array([make_p_grid(ps, n_iter, rng) for ps in pss])
end
"""
    search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                           rscv::RandomisedSearchCrossValidation,
                           rd::ReturnsResult)

Performs randomised search cross-validation for portfolio optimisation estimators. Samples parameter sets from distributions or vectors, applies cross-validation splits, fits and scores each configuration, and selects the optimal parameters using the provided scoring strategy.

# Arguments

  - `opt`: Portfolio optimisation estimator to be tuned.
  - `rscv`: Randomised search cross-validation estimator specifying parameter grid, CV splitter, risk measure, scorer, execution strategy, number of iterations, RNG, and options.
  - `rd`: Returns result containing asset returns data.

# Returns

  - `SearchCrossValidationResult`: Result type containing the optimal estimator, test and train scores, parameter grid, and selected index.

# Details

  - Samples parameter sets from vectors or distributions.
  - Applies cross-validation splits to the returns data.
  - Fits the estimator for each sampled parameter set and split.
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
                                 rscv::RandomisedSearchCrossValidation, rd::ReturnsResult)
    if !isnothing(rscv.seed)
        Random.seed!(rscv.rng, rscv.seed)
    end
    return search_cross_validation(opt,
                                   GridSearchCrossValidation(make_p_grid(rscv.p,
                                                                         rscv.n_iter,
                                                                         rscv.rng);
                                                             cv = rscv.cv, r = rscv.r,
                                                             pred_scorer = rscv.pred_scorer,
                                                             scorer = rscv.scorer,
                                                             ex = rscv.ex,
                                                             train_score = rscv.train_score,
                                                             kwargs = rscv.kwargs), rd)
end

export RandomisedSearchCrossValidation
