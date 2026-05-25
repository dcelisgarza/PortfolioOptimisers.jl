"""
$(DocStringExtensions.TYPEDEF)

Randomised search cross-validation estimator for portfolio optimisation. Samples parameter sets from distributions or vectors, applies cross-validation splits, fits and scores each configuration, and selects the optimal parameters using the provided scoring strategy.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RandomisedSearchCrossValidation(
        p::Union{AbstractVector{<:Pair{<:AbstractString, <:Any}},
                 AbstractVector{<:AbstractVector{<:Pair{<:AbstractString,
                                                        <:Any}}},
                 AbstractDict{<:AbstractString, <:Any},
                 AbstractVector{<:AbstractDict{<:AbstractString,
                                               <:Any}}};
        cv::SearchCV = KFold(),
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
    "$(field_dict[:p_cv])"
    p
    "$(field_dict[:cv])"
    cv
    "$(field_dict[:r])"
    r
    "$(field_dict[:scorer])"
    scorer
    "$(field_dict[:ex])"
    ex
    "$(field_dict[:n_iter])"
    n_iter
    "$(field_dict[:rng])"
    rng
    "$(field_dict[:seed])"
    seed
    "$(field_dict[:train_score])"
    train_score
    "$(field_dict[:kwargs])"
    kwargs
    function RandomisedSearchCrossValidation(p::Union{<:AbstractVector{<:Pair},
                                                      <:AbstractVector{<:AbstractVector{<:Pair}},
                                                      <:AbstractDict,
                                                      <:AbstractVector{<:AbstractDict}},
                                             cv::SearchCV, r::AbstractBaseRiskMeasure,
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
            @argcheck(all(x -> isa(x[1], GSCVKey), p))
            @argcheck(all(x -> isa(x[2], RSCVVal), p))
        elseif d_flag
            @argcheck(all(x -> isa(x, GSCVKey), keys(p)))
            @argcheck(all(x -> isa(x, RSCVVal), values(p)))
        elseif vp_flag || vd_flag
            @argcheck(all(!isempty, p), IsEmptyError)
            if vp_flag
                for _p in p
                    @argcheck(all(x -> isa(x[1], GSCVKey), _p))
                    @argcheck(all(x -> isa(x[2], RSCVVal), _p))
                end
            end
            if vd_flag
                for _p in p
                    @argcheck(all(x -> isa(x, GSCVKey), keys(_p)))
                    @argcheck(all(x -> isa(x, RSCVVal), values(_p)))
                end
            end
        end
        assert_nonempty_gt0_finite_val(n_iter, "n_iter")
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
                                         cv::SearchCV = KFold(),
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
                                                             scorer = rscv.scorer,
                                                             ex = rscv.ex,
                                                             train_score = rscv.train_score,
                                                             kwargs = rscv.kwargs), rd)
end

export RandomisedSearchCrossValidation
