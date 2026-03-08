"""
    struct RandomisedSearchCrossValidation{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
           AbstractSearchCrossValidationEstimator
        p::T1
        cv::T2
        r::T3
        scorer::T4
        ex::T5
        n_iter::T6
        rng::T7
        seed::T8
        train_score::T9
        kwargs::T10
    end

Randomised search cross-validation estimator for portfolio optimisation. Samples parameter sets from distributions or vectors, applies cross-validation splits, fits and scores each configuration, and selects the optimal parameters using the provided scoring strategy.

# Fields

  - `p`: Parameter grid, supporting vectors, dictionaries, or distributions for random sampling. If all parameters in a grid are vectors they are sampled uniformly without replacement, if at least one of them is a distribution they are sampled with replacement. When sampling without replacement the sampling is performed up until the list of candidates is exhausted. If a vector of grids, each grid is sampled independently and then they're all concatenated.
  - `cv`: Cross-validation splitter.
  - `r`: Risk measure used for scoring.
  - `scorer`: Scoring function to select optimal parameter set.
  - `ex`: Parallel execution strategy.
  - `n_iter`: Number of random parameter sets to sample.
  - `rng`: Random number generator for reproducibility.
  - `seed`: Optional seed for RNG.
  - `train_score`: Whether to record training scores.
  - `kwargs`: Additional keyword arguments for customisation.

# Constructors

```julia
RandomisedSearchCrossValidation(p::Union{AbstractVector{<:Pair{<:AbstractString, <:Any}},
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
                                train_score::Bool = false, kwargs::NamedTuple = (;))
```

Keyword arguments correspond to the fields above.

## Validation

  - Parameter grid `p` must not be empty.
  - All parameter values must be vectors or distributions.
  - For nested parameter grids, each must not be empty.
  - `n_iter` must be non-empty, greater than zero, and finite.

# Examples

```jldoctest
julia> RandomisedSearchCrossValidation(Dict("alpha" => [0.1, 0.2, 0.3],
                                            "beta" => Distributions.Normal(1.0, 0.5)))
RandomisedSearchCrossValidation
            p ┼ Dict{String, Any}: Dict{String, Any}("alpha" => [0.1, 0.2, 0.3], "beta" => Normal{Float64}(μ=1.0, σ=0.5))
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
        score ┼ PortfolioOptimisers.HighestMeanScore()
           ex ┼ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
       n_iter ┼ Int64: 10
          rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
         seed ┼ nothing
  train_score ┼ Bool: false
       kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractSearchCrossValidationEstimator`]-(@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`SearchCrossValidationResult`](@ref)
  - [`CrossValSearchScorer`](@ref)
"""
struct RandomisedSearchCrossValidation{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
       AbstractSearchCrossValidationEstimator
    p::T1
    cv::T2
    r::T3
    scorer::T4
    ex::T5
    n_iter::T6
    rng::T7
    seed::T8
    train_score::T9
    kwargs::T10
    function RandomisedSearchCrossValidation(p::Union{<:AbstractVector{<:Pair{<:AbstractString,
                                                                              <:Any}},
                                                      <:AbstractVector{<:AbstractVector{<:Pair{<:AbstractString,
                                                                                               <:Any}}},
                                                      <:AbstractDict{<:AbstractString,
                                                                     <:Any},
                                                      <:AbstractVector{<:AbstractDict{<:AbstractString,
                                                                                      <:Any}}},
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
            @argcheck(all(x->isa(x[2],
                                 Union{<:AbstractVector, <:Distributions.Distribution}), p))
        elseif d_flag
            @argcheck(all(x->isa(x, Union{<:AbstractVector, <:Distributions.Distribution}),
                          values(p)))
        elseif vp_flag || vd_flag
            @argcheck(all(!isempty, p), IsEmptyError)
            if vp_flag
                for _p in p
                    @argcheck(all(x->isa(x[2],
                                         Union{<:AbstractVector,
                                               <:Distributions.Distribution}), _p))
                end
            end
            if vd_flag
                for _p in p
                    @argcheck(all(x->isa(x,
                                         Union{<:AbstractVector,
                                               <:Distributions.Distribution}), values(_p)))
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
function RandomisedSearchCrossValidation(p::Union{<:AbstractVector{<:Pair{<:AbstractString,
                                                                          <:Any}},
                                                  <:AbstractVector{<:AbstractVector{<:Pair{<:AbstractString,
                                                                                           <:Any}}},
                                                  <:AbstractDict{<:AbstractString, <:Any},
                                                  <:AbstractVector{<:AbstractDict{<:AbstractString,
                                                                                  <:Any}}};
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
function make_p_grid(p::Pair{<:AbstractString}, n_iter::Integer, rng::Random.AbstractRNG,
                     replace::Bool = false)
    vals = if isa(p[2], Distributions.Distribution)
        rand(rng, p[2], n_iter)
    else
        StatsBase.sample(rng, p[2], min(length(p[2]), n_iter); replace = replace)
    end
    return Pair(p[1], vals)
end
function make_p_grid(ps::AbstractVector{<:Pair{<:AbstractString, <:Any}}, n_iter::Integer,
                     rng::Random.AbstractRNG)
    replace = any(x->isa(x[2], Distributions.Distribution), ps)
    return concrete_typed_array([make_p_grid(p, n_iter, rng, replace) for p in ps])
end
function make_p_grid(ps::AbstractDict{<:AbstractString, <:Any}, n_iter::Integer,
                     rng::Random.AbstractRNG)
    replace = any(x->isa(x[2], Distributions.Distribution), ps)
    return Dict(make_p_grid(key => val, n_iter, rng, replace) for (key, val) in ps)
end
function make_p_grid(pss::AbstractVector{<:Union{<:AbstractDict{<:AbstractString, <:Any},
                                                 <:AbstractVector{<:Pair{<:AbstractString,
                                                                         <:Any}}}},
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

# Validation

  - Sets RNG seed if provided.
  - Validates parameter grid and number of iterations.
  - Ensures sampled parameter sets are valid.

# Details

  - Samples parameter sets from vectors or distributions.
  - Applies cross-validation splits to the returns data.
  - Fits the estimator for each sampled parameter set and split.
  - Scores each configuration using the specified risk measure and scoring function.
  - Selects the optimal parameter set based on cross-validation scores.
  - Returns a result object encapsulating the optimal estimator and score matrices.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`]-(@ref)
  - [`RandomisedSearchCrossValidation`](@ref)
  - [`ReturnsResult`](@ref)
  - [`GridSearchCrossValidation`](@ref)

# Examples
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
