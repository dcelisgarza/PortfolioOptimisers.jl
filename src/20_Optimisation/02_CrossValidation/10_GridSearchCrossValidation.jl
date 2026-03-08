"""
    struct GridSearchCrossValidation{T1, T2, T3, T4, T5, T6, T7} <:
           AbstractSearchCrossValidationEstimator
        p::T1
        cv::T2
        r::T3
        scorer::T4
        ex::T5
        train_score::T6
        kwargs::T7
    end

Performs grid search cross-validation for portfolio optimisation estimators. Iterates over parameter grids, applies cross-validation splits, and scores each configuration to select the optimal parameters.

# Fields

  - `p`: Parameter grid for optimisation.
  - `cv`: Cross-validation splitter (e.g., KFold).
  - `r`: Risk measure used for scoring (e.g., ConditionalValueatRisk).
  - `scorer`: Scoring function to select optimal parameter set.
  - `ex`: Parallel execution strategy (e.g., FLoops.ThreadedEx).
  - `train_score`: Whether to record training scores.
  - `kwargs`: Additional keyword arguments for customisation.

# Constructors

```julia
GridSearchCrossValidation(p::MultiGSCVValType_VecMultiGSCVValType; cv::SearchCV = KFold(),
                          r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                          scorer::CrossValSearchScorer = HighestMeanScore(),
                          ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                          train_score::Bool = false, kwargs::NamedTuple = (;))
```

  - Arguments correspond to fields above.

## Validation

  - `@argcheck(!isempty(p), IsEmptyError)`: Parameter grid must not be empty.
  - If `p` is a vector of parameter sets, each must not be empty.
  - All validations use custom error types.

# Examples

```jldoctest
julia> GridSearchCrossValidation(Dict("alpha" => [0.1, 0.2], "beta" => [1.0, 2.0]))
GridSearchCrossValidation
            p ┼ Dict{String, Vector{Float64}}: Dict("alpha" => [0.1, 0.2], "beta" => [1.0, 2.0])
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
  train_score ┼ Bool: false
       kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`MultiGSCVValType_VecMultiGSCVValType`](@ref)
  - [`SearchCV`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`CrossValSearchScorer`](@ref)
  - [`search_cross_validation`](@ref)
"""
struct GridSearchCrossValidation{T1, T2, T3, T4, T5, T6, T7} <:
       AbstractSearchCrossValidationEstimator
    p::T1
    cv::T2
    r::T3
    scorer::T4
    ex::T5
    train_score::T6
    kwargs::T7
    function GridSearchCrossValidation(p::MultiGSCVValType_VecMultiGSCVValType,
                                       cv::SearchCV, r::AbstractBaseRiskMeasure,
                                       scorer::CrossValSearchScorer,
                                       ex::FLoops.Transducers.Executor, train_score::Bool,
                                       kwargs::NamedTuple)
        @argcheck(!isempty(p), IsEmptyError)
        if isa(p, VecMultiGSCVValType)
            @argcheck(all(!isempty, p), IsEmptyError)
        end
        return new{typeof(p), typeof(cv), typeof(r), typeof(scorer), typeof(ex),
                   typeof(train_score), typeof(kwargs)}(p, cv, r, scorer, ex, train_score,
                                                        kwargs)
    end
end
function GridSearchCrossValidation(p::MultiGSCVValType_VecMultiGSCVValType;
                                   cv::SearchCV = KFold(),
                                   r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                   scorer::CrossValSearchScorer = HighestMeanScore(),
                                   ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                   train_score::Bool = false, kwargs::NamedTuple = (;))
    return GridSearchCrossValidation(p, cv, r, scorer, ex, train_score, kwargs)
end
function lens_val_grid(estval::AbstractVector{<:Pair{<:String, <:AbstractVector}})
    vals = vec(collect(Iterators.product(map(x -> x[2], estval)...)))
    lenses = fill(map(x -> parse_lens(x[1]), estval), length(vals))
    return lenses, vals
end
function lens_val_grid(estval::AbstractDict{<:String, <:AbstractVector})
    vals = vec(collect(Iterators.product(values(estval)...)))
    lenses = fill(parse_lens.(keys(estval)), length(vals))
    return lenses, vals
end
function lens_val_grid(estvals::AbstractVector{<:Union{<:AbstractVector{<:Pair{<:String,
                                                                               <:AbstractVector}},
                                                       <:AbstractDict{<:String,
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

# Validation

  - Ensures parameter grid is not empty.
  - Validates cross-validation splits and indices.

# Details

  - Iterates over all parameter combinations in the grid.
  - Applies cross-validation splits to the returns data.
  - Fits the estimator for each parameter set and split.
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
                                 gscv::GridSearchCrossValidation, rd::ReturnsResult)
    p = gscv.p
    lens_grid, val_grid = lens_val_grid(p)
    cv = split(gscv.cv, rd)
    @argcheck(isa(cv.test_idx[1], VecInt))
    N = length(val_grid)
    M = length(cv.train_idx)
    test_scores = Matrix{eltype(rd.X)}(undef, M, N)
    train_scores = if gscv.train_score
        Matrix{eltype(rd.X)}(undef, M, N)
    else
        nothing
    end
    let opt = opt, test_scores = test_scores, train_scores = train_scores
        FLoops.@floop gscv.ex for (i, (lenses, vals)) in enumerate(zip(lens_grid, val_grid))
            local opti = opt
            for (lens, val) in zip(lenses, vals)
                opti = Accessors.set(opti, lens, val)
            end
            for (j, (train_idx, test_idx)) in enumerate(zip(cv.train_idx, cv.test_idx))
                test_score, train_score = fit_and_score(opti, gscv, rd, train_idx, test_idx)
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
    return SearchCrossValidationResult(opt, test_scores, train_scores, lens_grid, val_grid,
                                       opt_idx)
end
export search_cross_validation, GridSearchCrossValidation
