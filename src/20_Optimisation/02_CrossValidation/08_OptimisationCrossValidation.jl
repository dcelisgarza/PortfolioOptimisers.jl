"""
$(DocStringExtensions.TYPEDEF)

Wraps a cross-validation scheme and an optional scorer to form a complete optimisation
cross-validation pipeline.

# Fields

  - `cv::OptCVER`: Cross-validation scheme (e.g. [`KFold`](@ref) or
    [`WalkForwardEstimator`](@ref)).
  - `scorer::Option{<:PredictionCrossValScorer}`: Optional scorer used to select a
    single prediction from the cross-validation results. Defaults to `nothing`.

# Constructors

    OptimisationCrossValidation(;
        cv::OptCVER = KFold(),
        scorer::Option{<:PredictionCrossValScorer} = nothing
    ) -> OptimisationCrossValidation

# Related

  - [`KFold`](@ref)
  - [`WalkForwardEstimator`](@ref)
  - [`PredictionCrossValScorer`](@ref)
  - [`NearestQuantilePrediction`](@ref)
"""
@concrete struct OptimisationCrossValidation <: AbstractEstimator
    cv
    scorer
    function OptimisationCrossValidation(cv::OptCVER,
                                         scorer::Option{<:PredictionCrossValScorer})
        return new{typeof(cv), typeof(scorer)}(cv, scorer)
    end
end
function OptimisationCrossValidation(; cv::OptCVER = KFold(),
                                     scorer::Option{<:PredictionCrossValScorer} = nothing)
    return OptimisationCrossValidation(cv, scorer)
end
const NonCombOptCV = Union{<:KFold, <:WalkForwardEstimator}

export OptimisationCrossValidation
