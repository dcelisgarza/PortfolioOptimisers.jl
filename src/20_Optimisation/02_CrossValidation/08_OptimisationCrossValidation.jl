"""
$(DocStringExtensions.TYPEDEF)

Wraps a cross-validation scheme and an optional scorer to form a complete optimisation
cross-validation pipeline.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimisationCrossValidation(;
        cv::OptCVER = KFold(),
        scorer::Option{<:PredictionCrossValScorer} = nothing
    ) -> OptimisationCrossValidation

Keywords correspond to the struct's fields.

# Related

  - [`KFold`](@ref)
  - [`WalkForwardEstimator`](@ref)
  - [`PredictionCrossValScorer`](@ref)
  - [`NearestQuantilePrediction`](@ref)
"""
@concrete struct OptimisationCrossValidation <: AbstractEstimator
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:scorer])
    """
    scorer
    function OptimisationCrossValidation(cv::OptCVER,
                                         scorer::Option{<:PredictionCrossValScorer})
        return new{typeof(cv), typeof(scorer)}(cv, scorer)
    end
end
function OptimisationCrossValidation(; cv::OptCVER = KFold(),
                                     scorer::Option{<:PredictionCrossValScorer} = nothing)::OptimisationCrossValidation
    return OptimisationCrossValidation(cv, scorer)
end
"""
    const NonCombOptCV = Union{<:KFold, <:WalkForwardEstimator}

Alias for non-combinatorial optimisation cross-validation schemes.

Matches either a [`KFold`](@ref) or a [`WalkForwardEstimator`](@ref). Used for dispatch in routines that require sequential or fold-based (non-combinatorial) cross-validation.

# Related

  - [`KFold`](@ref)
  - [`WalkForwardEstimator`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
const NonCombOptCV = Union{<:KFold, <:WalkForwardEstimator}

export OptimisationCrossValidation
