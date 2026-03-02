struct OptimisationCrossValidation{T1, T2} <: AbstractEstimator
    cv::T1
    score::T2
    function OptimisationCrossValidation(cv::OptimisationCrossValidationEstimator,
                                         score::Option{<:PredictionCrossValScorer})
        return new{typeof(cv), typeof(score)}(cv, score)
    end
end
function OptimisationCrossValidation(; cv::OptimisationCrossValidationEstimator = KFold(),
                                     score::Option{<:PredictionCrossValScorer} = nothing)
    return OptimisationCrossValidation(cv, score)
end
const NonCombOptCV = Union{<:KFold, <:WalkForwardEstimator}

export OptimisationCrossValidation
