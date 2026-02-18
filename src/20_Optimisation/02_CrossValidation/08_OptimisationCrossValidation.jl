struct OptimisationCrossValidation{T1, T2} <: AbstractEstimator
    cv::T1
    score::T2
    function OptimisationCrossValidation(cv::OptimisationCrossValidationEstimator,
                                         score::Option{<:CrossValScorer})
        return new{typeof(cv), typeof(score)}(cv, score)
    end
end
function OptimisationCrossValidation(; cv::OptimisationCrossValidationEstimator = KFold(),
                                     score::Option{<:CrossValScorer} = nothing)
    return OptimisationCrossValidation(cv, score)
end
