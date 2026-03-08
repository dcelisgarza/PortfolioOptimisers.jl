struct OptimisationCrossValidation{T1, T2} <: AbstractEstimator
    cv::T1
    scorer::T2
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
