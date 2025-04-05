abstract type AbstractEstimator end
abstract type AbstractAlgorithm end
abstract type AbstractResult end
function fit_estimator end
function fit_estimator! end
function fit_estimator(result::AbstractResult, args...; kwargs...)
    return result
end
function fit_estimator!(result::AbstractResult, args...; kwargs...)
    return result
end
function fit_estimator(::Nothing, args...; kwargs...)
    return nothing
end
function fit_estimator!(::Nothing, args...; kwargs...)
    return nothing
end

export fit, fit_estimator!
