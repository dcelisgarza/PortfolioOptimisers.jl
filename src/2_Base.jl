abstract type AbstractEstimator end
abstract type AbstractAlgorithm end
abstract type AbstractResult end
function fit end
function fit! end
function fit(result::AbstractResult, args...; kwargs...)
    return result
end
function fit!(result::AbstractResult, args...; kwargs...)
    return result
end
function fit(::Nothing, args...; kwargs...)
    return nothing
end
function fit!(::Nothing, args...; kwargs...)
    return nothing
end

export fit, fit!
