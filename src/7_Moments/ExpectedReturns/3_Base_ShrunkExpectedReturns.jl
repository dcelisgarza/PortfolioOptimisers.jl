abstract type AbstractShrunkExpectedReturnsEstimator <: AbstractExpectedReturnsEstimator end
abstract type AbstractShrunkExpectedReturnsTarget <: AbstractExpectedReturnsAlgorithm end

struct GrandMean <: AbstractShrunkExpectedReturnsTarget end
struct VolatilityWeighted <: AbstractShrunkExpectedReturnsTarget end
struct MeanSquareError <: AbstractShrunkExpectedReturnsTarget end

function target_mean(::GrandMean, mu::AbstractArray, sigma::AbstractMatrix; kwargs...)
    return fill(mean(mu), length(mu))
end
function target_mean(::VolatilityWeighted, mu::AbstractArray, sigma::AbstractMatrix;
                     isigma = nothing, kwargs...)
    if isnothing(isigma)
        isigma = sigma \ I
    end
    return fill(sum(isigma * mu) / sum(isigma), length(mu))
end
function target_mean(::MeanSquareError, mu::AbstractArray, sigma::AbstractMatrix;
                     T::Integer, kwargs...)
    return fill(tr(sigma) / T, length(mu))
end

export GrandMean, VolatilityWeighted, MeanSquareError, target_mean
