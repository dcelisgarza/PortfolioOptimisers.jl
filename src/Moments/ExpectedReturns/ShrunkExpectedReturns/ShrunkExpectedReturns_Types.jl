abstract type ShrunkExpectedReturnsEstimator <: ExpectedReturnsEstimator end
abstract type ShrunkExpectedReturnsTarget end

struct SERT_GrandMean <: ShrunkExpectedReturnsTarget end
struct SERT_VolatilityWeighted <: ShrunkExpectedReturnsTarget end
struct SERT_MeanSquareError <: ShrunkExpectedReturnsTarget end

function target_mean(::SERT_GrandMean, mu::AbstractArray, sigma::AbstractMatrix; kwargs...)
    return fill(mean(mu), length(mu))
end
function target_mean(::SERT_VolatilityWeighted, mu::AbstractArray, sigma::AbstractMatrix;
                     isigma = nothing, kwargs...)
    if isnothing(isigma)
        isigma = sigma \ I
    end
    return fill(sum(isigma * mu) / sum(isigma), length(mu))
end
function target_mean(::SERT_MeanSquareError, mu::AbstractArray, sigma::AbstractMatrix;
                     T::Integer, kwargs...)
    return fill(tr(sigma) / T, length(mu))
end

export SERT_GrandMean, SERT_VolatilityWeighted, SERT_MeanSquareError, target_mean
