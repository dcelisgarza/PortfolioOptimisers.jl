struct GeneralVariationInfoDistance{T1 <: Integer,
                                    T2 <: PortfolioOptimisersVarianceEstimator,
                                    T3 <: Union{<:Integer, <:AbstractBins}, T4 <: Bool} <:
       PortfolioOptimisersDistanceMetric
    power::T1
    ve::T2
    bins::T3
    normalise::T4
end
function GeneralVariationInfoDistance(; power::Integer = 1,
                                      ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                                      bins::Union{<:Integer, <:AbstractBins} = HacineGharbiRavier(),
                                      normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    @smart_assert(power >= one(power))
    return GeneralVariationInfoDistance{typeof(power), typeof(ve), typeof(bins),
                                        typeof(normalise)}(power, ve, bins, normalise)
end
function distance(de::GeneralVariationInfoDistance, ::Any, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.bins, de.normalise) .^ de.power
end

export GeneralVariationInfoDistance
