struct GeneralVariationInfoDistanceDistance{T1 <: Integer,
                                            T2 <: PortfolioOptimisersVarianceEstimator,
                                            T3 <: Union{<:Integer, <:AbstractBins},
                                            T4 <: Bool, T5 <: Distances.Metric, T6 <: Tuple,
                                            T7 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    power::T1
    ve::T2
    bins::T3
    normalise::T4
    dist::T5
    args::T6
    kwargs::T7
end
function GeneralVariationInfoDistanceDistance(; power::Integer = 1,
                                              ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                                              bins::Union{<:Integer, <:AbstractBins} = B_HacineGharbiRavier(),
                                              normalise::Bool = true,
                                              dist::Distances.Metric = Distances.Euclidean(),
                                              args::Tuple = (), kwargs::NamedTuple = (;))
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    @smart_assert(power >= one(power))
    return GeneralVariationInfoDistanceDistance{typeof(power), typeof(ve), typeof(bins),
                                                typeof(normalise), typeof(dist),
                                                typeof(args), typeof(kwargs)}(power, ve,
                                                                              bins,
                                                                              normalise,
                                                                              dist, args,
                                                                              kwargs)
end
function distance(de::GeneralVariationInfoDistanceDistance, ::Any, X::AbstractMatrix;
                  dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    dist = variation_info(X, de.bins, de.normalise) .^ de.power
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export GeneralVariationInfoDistanceDistance
