struct VariationInfoDistanceDistance{T1 <: PortfolioOptimisersVarianceEstimator,
                                     T2 <: Union{<:Integer, <:AbstractBins}, T3 <: Bool,
                                     T4 <: Distances.Metric, T5 <: Tuple,
                                     T6 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    ve::T1
    bins::T2
    normalise::T3
    dist::T4
    args::T5
    kwargs::T6
end
function VariationInfoDistanceDistance(;
                                       ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                                       bins::Union{<:Integer, <:AbstractBins} = B_HacineGharbiRavier(),
                                       normalise::Bool = true,
                                       dist::Distances.Metric = Distances.Euclidean(),
                                       args::Tuple = (), kwargs::NamedTuple = (;))
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return VariationInfoDistanceDistance{typeof(ve), typeof(bins), typeof(normalise),
                                         typeof(dist), typeof(args), typeof(kwargs)}(ve,
                                                                                     bins,
                                                                                     normalise,
                                                                                     dist,
                                                                                     args,
                                                                                     kwargs)
end
function distance(de::VariationInfoDistanceDistance, ::Any, X::AbstractMatrix;
                  dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    dist = variation_info(X, de.bins, de.normalise)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export VariationInfoDistanceDistance
