struct VariationInfoDistanceDistance{T1 <: Union{<:Integer, <:AbstractBins}, T2 <: Bool,
                                     T3 <: Distances.Metric, T4 <: Tuple,
                                     T5 <: NamedTuple} <:
       PortfolioOptimisersDistanceDistanceMetric
    bins::T1
    normalise::T2
    dist::T3
    args::T4
    kwargs::T5
end
function VariationInfoDistanceDistance(;
                                       bins::Union{<:Integer, <:AbstractBins} = HacineGharbiRavier(),
                                       normalise::Bool = true,
                                       dist::Distances.Metric = Distances.Euclidean(),
                                       args::Tuple = (), kwargs::NamedTuple = (;))
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return VariationInfoDistanceDistance{typeof(bins), typeof(normalise), typeof(dist),
                                         typeof(args), typeof(kwargs)}(bins, normalise,
                                                                       dist, args, kwargs)
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
