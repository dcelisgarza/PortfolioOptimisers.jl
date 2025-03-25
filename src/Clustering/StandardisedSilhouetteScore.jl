struct StandardisedSilhouetteScore{T1 <: Union{Nothing, <:Integer},
                                   T2 <: Union{Nothing, <:Distances.SemiMetric}} <:
       NumberClustersHeuristic
    max_k::T1
    metric::T2
end
function StandardisedSilhouetteScore(; max_k::Union{Nothing, <:Integer} = nothing,
                                     metric::Union{Nothing, <:Distances.SemiMetric} = nothing)
    if !isnothing(max_k)
        @smart_assert(max_k >= one(max_k))
    end
    return StandardisedSilhouetteScore{typeof(max_k), typeof(metric)}(max_k, metric)
end

export StandardisedSilhouetteScore
