struct StandardisedSilhouetteScore{T1 <: Union{Nothing, <:Distances.SemiMetric},
                                   T2 <: Integer} <: NumberClustersHeuristic
    metric::T1
    max_k::T2
end
function StandardisedSilhouetteScore(;
                                     metric::Union{Nothing, <:Distances.SemiMetric} = nothing,
                                     max_k::Integer = 0)
    @smart_assert(max_k >= 0)
    return StandardisedSilhouetteScore{typeof(metric), typeof(max_k)}(metric, max_k)
end

export StandardisedSilhouetteScore
