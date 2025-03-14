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
function optimal_number_clusters(nch::StandardisedSilhouetteScore, clustering::Hclust,
                                 dist::AbstractMatrix)
    metric = nch.metric
    max_k = nch.max_k
    N = size(dist, 1)
    if iszero(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    c1 = min(ceil(Int, sqrt(N)), max_k)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:c1]
    W_list = Vector{eltype(dist)}(undef, c1)
    W_list[1] = -Inf
    for i ∈ 2:c1
        lvl = cluster_lvls[i]
        sl = silhouettes(lvl, dist; metric = metric)
        msl = mean(sl)
        W_list[i] = msl / std(sl; mean = msl)
    end
    return valid_k_clusters(clustering, W_list)
end

export StandardisedSilhouetteScore
