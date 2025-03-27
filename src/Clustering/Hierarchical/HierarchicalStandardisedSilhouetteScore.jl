function optimal_number_clusters(nch::StandardisedSilhouetteScore, clustering::Hclust,
                                 dist::AbstractMatrix)
    max_k = nch.max_k
    N = size(dist, 1)
    if isnothing(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    c1 = min(ceil(Int, sqrt(N)), max_k)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:c1]
    W_list = Vector{eltype(dist)}(undef, c1)
    W_list[1] = typemin(eltype(W_list))
    for i ∈ 2:c1
        lvl = cluster_lvls[i]
        sl = silhouettes(lvl, dist; metric = nch.metric)
        msl = mean(sl)
        W_list[i] = msl / std(sl; mean = msl)
    end
    return valid_k_clusters(clustering, W_list)
end
