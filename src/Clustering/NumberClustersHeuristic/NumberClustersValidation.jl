function validate_k_value(clustering::Hclust, nodes::AbstractVector, k::Integer)
    idx = cutree(clustering; k = k)
    clusters = Vector{Vector{Int}}(undef, length(minimum(idx):maximum(idx)))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(idx .== i)
    end
    for i ∈ nodes[1:(k - 1)]
        if is_leaf(i)
            continue
        end
        count = 0
        ln = pre_order(i.left)
        rn = pre_order(i.right)
        for cluster ∈ clusters
            if issubset(cluster, ln) || issubset(cluster, rn)
                count += 1
            end
        end
        if count == 0
            return false
        end
    end
    return true
end
function valid_k_clusters(clustering::Hclust, arr::AbstractVector)
    nodes = to_tree(clustering)[2]
    heights = [i.height for i ∈ nodes]
    nodes = nodes[sortperm(heights; rev = true)]
    while true
        k = all(.!isfinite.(arr)) ? length(arr) : argmax(arr)
        if validate_k_value(clustering, nodes, k)
            return k
        elseif all(isinf.(arr))
            return 1
        end
        arr[k] = -Inf
    end
end
