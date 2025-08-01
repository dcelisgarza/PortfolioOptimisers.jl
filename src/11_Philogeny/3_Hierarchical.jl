struct ClusterNode{tid, tl, tr, td, tcnt}
    id::tid
    left::tl
    right::tr
    height::td
    level::tcnt
    function ClusterNode(id, left::Union{Nothing, <:ClusterNode} = nothing,
                         right::Union{Nothing, <:ClusterNode} = nothing, height::Real = 0.0,
                         level::Int = 1)
        ilevel = isnothing(left) ? level : (left.level + right.level)
        return new{typeof(id), typeof(left), typeof(right), typeof(height), typeof(level)}(id,
                                                                                           left,
                                                                                           right,
                                                                                           height,
                                                                                           ilevel)
    end
end
function is_leaf(a::ClusterNode)
    return isnothing(a.left)
end
abstract type AbstractPreorderBy <: AbstractAlgorithm end
struct PreorderTreeByID <: AbstractPreorderBy end
get_node_property(::PreorderTreeByID, a::ClusterNode) = a.id
function pre_order(a::ClusterNode, preorder_by::AbstractPreorderBy = PreorderTreeByID())
    curNode = Vector{ClusterNode}(undef, 2 * a.level)
    lvisited = Set{typeof(get_node_property(preorder_by, a))}()
    rvisited = Set{typeof(get_node_property(preorder_by, a))}()
    curNode[1] = a
    k::Int = 1
    preorder = Vector{Int}(undef, 0)
    while k >= 1
        nd = curNode[k]
        ndid = nd.id
        if is_leaf(nd)
            push!(preorder, get_node_property(preorder_by, nd))
            k = k - one(k)
        else
            if ndid ∉ lvisited
                k = k + one(k)
                curNode[k] = nd.left
                push!(lvisited, ndid)
            elseif ndid ∉ rvisited
                k = k + one(k)
                curNode[k] = nd.right
                push!(rvisited, ndid)
                # If we've visited the left and right of this non-leaf
                # node already, go up in the tree.
            else
                k = k - 1
            end
        end
    end

    return preorder
end
function to_tree(a::Hclust)
    N = length(a.order)
    d = Vector{ClusterNode}(undef, 2 * N - 1)
    for i in eachindex(a.order)
        d[i] = ClusterNode(i)
    end
    merges = a.merges
    heights = a.heights
    nd = nothing
    for (i, height) in pairs(heights)
        fi = merges[i, 1]
        fj = merges[i, 2]
        fi = ifelse(fi < zero(eltype(merges)), -fi, fi + N)
        fj = ifelse(fj < zero(eltype(merges)), -fj, fj + N)
        nd = ClusterNode(i + N, d[fi], d[fj], height)
        d[N + i] = nd
    end
    return nd, d
end
function clusterise(cle::ClusteringEstimator{<:Any, <:Any, <:HClustAlgorithm, <:Any},
                    X::AbstractMatrix{<:Real}; branchorder::Symbol = :optimal,
                    dims::Int = 1, kwargs...)
    # S = cor(cle.ce, X; dims = dims, kwargs...)
    # D = distance(cle.de, S, X; dims = dims, kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    clustering = hclust(D; linkage = cle.alg.linkage, branchorder = branchorder)
    k = optimal_number_clusters(cle.onc, clustering, D)
    return HierarchicalClustering(; clustering = clustering, S = S, D = D, k = k)
end
function validate_k_value(clustering::Hclust, nodes::AbstractVector, k::Integer)
    idx = cutree(clustering; k = k)
    clusters = Vector{Vector{Int}}(undef, length(minimum(idx):maximum(idx)))
    for i in eachindex(clusters)
        clusters[i] = findall(idx .== i)
    end
    for i in nodes[1:(k - 1)]
        if is_leaf(i)
            continue
        end
        count = 0
        ln = pre_order(i.left)
        rn = pre_order(i.right)
        for cluster in clusters
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
    heights = [i.height for i in nodes]
    nodes = nodes[sortperm(heights; rev = true)]
    while true
        k = all(!isfinite, arr) ? length(arr) : argmax(arr)
        if validate_k_value(clustering, nodes, k)
            return k
        elseif all(isinf, arr)
            return 1
        end
        arr[k] = typemin(eltype(arr))
    end
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any,
                                                            <:PredefinedNumberClusters},
                                 clustering::Hclust, args...)
    k = onc.alg.k
    max_k = onc.max_k
    N = length(clustering.order)
    if isnothing(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    max_k = min(ceil(Int, sqrt(N)), max_k)
    if k > max_k
        k = max_k
    end
    nodes = to_tree(clustering)[2]
    heights = [i.height for i in nodes]
    nodes = nodes[sortperm(heights; rev = true)]
    flag = validate_k_value(clustering, nodes, k)
    if !flag
        # Above k
        flagu = false
        du = 0
        ku = k
        for i in (k + 1):max_k
            flagu = validate_k_value(clustering, nodes, i)
            if flagu
                ku = i
                break
            end
        end
        if flagu
            du = ku - k
        end
        # Below k
        flagl = false
        dl = 0
        kl = k
        for i in (k - 1):-1:1
            flagl = validate_k_value(clustering, nodes, i)
            if flagl
                kl = i
                break
            end
        end
        if flagl
            dl = k - kl
        end
        if du != 0 && dl == 0
            k = ku
        elseif du == 0 && dl != 0
            k = kl
        elseif du == dl
            k = max_k - ku > kl - 1 ? ku : kl
        else
            k = min(du, dl) == du ? ku : kl
        end
    end
    return k
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference},
                                 clustering::Hclust, dist::AbstractMatrix)
    max_k = onc.max_k
    N = size(dist, 1)
    if isnothing(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    c1 = min(ceil(Int, sqrt(N)), max_k)
    cluster_lvls = [cutree(clustering; k = i) for i in 1:c1]
    W_list = Vector{eltype(dist)}(undef, c1)
    W_list[1] = typemin(eltype(dist))
    for i in 2:c1
        lvl = cluster_lvls[i]
        c2 = maximum(unique(lvl))
        D_list = Vector{eltype(dist)}(undef, c2)
        for j in 1:c2
            cluster = findall(lvl .== j)
            cluster_dist = dist[cluster, cluster]
            if isempty(cluster_dist)
                continue
            end
            M = size(cluster_dist, 1)
            C_list = Vector{eltype(dist)}(undef, Int(M * (M - 1) / 2))
            k = 1
            for col in 1:M
                for row in (col + 1):M
                    C_list[k] = cluster_dist[row, col]
                    k += 1
                end
            end
            D_list[j] = k == 1 ? zero(eltype(dist)) : std(C_list; corrected = false)
        end
        W_list[i] = sum(D_list)
    end
    gaps = fill(typemin(eltype(dist)), c1)
    if c1 > 2
        gaps[1:(end - 2)] .= W_list[1:(end - 2)] + W_list[3:end] - 2 * W_list[2:(end - 1)]
    end
    return valid_k_clusters(clustering, gaps)
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any,
                                                            <:StandardisedSilhouetteScore},
                                 clustering::Hclust, dist::AbstractMatrix)
    max_k = onc.max_k
    N = size(dist, 1)
    if isnothing(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    c1 = min(ceil(Int, sqrt(N)), max_k)
    cluster_lvls = [cutree(clustering; k = i) for i in 1:c1]
    W_list = Vector{eltype(dist)}(undef, c1)
    W_list[1] = typemin(eltype(dist))
    for i in 2:c1
        lvl = cluster_lvls[i]
        sl = silhouettes(lvl, dist; metric = onc.alg.metric)
        msl = mean(sl)
        W_list[i] = msl / std(sl; mean = msl)
    end
    return valid_k_clusters(clustering, W_list)
end

export HClustAlgorithm, ClusterNode
