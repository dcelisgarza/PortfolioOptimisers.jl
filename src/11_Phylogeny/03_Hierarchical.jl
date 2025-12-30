"""
    struct ClusterNode{tid, tl, tr, td, tcnt} <: AbstractResult
        id::tid
        left::tl
        right::tr
        height::td
        level::tcnt
    end

Node type for representing clusters in a hierarchical clustering tree.

`ClusterNode` encapsulates the structure of a node in a clustering tree, including its unique identifier, left and right child nodes, height, and level in the tree. Leaf nodes have `left` and `right` set to `nothing`.

# Fields

  - `id`: Unique identifier for the node.
  - `left`: Left child node.
  - `right`: Right child node.
  - `height`: Height of the node in the tree.
  - `level`: Level of the node in the tree.

# Constructor

    ClusterNode(id, left::Option{<:ClusterNode} = nothing,
                right::Option{<:ClusterNode} = nothing, height::Number = 0.0,
                level::Int = 1)

Positional and keyword arguments correspond to the fields above. The `level` is automatically computed based on the levels of child nodes if they exist.

# Examples

```jldoctest
julia> ClusterNode(1)
ClusterNode
      id ┼ Int64: 1
    left ┼ nothing
   right ┼ nothing
  height ┼ Float64: 0.0
   level ┴ Int64: 1
```

# Related

  - [`is_leaf`](@ref)
  - [`pre_order`](@ref)
"""
struct ClusterNode{tid, tl, tr, td, tcnt} <: AbstractResult
    id::tid
    left::tl
    right::tr
    height::td
    level::tcnt
    function ClusterNode(id, left::Option{<:ClusterNode} = nothing,
                         right::Option{<:ClusterNode} = nothing, height::Number = 0.0,
                         level::Int = 1)
        ilevel = isnothing(left) ? level : (left.level + right.level)
        return new{typeof(id), typeof(left), typeof(right), typeof(height), typeof(level)}(id,
                                                                                           left,
                                                                                           right,
                                                                                           height,
                                                                                           ilevel)
    end
end
const VecClN = AbstractVector{<:ClusterNode}
"""
    is_leaf(a::ClusterNode)

Determine if a `ClusterNode` is a leaf node.

Returns `true` if the node has no left child (`left == nothing`), indicating it is a leaf in the clustering tree.

# Arguments

  - `a`: The node to check.

# Returns

  - `Bool`: `true` if the node is a leaf, `false` otherwise.

# Examples

```jldoctest
julia> PortfolioOptimisers.is_leaf(ClusterNode(1))
true
```

# Related

  - [`ClusterNode`](@ref)
"""
function is_leaf(a::ClusterNode)
    return isnothing(a.left)
end
"""
    abstract type AbstractPreorderBy <: AbstractAlgorithm end

Abstract supertype for all preorder traversal strategies in PortfolioOptimisers.jl.

Concrete types implementing specific preorder traversal logic should subtype `AbstractPreorderBy`. This enables flexible extension and dispatch of preorder routines for hierarchical clustering trees.

# Related

  - [`PreorderTreeByID`](@ref)
  - [`pre_order`](@ref)
"""
abstract type AbstractPreorderBy <: AbstractAlgorithm end
"""
    struct PreorderTreeByID <: AbstractPreorderBy end

Preorder traversal strategy that visits nodes by their ID.

`PreorderTreeByID` is used to specify that preorder traversal should be performed using the node's `id` property.

# Related

  - [`AbstractPreorderBy`](@ref)
  - [`get_node_property`](@ref)
  - [`pre_order`](@ref)
"""
struct PreorderTreeByID <: AbstractPreorderBy end
"""
    get_node_property(preorder_by::PreorderTreeByID, a::ClusterNode)

Get the property of a node used for preorder traversal.

For `PreorderTreeByID`, this returns the node's `id`.

# Arguments

  - `preorder_by`: Preorder traversal strategy.
  - `a`: The node.

# Returns

  - The node's identifier.

# Related

  - [`PreorderTreeByID`](@ref)
  - [`pre_order`](@ref)
"""
get_node_property(::PreorderTreeByID, a::ClusterNode) = a.id

"""
    pre_order(a::ClusterNode; preorder_by::AbstractPreorderBy = PreorderTreeByID())

Perform a preorder traversal of a hierarchical clustering tree.

Returns a vector of node properties (by default, node IDs) in preorder (root, left, right) order. The traversal strategy can be customised by providing a subtype of `AbstractPreorderBy`.

# Arguments

  - `a`: The root node of the tree.
  - `preorder_by`: Traversal strategy.

# Returns

  - `res::Vector{Int}`: Vector of node properties in preorder.

# Related

  - [`ClusterNode`](@ref)
  - [`AbstractPreorderBy`](@ref)
  - [`PreorderTreeByID`](@ref)
  - [`get_node_property`](@ref)
"""
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
"""
    to_tree(a::Hclust)

Convert a hierarchical clustering result to a tree of `ClusterNode` objects.

This function takes a hierarchical clustering object from [`Clustering.jl`](https://juliastats.org/Clustering.jl/stable/hclust.html) and constructs a tree representation using `ClusterNode` nodes. It returns the root node and a vector of all nodes in the tree.

# Arguments

  - `a`: Hierarchical clustering object.

# Returns

  - `root::ClusterNode`: The root node of the clustering tree.
  - `nodes::Vector{ClusterNode}`: Vector containing all nodes in the tree.

# Related

  - [`ClusterNode`](@ref)
  - [`pre_order`](@ref)
"""
function to_tree(a::Clustering.Hclust)
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
"""
    clusterise(cle::HierarchicalClusteringEstimator{<:Any, <:Any, <:HClustAlgorithm, <:Any},
               X::MatNum; branchorder::Symbol = :optimal, dims::Int = 1,
               kwargs...)

Run hierarchical clustering and return the result as a [`HierarchicalClustering`](@ref) object.

This function applies the specified clustering estimator to the input data matrix, computes the similarity and distance matrices, performs hierarchical clustering, and selects the optimal number of clusters. The result is returned as a `HierarchicalClustering` object.

# Arguments

  - `cle`: Clustering estimator.
  - `X`: Data matrix (observations × assets).
  - `branchorder`: Branch ordering strategy for hierarchical clustering.
  - `dims`: Dimension along which to cluster.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `res::HierarchicalClustering`: Result object containing clustering, similarity, distance matrices, and number of clusters.

# Related

  - [`HierarchicalClustering`](@ref)
  - [`HierarchicalClusteringEstimator`](@ref)
"""
function clusterise(cle::HierarchicalClusteringEstimator{<:Any, <:Any, <:HClustAlgorithm,
                                                         <:Any}, X::MatNum;
                    branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    clustering = Clustering.hclust(D; linkage = cle.alg.linkage, branchorder = branchorder)
    k = optimal_number_clusters(cle.onc, clustering, D)
    return HierarchicalClustering(; clustering = clustering, S = S, D = D, k = k)
end
"""
    validate_k_value(clustering::Clustering.Hclust, nodes::VecClN, k::Integer)

Validate whether a given number of clusters `k` is consistent with the hierarchical clustering tree.

This function checks if the clustering assignment for `k` clusters is compatible with the tree structure, ensuring that each non-leaf node's children correspond to valid clusters.

# Arguments

  - `clustering`: Hierarchical clustering object.
  - `nodes`: Vector of nodes in the clustering tree.
  - `k`: Number of clusters to validate.

# Returns

  - `flag::Bool`: `true` if `k` is a valid number of clusters, `false` otherwise.

# Related

  - [`optimal_number_clusters`](@ref)
  - [`ClusterNode`](@ref)
"""
function validate_k_value(clustering::Clustering.Hclust, nodes::VecClN, k::Integer)
    idx = Clustering.cutree(clustering; k = k)
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
"""
    valid_k_clusters(clustering::Hclust, arr::VecNum)

Find a valid number of clusters for a hierarchical clustering tree given a scoring array.

This function iteratively searches for a valid `k` (number of clusters) by checking the scoring array and validating each candidate using [`validate_k_value`](@ref). Returns the first valid `k` found, or `1` if none are valid.

# Arguments

  - `clustering`: Hierarchical clustering object.
  - `arr`: Array of scores for each possible number of clusters.

# Returns

  - `k::Integer`: Valid number of clusters.

# Related

  - [`validate_k_value`](@ref)
  - [`optimal_number_clusters`](@ref)
"""
function valid_k_clusters(clustering::Clustering.Hclust, arr::VecNum)
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
"""
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:Integer}, clustering::Hclust,
                            args...)
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference},
                            clustering::Hclust, dist::MatNum)
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:StandardisedSilhouetteScore},
                            clustering::Hclust, dist::MatNum)

Select the optimal number of clusters for a hierarchical clustering tree.

This function applies the specified optimal number of clusters estimator (`onc`) to a hierarchical clustering result and distance matrix, using the configured algorithm (e.g., [`SecondOrderDifference`](@ref), [`StandardisedSilhouetteScore`](@ref), or given directly). The selection is based on cluster validity and scoring metrics.

# Arguments

  - `onc`: Optimal number of clusters estimator.

      + `onc::OptimalNumberClusters{<:Any, <:Integer}`: Uses a user-specified fixed number of clusters `k` directly. If `k` is not valid, searches above and below for the nearest valid cluster count.
      + `onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference}`: Computes the second-order difference of a clustering evaluation metric for each possible cluster count, and selects the first valid `k` that maximises the difference.
      + `onc::OptimalNumberClusters{<:Any, <:StandardisedSilhouetteScore}`: Computes the standardised silhouette score for each possible cluster count, and selects the first valid `k` that maximises the score.

  - `clustering`: Hierarchical clustering object.
  - `dist`: Distance matrix used for clustering.

# Returns

  - `onc::Integer`: Selected optimal number of clusters.

# Related

  - [`OptimalNumberClusters`](@ref)
  - [`valid_k_clusters`](@ref)
  - [`validate_k_value`](@ref)
"""
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:Integer},
                                 clustering::Clustering.Hclust, args...)
    k = onc.alg
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
                                 clustering::Clustering.Hclust, dist::MatNum)
    max_k = onc.max_k
    N = size(dist, 1)
    if isnothing(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    c1 = min(ceil(Int, sqrt(N)), max_k)
    cluster_lvls = [Clustering.cutree(clustering; k = i) for i in 1:c1]
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
            D_list[j] = if k == 1
                zero(eltype(dist))
            else
                Statistics.std(C_list; corrected = false)
            end
        end
        W_list[i] = sum(D_list)
    end
    gaps = fill(typemin(eltype(dist)), c1)
    if c1 > 2
        gaps[1:(end - 2)] = W_list[1:(end - 2)] + W_list[3:end] - 2 * W_list[2:(end - 1)]
    end
    return valid_k_clusters(clustering, gaps)
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any,
                                                            <:StandardisedSilhouetteScore},
                                 clustering::Clustering.Hclust, dist::MatNum)
    max_k = onc.max_k
    N = size(dist, 1)
    if isnothing(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    c1 = min(ceil(Int, sqrt(N)), max_k)
    cluster_lvls = [Clustering.cutree(clustering; k = i) for i in 1:c1]
    W_list = Vector{eltype(dist)}(undef, c1)
    W_list[1] = typemin(eltype(dist))
    for i in 2:c1
        lvl = cluster_lvls[i]
        sl = Clustering.silhouettes(lvl, dist; metric = onc.alg.metric)
        msl = Statistics.mean(sl)
        W_list[i] = msl / Statistics.std(sl; mean = msl)
    end
    return valid_k_clusters(clustering, W_list)
end

export ClusterNode, is_leaf, PreorderTreeByID, pre_order, to_tree, optimal_number_clusters
