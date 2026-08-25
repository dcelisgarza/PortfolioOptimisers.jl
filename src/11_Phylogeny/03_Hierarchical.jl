"""
$(DocStringExtensions.TYPEDEF)

Binds one merge of a dendrogram to the two clusters it joined.

The tree form of a linkage matrix: [`to_tree`](@ref) turns a `Clustering.Hclust` into one of these per merge, plus one per asset, and the last one built is the root. A leaf carries `left` and `right` as `nothing`, which is what [`is_leaf`](@ref) tests.

# `level` counts leaves, it does not measure depth

`level` is the number of assets in the subtree below the node — `1` for a leaf, and the sum of the two children's counts for a merge. It is the **fourth column of a linkage matrix**, not a position in the tree, and on an eight-asset universe the two disagree: the root carries `level = 8` where its depth is `5`.

[`pre_order`](@ref) sizes its traversal stack as `2 * a.level`, so a depth would undersize it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ClusterNode(
        id,
        left::Option{<:ClusterNode} = nothing,
        right::Option{<:ClusterNode} = nothing,
        height::Number = 0.0,
        level::Int = 1
    ) -> ClusterNode

Arguments correspond to the struct's fields. A node given children ignores the `level` argument and takes `left.level + right.level` instead, so only a leaf's `level` comes from the caller.

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
  - [`to_tree`](@ref)
  - [`Clusters`](@ref)

# References

  - $(ref_dict[:virtanen2020])
  - $(ref_dict[:cajas2025]) Section 12.1.1, Equation 12.5.
"""
struct ClusterNode{tid, tl, tr, td, tcnt} <: AbstractResult
    """
    $(field_dict[:id_node])
    """
    id::tid
    """
    $(field_dict[:left_node])
    """
    left::tl
    """
    $(field_dict[:right_node])
    """
    right::tr
    """
    $(field_dict[:height_node])
    """
    height::td
    """
    $(field_dict[:level_node])
    """
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
"""
    const VecClN = AbstractVector{<:ClusterNode}

Alias for a vector of [`ClusterNode`](@ref) objects.

Represents an ordered collection of cluster nodes, typically used in hierarchical tree traversal and linkage computation.

# Related

  - [`ClusterNode`](@ref)
  - [`to_tree`](@ref)
  - [`pre_order`](@ref)
"""
const VecClN = AbstractVector{<:ClusterNode}
"""
    is_leaf(a::ClusterNode)

Is this node an asset, or a merge of two clusters?

Tests `left` alone. A [`ClusterNode`](@ref) is built with both children or with neither, so one test settles it.

# Arguments

  - `a`: The node to check.

# Returns

  - `flag::Bool`: `true` when the node has no children.

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
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preorder traversal strategies.

All concrete and/or abstract types implementing specific preorder traversal logic should be subtypes of `AbstractPreorderBy`.

A strategy decides **which property a leaf contributes** to [`pre_order`](@ref)'s output. It does not change the order of the walk, which is always left subtree before right.

# Interfaces

In order to implement a new traversal strategy that works seamlessly with the library, subtype `AbstractPreorderBy` and implement the following method:

## Required method

  - `get_node_property(preorder_by::MyPreorderBy, a::ClusterNode)`: Return the property that `a` contributes when it is reached as a leaf.

### Arguments

  - `preorder_by`: The concrete traversal strategy.
  - `a`: Node reached by the walk.

### Returns

  - The property to collect. It must be an `Int` today; see below.

# The property must be an `Int`

[`pre_order`](@ref) collects into a `Vector{Int}` and keys its two visited sets on the type of the property while storing the node's `id` in them. A strategy whose property is not an `Int` therefore fails: one returning a `Float64` **silently truncates** every collected value, and one returning a `String` raises `MethodError: Cannot convert an object of type Int64 to an object of type String`. [`PreorderTreeByID`](@ref) is the only strategy that ships, and its property is the node's `id`, so no shipped path meets this. Issue #510 carries the repair.

# Related

  - [`PreorderTreeByID`](@ref)
  - [`get_node_property`](@ref)
  - [`pre_order`](@ref)
"""
abstract type AbstractPreorderBy <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Collects each leaf's `id`, which for a leaf is its asset index.

The default strategy, and the only one that ships. [`to_tree`](@ref) numbers the leaves `1:N` in the order of the clustering's own asset axis, so a [`pre_order`](@ref) under this strategy returns asset indices ready to index a returns matrix with.

# Related

  - [`AbstractPreorderBy`](@ref)
  - [`get_node_property`](@ref)
  - [`pre_order`](@ref)
  - [`to_tree`](@ref)

# References

  - $(ref_dict[:virtanen2020])
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
    pre_order(a::ClusterNode, preorder_by::AbstractPreorderBy = PreorderTreeByID())

List the leaves below a node, left to right.

Walks the subtree rooted at `a` in preorder and collects one property per **leaf**; an internal node contributes nothing but the order it imposes on its two children. The property collected is [`get_node_property`](@ref)'s, so `preorder_by` is what a caller changes to collect something other than the node's `id`.

`preorder_by` is **positional**, not a keyword.

# Algorithm

 1. Open the stack `curNode`, sized at `2 * a.level`. [`ClusterNode`](@ref)'s `level` counts the leaves below the node, so the stack holds twice as many slots as the walk can ever need.
 2. Put `a` in the first slot, and open the two sets `lvisited` and `rvisited`, which record the internal nodes whose left and whose right child the walk has already pushed.
 3. Read the node `nd` on top of the stack.
 4. A leaf pushes [`get_node_property`](@ref)'s value onto `preorder` and is popped.
 5. An internal node outside `lvisited` pushes its `left` child and joins `lvisited`.
 6. An internal node inside `lvisited` and outside `rvisited` pushes its `right` child and joins `rvisited`.
 7. An internal node inside both sets is popped, because the walk below it is finished.
 8. Repeat from step 3 until the stack is empty, giving `preorder`, one property per leaf in left-to-right order.

# Arguments

  - `a`: Root node of the subtree to walk.
  - `preorder_by`: Traversal strategy, deciding which property each leaf contributes.

# Returns

  - `res::Vector{Int}`: One property per leaf, in left-to-right order. Its length is `a.level`.

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

Rebuild a linkage matrix as a tree of [`ClusterNode`](@ref) objects.

Reads a `Clustering.Hclust` from [`Clustering.jl`](https://juliastats.org/Clustering.jl/stable/hclust.html) and builds `2N - 1` nodes: one leaf per asset, numbered `1:N` in the clustering's own asset order, then one node per merge, numbered `N+1` upward in the order the merges happened. The last merge is therefore the root.

# Algorithm

 1. Read `N`, the number of assets, from the length of `a.order`, and open `d`, a vector of `2N - 1` nodes.
 2. Build one leaf per asset, `d[i] = ClusterNode(i)`, so the leaves carry the ids `1:N` in the clustering's own asset order.
 3. Walk the merges in the order `a.heights` gives them, which is the order they happened in.
 4. Resolve each side of row `i` of `a.merges` to an index into `d`: a **negative** entry `fi` names the asset `-fi`, and a **positive** entry names the merge `fi + N`.
 5. Build the merge node `ClusterNode(i + N, d[fi], d[fj], height)` and store it at `d[N + i]`, so the merge nodes carry the ids `N+1:2N-1` in merge order.
 6. Return the node built last, which is the root, together with `d`.

# Arguments

  - `a`: Hierarchical clustering object.

# Returns

  - `root::ClusterNode`: Root of the tree, which is the node of the last merge.
  - `nodes::Vector{ClusterNode}`: All `2N - 1` nodes, leaves first, then merges in merge order. The vector is **not** sorted by height; a caller that needs that ordering sorts it, as [`optimal_number_clusters`](@ref) does.

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
    clusterise(cle::ClustersEstimator{<:Any, <:Any, <:HClustAlgorithm, <:Any},
               X::MatNum; branchorder::Symbol = :optimal, dims::Int = 1,
               kwargs...)

Run hierarchical clustering and return the result as a [`Clusters`](@ref) object.

Estimates the similarity and distance matrices from `X`, runs the linkage `cle.alg` names, and cuts the dendrogram at the count `cle.onc` selects.

# Algorithm

 1. Estimate the similarity matrix `S` and the distance matrix `D` from `X` with [`cor_and_dist`](@ref), under `cle.de` and `cle.ce`.
 2. Cluster `D` with `Clustering.hclust` under the linkage `cle.alg.linkage` and the branch order `branchorder`, giving `res`, the dendrogram.
 3. Choose the number of clusters with [`optimal_number_clusters`](@ref)`(cle.onc, res, D)`, giving `k`.
 4. Return `Clusters(; res = res, S = S, D = D, k = k)`. `P` is left as `nothing`, because the clustering ran on `D` itself.

# Arguments

  - `cle`: Clustering estimator.
  - `X`: Data matrix (observations × assets).
  - `branchorder`: Branch ordering strategy for hierarchical clustering.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `res::Clusters`: Result object containing clustering, similarity, distance matrices, and number of clusters.

# Related

  - [`Clusters`](@ref)
  - [`ClustersEstimator`](@ref)
"""
function clusterise(cle::ClustersEstimator{<:Any, <:Any, <:HClustAlgorithm, <:Any},
                    X::MatNum; branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    res = Clustering.hclust(D; linkage = cle.alg.linkage, branchorder = branchorder)
    k = optimal_number_clusters(cle.onc, res, D)
    return Clusters(; res = res, S = S, D = D, k = k)
end
"""
    validate_k_value(res::Clustering.Hclust, nodes::VecClN, k::Integer)

Can this tree be cut into exactly `k` clusters?

Cuts the tree at `k`, then walks the `k - 1` tallest nodes. Each non-leaf node among them must have at least one of the `k` clusters contained wholly within one of its two subtrees; a node for which no cluster does makes `k` invalid.

# Only a tie in the heights makes the answer `false`

When every height is distinct, the `k - 1` tallest nodes **are** the merges that the cut removes, so each of them carries a whole cluster below one of its children and the answer is `true`. Two merges of equal height break that correspondence, because `nodes` is ordered by `sortperm` and the cut is not, so the walk can reach a node the cut left standing. Measured over 160 dendrograms with distinct heights — 40 universes of 16 assets under each of the four linkages — no `k` from `1` to `6` was ever rejected; on an eight-leaf balanced tree whose four lowest merges tie and whose two middle merges tie, `3`, `5`, `6` and `7` are all rejected.

`k = 1` is always valid: the walk of step 3 is then empty.

# Algorithm

 1. Cut the tree at `k` with `Clustering.cutree`, giving `idx`, one label per asset.
 2. Collect the assets carrying each label into `clusters`, one vector of asset indices per cluster.
 3. Walk the `k - 1` tallest entries of `nodes`, skipping any leaf among them.
 4. List the leaves below the node's `left` child and below its `right` child with [`pre_order`](@ref), giving `ln` and `rn`.
 5. Count the clusters of step 2 that lie wholly inside `ln` or wholly inside `rn`, giving `count`. A `count` of zero answers `false` at once.
 6. Answer `true` once every node of step 3 has carried at least one such cluster.

# Arguments

  - `res`: Hierarchical clustering object.
  - `nodes`: Vector of nodes in the clustering tree, sorted by descending height.
  - `k`: Number of clusters to validate.

# Returns

  - `flag::Bool`: `true` if `k` is a valid number of clusters, `false` otherwise.

# Related

  - [`optimal_number_clusters`](@ref)
  - [`ClusterNode`](@ref)
"""
function validate_k_value(res::Clustering.Hclust, nodes::VecClN, k::Integer)
    idx = Clustering.cutree(res; k = k)
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
    valid_k_clusters(res::Hclust, arr::VecNum)

Take the highest-scoring number of clusters the tree can be cut at.

Takes a candidate, tests it with [`validate_k_value`](@ref), and on failure blanks that entry to `typemin(eltype(arr))` and takes the next. The candidate is `argmax(arr)`, or `length(arr)` when no entry of `arr` is finite. It returns `1` instead when a rejected candidate leaves no finite entry behind.

!!! warning

    The scores are trusted as they arrive, and a `NaN` is not rejected. `argmax` returns the index of the first `NaN` in an array that carries one, ahead of every real score. An array that is `NaN` **throughout** takes the `length(arr)` branch instead, because a `NaN` is neither finite nor infinite, and answers `1` when the tree rejects that candidate. [`SecondOrderDifference`](@ref)'s default measure no longer produces such an array: ADR 0080 makes an undefined standard deviation divide by one, so a cluster of exactly two assets contributes its single pairwise distance rather than a `NaN`.

# Algorithm

 1. Rebuild the tree with [`to_tree`](@ref) and order its nodes by descending height, giving `nodes`.
 2. Take the candidate `k`: `argmax(arr)`, or `length(arr)` when no entry of `arr` is finite.
 3. Ask [`validate_k_value`](@ref) whether the tree can be cut at `k`, and return `k` when it can.
 4. Return `1` when step 3 rejected `k` and no entry of `arr` is finite, because blanking a non-finite entry offers the same candidate again.
 5. Write `typemin(eltype(arr))` into `arr[k]` and repeat from step 2.

The loop ends after at most `length(arr)` rejections. Step 5 blanks a rejected entry, so `argmax` never offers it twice; once every entry is `typemin` step 2 takes the `length(arr)` branch, and step 4 then answers `1`. The search usually ends far sooner, because `k = 1` is always a valid cut.

# Arguments

  - `res`: Hierarchical clustering object.
  - `arr`: Score for each candidate number of clusters. **Modified in place**: a candidate that fails validation is set to `typemin(eltype(arr))` so that the next iteration skips it. Both callers pass a local score array and neither reads it after the call, so the mutation reaches no caller.

# Returns

  - `k::Integer`: Valid number of clusters.

# Related

  - [`validate_k_value`](@ref)
  - [`optimal_number_clusters`](@ref)
"""
function valid_k_clusters(res::Clustering.Hclust, arr::VecNum)
    nodes = to_tree(res)[2]
    heights = [i.height for i in nodes]
    nodes = nodes[sortperm(heights; rev = true)]
    while true
        k = all(!isfinite, arr) ? length(arr) : argmax(arr)
        if validate_k_value(res, nodes, k)
            return k
            # No entry is finite, so `k` was `length(arr)` and blanking it changes nothing.
            # Answering `1` here is what ends the loop; `all(isinf, arr)` let a `NaN` array
            # re-offer the same rejected candidate for ever.
        elseif all(!isfinite, arr)
            return 1
        end
        arr[k] = typemin(eltype(arr))
    end
end
"""
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:Integer}, res::Hclust,
                            args...)
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference},
                            res::Hclust, D::MatNum)
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SilhouetteScore},
                            res::Hclust, D::MatNum)

Cut a dendrogram at the number of clusters `onc` selects.

Scores every candidate count up to the ceiling `onc` sets, then takes the highest-scoring count the **tree can actually be cut at**. A count no node of the dendrogram supports is rejected by [`validate_k_value`](@ref) and the next-highest score is tried, so the answer is the best *valid* count rather than the best score.

Every method returns a **bare** `k`. The non-hierarchical methods of the same name return the tuple `(res, k)` instead, because a flat partition cannot be re-cut and the clustering *is* the choice of `k`.

# Algorithm

The `Integer` method runs these steps. It is a **search**, not a test: an invalid stated count is replaced, never refused.

 1. Read the stated count `onc.alg` into `k`, and the ceiling into `max_k`. The ceiling is `min(floor(Int, sqrt(N)), onc.max_k)`, where `N` is the number of assets; a `max_k` of `nothing` leaves it at `floor(Int, sqrt(N))`.
 2. Lower `k` to `max_k` when it exceeds it.
 3. Rebuild the tree with [`to_tree`](@ref) and order its nodes by descending height, giving `nodes`.
 4. Ask [`validate_k_value`](@ref) whether the tree can be cut at `k`, and return `k` when it can.
 5. Search upward from `k + 1` to `max_k` for the first valid count, giving `ku` and its distance `du = ku - k`. Both stay at `k` and `0` when the search finds none.
 6. Search downward from `k - 1` to `1` for the first valid count, giving `kl` and its distance `dl = k - kl`. This search always succeeds when `k > 1`, because `k = 1` is always a valid cut.
 7. Take the count. When one side alone found one, take that side. When both found one and `du != dl`, take the nearer. When both found one and `du == dl`, take `ku` if `max_k - ku > kl - 1` and `kl` otherwise, so a tie goes to the side with more room left.

The `SecondOrderDifference` and `SilhouetteScore` methods run the steps their own algorithm types state. Both end by handing the score array to [`valid_k_clusters`](@ref), which walks down from the largest entry until the dendrogram admits the count.

# Arguments

  - `onc`: Optimal number of clusters estimator.

      + `onc::OptimalNumberClusters{<:Any, <:Integer}`: Takes the stated `k`, lowered to the ceiling. If that `k` is not valid, searches upward and downward for the nearest valid count and takes the nearer of the two; a tie goes to whichever side has more room left.
      + `onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference}`: Scores each count by the two-difference gap statistic of the within-cluster dispersions, then hands the scores to [`valid_k_clusters`](@ref). The dispersion is `onc.alg.alg` applied to one cluster's pairwise distances, summed over clusters.
      + `onc::OptimalNumberClusters{<:Any, <:SilhouetteScore}`: Scores each count by `onc.alg.alg` applied to the vector of per-asset silhouettes, then hands the scores to [`valid_k_clusters`](@ref).

  - `res`: Hierarchical clustering object.

  - `D`: Distance matrix the clustering was run on.

# Returns

  - `k::Integer`: Selected number of clusters, and always a count the dendrogram can be cut at.

# Related

  - [`OptimalNumberClusters`](@ref)
  - [`valid_k_clusters`](@ref)
  - [`validate_k_value`](@ref)
"""
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:Integer},
                                 res::Clustering.Hclust, args...)
    k = onc.alg
    max_k = onc.max_k
    N = length(res.order)
    if isnothing(max_k)
        max_k = floor(Int, sqrt(N))
    end
    max_k = min(floor(Int, sqrt(N)), max_k)
    if k > max_k
        k = max_k
    end
    nodes = to_tree(res)[2]
    heights = [i.height for i in nodes]
    nodes = nodes[sortperm(heights; rev = true)]
    flag = validate_k_value(res, nodes, k)
    if !flag
        # Above k
        flagu = false
        du = 0
        ku = k
        for i in (k + 1):max_k
            flagu = validate_k_value(res, nodes, i)
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
            flagl = validate_k_value(res, nodes, i)
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
                                 res::Clustering.Hclust, D::MatNum)
    N = size(D, 1)
    max_k = isnothing(onc.max_k) ? floor(Int, sqrt(N)) : onc.max_k
    c1 = min(min(floor(Int, sqrt(N)), max_k) + 2, N)
    cluster_lvls = [Clustering.cutree(res; k = k) for k in 1:c1]
    measure_alg = onc.alg.alg
    W_list = Vector{eltype(D)}(undef, c1)
    W_list[1] = typemin(eltype(D))
    for i in 2:c1
        lvl = cluster_lvls[i]
        c2 = maximum(unique(lvl))
        D_list = Vector{eltype(D)}(undef, c2)
        for j in 1:c2
            cluster = lvl .== j
            cluster_D = D[cluster, cluster]
            # No `isempty` guard. `Clustering.cutree` labels the assets `1:c2`, so no `j`
            # selects an empty submatrix; and an empty one would fall through to the
            # `isone(k)` arm below and contribute a zero, where the guard it replaced left
            # `D_list[j]` undefined for `sum` to read.
            M = size(cluster_D, 1)
            C_list = Vector{eltype(D)}(undef, Int(M * (M - 1) / 2))
            k = 1
            for col in 1:M
                for row in (col + 1):M
                    C_list[k] = cluster_D[row, col]
                    k += 1
                end
            end
            D_list[j] = if isone(k)
                zero(eltype(D))
            else
                vec_to_real_measure(measure_alg, C_list)
            end
        end
        W_list[i] = sum(D_list)
    end
    return if c1 > 2
        gaps = W_list[1:(end - 2)] + W_list[3:end] - 2 * W_list[2:(end - 1)]
        valid_k_clusters(res, gaps)
    else
        c1
    end
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SilhouetteScore},
                                 res::Clustering.Hclust, D::MatNum)
    N = size(D, 1)
    max_k = isnothing(onc.max_k) ? floor(Int, sqrt(N)) : onc.max_k
    c1 = min(floor(Int, sqrt(N)), max_k)
    cluster_lvls = [Clustering.cutree(res; k = i) for i in 1:c1]
    measure_alg = onc.alg.alg
    W_list = Vector{eltype(D)}(undef, c1)
    W_list[1] = typemin(eltype(D))
    for i in 2:c1
        sl = Clustering.silhouettes(cluster_lvls[i], D)
        W_list[i] = vec_to_real_measure(measure_alg, sl)
    end
    return valid_k_clusters(res, W_list)
end
function Clustering.assignments(clr::Clusters{<:Clustering.Hclust, <:Any, <:Any, <:Any})
    return Clustering.cutree(clr.res; k = clr.k)
end

export ClusterNode, is_leaf, PreorderTreeByID, pre_order, to_tree, optimal_number_clusters,
       assignments
