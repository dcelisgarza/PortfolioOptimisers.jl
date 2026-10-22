"""
$(DocStringExtensions.TYPEDEF)

Binds one merge of a dendrogram to the two clusters it joined.

It is the tree form of a linkage matrix. [`to_tree`](@ref) makes one node per asset and one per merge from a `Clustering.Hclust`, and the last node it makes is the root. A leaf has `nothing` in `left` and in `right`, and [`is_leaf`](@ref) tests that.

# `level` counts leaves and is not a depth

`level` is the number of assets in the subtree below the node. It is `1` for a leaf, and the sum of the counts of the two children for a merge. It is the fourth column of a linkage matrix and not a position in the tree. The root of a balanced tree of eight leaves has `level = 8`, and its depth is `3`.

[`pre_order`](@ref) sizes its traversal stack as `2 * a.level`. The stack holds the path from `a` to the node that the walk reads, and that path has at most `depth + 1 <= a.level` nodes.

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

Arguments correspond to the struct's fields. A node with children ignores the `level` argument and takes `left.level + right.level`, so only a leaf takes its `level` from the caller.

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

  - $(ref_dict[:virtanen2020]) The type ports `ClusterNode` and `to_tree` of `scipy.cluster.hierarchy`, which that work names and does not describe.
  - $(ref_dict[:cajas2025]) Section 12.1.1, Equation 12.5, for the four columns of a linkage matrix.
"""
@concrete struct ClusterNode <: AbstractResult
    """
    $(field_dict[:id_node])
    """
    id
    """
    $(field_dict[:left_node])
    """
    left
    """
    $(field_dict[:right_node])
    """
    right
    """
    $(field_dict[:height_node])
    """
    height
    """
    $(field_dict[:level_node])
    """
    level
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

[`to_tree`](@ref) returns one, with the leaves first and the merges after them. [`validate_k_value`](@ref) walks one sorted by descending height.

# Related

  - [`ClusterNode`](@ref)
  - [`to_tree`](@ref)
  - [`pre_order`](@ref)
"""
const VecClN = AbstractVector{<:ClusterNode}
"""
    is_leaf(a::ClusterNode)

Test whether a node is a leaf, which is the node of one asset.

It tests `left` alone. A [`ClusterNode`](@ref) has both children or neither, so one test is enough.

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

Abstract supertype for the strategies of a preorder traversal.

A strategy sets the property that each leaf adds to the output of [`pre_order`](@ref). It does not change the order of the walk, which visits the left subtree before the right one.

# Interfaces

To implement a new traversal strategy, subtype `AbstractPreorderBy` and implement the following method:

## Required method

  - `get_node_property(preorder_by::MyPreorderBy, a::ClusterNode)`: Return the property that `a` adds to the output when the walk reaches it as a leaf.

### Arguments

  - `preorder_by`: The concrete traversal strategy.
  - `a`: The node that the walk reached.

### Returns

  - The property to collect. Every leaf below one root must give a value of the same type.

# The property sets the element type of the walk

[`pre_order`](@ref) takes the element type of its output from the strategy. It reads `get_node_property(preorder_by, a)` at the root of the walk and collects into a vector of the type of that value, so a property of any type works. [`PreorderTreeByID`](@ref) is the one strategy of the library. Its property is the `id` of the node, so the default output is a `Vector{Int}`.

# Related

  - [`PreorderTreeByID`](@ref)
  - [`get_node_property`](@ref)
  - [`pre_order`](@ref)
"""
abstract type AbstractPreorderBy <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Collects the `id` of each leaf, which is the index of its asset.

It is the default strategy and the one strategy of the library. [`to_tree`](@ref) numbers the leaves `1:N` in the order of the asset axis of the clustering, so [`pre_order`](@ref) under this strategy returns asset indices that can index the columns of a returns matrix.

# Related

  - [`AbstractPreorderBy`](@ref)
  - [`get_node_property`](@ref)
  - [`pre_order`](@ref)
  - [`to_tree`](@ref)

# References

  - $(ref_dict[:virtanen2020]) The traversal ports `ClusterNode.pre_order` of `scipy.cluster.hierarchy`, which that work names and does not describe.
"""
struct PreorderTreeByID <: AbstractPreorderBy end
"""
    get_node_property(preorder_by::PreorderTreeByID, a::ClusterNode)

Return the property that a leaf adds to the output of a preorder traversal.

The [`PreorderTreeByID`](@ref) method returns the `id` of the node.

# Arguments

  - `preorder_by`: The traversal strategy.
  - `a`: The node.

# Returns

  - `id`: The identifier of the node.

# Related

  - [`PreorderTreeByID`](@ref)
  - [`pre_order`](@ref)
"""
get_node_property(::PreorderTreeByID, a::ClusterNode) = a.id

"""
    pre_order(a::ClusterNode, preorder_by::AbstractPreorderBy = PreorderTreeByID())

List the leaves below a node, left to right.

Walks the subtree below `a` in preorder and collects one property per leaf. An internal node adds no value. It only sets the order of its two children. [`get_node_property`](@ref) gives the property, so a caller changes `preorder_by` to collect a property other than the `id` of the node.

`preorder_by` is a positional argument, not a keyword.

# Algorithm

 1. Make the stack `curNode` with `2 * a.level` slots. The stack holds the path from `a` to the node that the walk reads. The `level` of a [`ClusterNode`](@ref) counts the leaves below it, so that path has at most `a.level` nodes.
 2. Put `a` in the first slot. Make the two sets `lvisited` and `rvisited`, which record the internal nodes whose left child and whose right child the walk pushed.
 3. Read the node `nd` on top of the stack.
 4. If `nd` is a leaf, push the value of [`get_node_property`](@ref) onto `preorder` and pop `nd`.
 5. If `nd` is an internal node outside `lvisited`, push its `left` child and add `nd` to `lvisited`.
 6. If `nd` is inside `lvisited` and outside `rvisited`, push its `right` child and add `nd` to `rvisited`.
 7. If `nd` is inside both sets, pop it, because the walk below it is finished.
 8. Repeat from step 3 until the stack is empty. The result is `preorder`, one property per leaf in left-to-right order.

# Arguments

  - `a`: The root of the subtree to walk.
  - `preorder_by`: The traversal strategy. It sets the property that each leaf adds.

# Returns

  - `res::Vector`: One property per leaf, in left-to-right order. Its length is `a.level`. The element type is the type of `get_node_property(preorder_by, a)`, so the default strategy gives a `Vector{Int}`.

# Related

  - [`ClusterNode`](@ref)
  - [`AbstractPreorderBy`](@ref)
  - [`PreorderTreeByID`](@ref)
  - [`get_node_property`](@ref)
"""
function pre_order(a::ClusterNode, preorder_by::AbstractPreorderBy = PreorderTreeByID())
    curNode = Vector{ClusterNode}(undef, 2 * a.level)
    lvisited = Set{typeof(a.id)}()
    rvisited = Set{typeof(a.id)}()
    curNode[1] = a
    k::Int = 1
    preorder = Vector{typeof(get_node_property(preorder_by, a))}(undef, 0)
    while k >= 1
        nd = curNode[k]
        ndid = nd.id
        if is_leaf(nd)
            push!(preorder, get_node_property(preorder_by, nd))
            k = k - one(k)
        elseif ndid ∉ lvisited
            k = k + one(k)
            curNode[k] = nd.left
            push!(lvisited, ndid)
        elseif ndid ∉ rvisited
            k = k + one(k)
            curNode[k] = nd.right
            push!(rvisited, ndid)
        else
            # Both children of this non-leaf node are visited, so go up in the tree.
            k = k - 1
        end
    end
    return preorder
end
"""
    to_tree(a::Hclust)

Rebuild a linkage matrix as a tree of [`ClusterNode`](@ref) objects.

Reads a `Clustering.Hclust` from [`Clustering.jl`](https://juliastats.org/Clustering.jl/stable/hclust.html) and makes `2N - 1` nodes. The `N` leaves come first, numbered `1:N` in the asset order of the clustering. One node per merge comes next, numbered from `N+1` in the order of the merges, so the node of the last merge is the root.

# Algorithm

 1. Read `N`, the number of assets, from the length of `a.order`. Make `d`, a vector of `2N - 1` nodes.
 2. Make one leaf per asset, `d[i] = ClusterNode(i)`, so the leaves have the ids `1:N` in the asset order of the clustering.
 3. Walk the merges in the order of `a.heights`, which is the order in which they happened.
 4. Resolve each side of row `i` of `a.merges` to an index into `d`. A negative entry `fi` names the asset `-fi`, and a positive entry names the merge `fi + N`.
 5. Make the merge node `ClusterNode(i + N, d[fi], d[fj], height)` and store it at `d[N + i]`, so the merge nodes have the ids `N+1:2N-1` in the order of the merges.
 6. Return the node made last, which is the root, together with `d`.

# Arguments

  - `a`: The dendrogram.

# Returns

  - `root::ClusterNode`: The root of the tree, which is the node of the last merge.
  - `nodes::Vector{ClusterNode}`: All `2N - 1` nodes, the leaves first and then the merges in their order. The vector is not sorted by height. A caller that needs that order sorts it, as [`valid_k_clusters`](@ref) and [`optimal_number_clusters`](@ref) do.

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
    join_leaf_orders!(tab::NamedTuple, A::AbstractVector{Int}, B::AbstractVector{Int},
                      D::MatNum)

Fill the least cost of each order of a merge from the least costs of the orders of its two children.

A merge joins the child with the leaves `A` to the child with the leaves `B`. An order of the merge starts at a leaf `i` of `A` and stops at a leaf `j` of `B`. The cost of an order is the sum of the distances between its adjacent leaves.

# Mathematical definition

```math
\\begin{align}
M(i, j) &= \\underset{k \\in A_i,\\, l \\in B_j}{\\min} M(i, k) + D_{kl} + M(l, j)\\,, \\\\
M(j, i) &= M(i, j)\\,.
\\end{align}
```

Where:

  - ``M(i, j)``: the least cost of an order of the leaves below a merge that starts at leaf ``i`` and stops at leaf ``j``, with ``M(i, i) = 0`` for a leaf.
  - ``A_i``: the leaves of ``A`` that can stop an order of ``A`` that starts at ``i``. For a child with one leaf it is ``\\{i\\}``. For a child that is a merge it holds the leaves on the other side of that merge from ``i``.
  - ``B_j``: the leaves of ``B`` that can start an order of ``B`` that stops at ``j``, by the same rule.
  - ``D_{kl}``: the distance between leaves ``k`` and ``l``.

# Algorithm

 1. For each leaf `i` of `A` and `l` of `B`, find the least `tab.M[i, k] + D[k, l]` over the leaves `k` of `A` that can stop an order that starts at `i`. Store the cost in `T[i, l]` and the leaf `k` in `K[i, l]`. Two leaves `i ≠ k` qualify when `tab.side` puts them on different sides of their child. The one leaf of a child with one leaf qualifies with itself.
 2. For each leaf `i` of `A` and `j` of `B`, find the least `T[i, l] + tab.M[l, j]` over the leaves `l` of `B` that qualify with `j` by the same rule. Store it in `tab.M[i, j]`.
 3. Write the same cost to `tab.M[j, i]`. Write the leaf `k` to `tab.S[i, j]` and the leaf `l` to `tab.S[j, i]`, so that the order from `i` to `j` is the order from `i` to `tab.S[i, j]`, followed by the order from `tab.S[j, i]` to `j`.

A tie goes to the first least value, so the result is deterministic.

# Arguments

  - `tab`: The tables of the dynamic programme. The function changes them in place.
      + `M`: The least cost of each pair of end leaves, ``M(i, j)`` above.
      + `S`: The leaf next to the split of each order, as step 3 states.
      + `side`: The side of its child that holds each leaf, `1` for the left side and `2` for the right side.
  - `A`, `B`: The leaves below the left child and below the right child of the merge.
  - `D`: The distance matrix.

# Returns

  - `tab::NamedTuple`: The same tables, with the pairs of `A` and `B` filled.

# Related

  - [`optimal_leaf_order!`](@ref)
"""
function join_leaf_orders!(tab::NamedTuple, A::AbstractVector{Int}, B::AbstractVector{Int},
                           D::MatNum)
    (; M, S, side) = tab
    T = similar(M, length(A), length(B))
    K = zeros(Int, length(A), length(B))
    for (b, l) in pairs(B), (a, i) in pairs(A), k in A
        if (side[i] != side[k] || isone(length(A))) &&
           (iszero(K[a, b]) || M[i, k] + D[k, l] < T[a, b])
            T[a, b] = M[i, k] + D[k, l]
            K[a, b] = k
        end
    end
    for j in B, (a, i) in pairs(A)
        best = 0
        for (b, l) in pairs(B)
            if (side[l] != side[j] || isone(length(B))) &&
               (iszero(best) || T[a, b] + M[l, j] < M[i, j])
                M[i, j] = T[a, b] + M[l, j]
                best = b
            end
        end
        M[j, i] = M[i, j]
        S[i, j] = K[a, best]
        S[j, i] = B[best]
    end
    return tab
end
"""
    trace_leaf_order(S::AbstractMatrix{Int}, i::Int, j::Int)

Rebuild the order from leaf `i` to leaf `j` from the split leaves that [`join_leaf_orders!`](@ref) wrote.

# Algorithm

 1. Start a stack that holds the pair `(i, j)`.
 2. Take the last pair `(a, b)` from the stack. When `a == b`, put the leaf `a` at the end of `order`.
 3. Otherwise, put `(S[b, a], b)` and then `(a, S[a, b])` on the stack, so that the order from `a` to `S[a, b]` comes first.
 4. Repeat from step 2 until the stack is empty.

The stack replaces a recursion, so a tree as deep as its count of leaves does not overflow the call stack.

# Arguments

  - `S`: The split leaves of each pair of end leaves.
  - `i`, `j`: The first and the last leaf of the order.

# Returns

  - `order::Vector{Int}`: The leaves, in the order from `i` to `j`.

# Related

  - [`join_leaf_orders!`](@ref)
  - [`optimal_leaf_order!`](@ref)
"""
function trace_leaf_order(S::AbstractMatrix{Int}, i::Int, j::Int)
    order = Int[]
    stack = [(i, j)]
    while !isempty(stack)
        a, b = pop!(stack)
        if a == b
            push!(order, a)
        else
            push!(stack, (S[b, a], b))
            push!(stack, (a, S[a, b]))
        end
    end
    return order
end
"""
    optimal_leaf_order!(ml::AbstractVector{Int}, mr::AbstractVector{Int}, D::MatNum)

Order the leaves of a dendrogram so that the sum of the distances between adjacent leaves is the least that the tree permits.

A flip of the two children of a merge keeps the tree and reverses the leaves below that merge. This method solves the problem of [barjoseph2001](@cite) exactly by dynamic programming. It finds the least sum over every combination of flips in ``O(N^3)`` time and ``O(N^2)`` memory. The algorithm of [barjoseph2001](@cite) runs in ``O(N^4)`` time, and Section 3.2 of [barjoseph2003](@cite) gives the ``O(N^3)`` form. Its temporary table, Equation 3, is the table `T` of [`join_leaf_orders!`](@ref). Both works maximise a sum of similarities, and this method minimises the same sum of distances. `Clustering.orderbranches_barjoseph!` is a heuristic from the same source. It decides each merge once, from its four outermost leaves, so it can miss the least sum.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\pi}^{\\star} &= \\underset{\\boldsymbol{\\pi} \\in \\Pi(\\mathcal{T})}{\\arg\\min} \\sum_{t=1}^{N-1} D_{\\pi_t \\pi_{t+1}}\\,.
\\end{align}
```

Where:

  - ``\\Pi(\\mathcal{T})``: the orders of the leaves that the dendrogram ``\\mathcal{T}`` permits, one for each combination of flips.
  - ``\\pi_t``: the leaf at position ``t`` of the order ``\\boldsymbol{\\pi}``.
  - ``D_{kl}``: the distance between leaves ``k`` and ``l``.
  - ``N``: the number of leaves.

# Algorithm

 1. Make the tables `tab`. They hold the costs `M`, which are zero for a leaf alone, the split leaves `S` and the sides `side`.
 2. Walk the merges in their order. For merge `v`, take the leaves `A` below `ml[v]` and `B` below `mr[v]`, and fill the costs of the pairs of `A` and `B` with [`join_leaf_orders!`](@ref). Then mark the leaves of `A` as side `1` and the leaves of `B` as side `2`.
 3. Take the pair `(i, j)` of the last merge with the least cost, with `i` below `ml[end]`.
 4. Rebuild the order `order` from `i` to `j` with [`trace_leaf_order`](@ref).
 5. Swap `ml[v]` and `mr[v]` for each merge whose left child does not come first in `order`.

# Arguments

  - `ml`, `mr`: The left and the right child of each merge, in the form of `Clustering.hclust`. An entry `-i` is the leaf `i`, a positive entry `v` is the merge `v`, and a merge comes after its children. Step 5 changes them in place.
  - `D`: The distance matrix of the leaves.

# Returns

  - `order::Vector{Int}`: The leaves in the order that step 4 rebuilds. It is the leaf order of `Clustering.Hclust` for the merges that step 5 writes.

# Related

  - [`branch_ordered_hclust`](@ref)
  - [`join_leaf_orders!`](@ref)
  - [`trace_leaf_order`](@ref)

# References

  - $(ref_dict[:barjoseph2001]) Sections "Problem definition" and "Algorithm", for the problem and the recurrence.
  - $(ref_dict[:barjoseph2003]) Section 3.2, Equations 2-4, for the ``O(N^3)`` time.
"""
function optimal_leaf_order!(ml::AbstractVector{Int}, mr::AbstractVector{Int}, D::MatNum)
    N = length(ml) + 1
    tab = (; M = zeros(eltype(D), N, N), S = zeros(Int, N, N), side = zeros(Int, N))
    leaves = Vector{Vector{Int}}(undef, length(ml))
    A = B = [1]
    for v in eachindex(ml, mr)
        A = ml[v] < 0 ? [-ml[v]] : leaves[ml[v]]
        B = mr[v] < 0 ? [-mr[v]] : leaves[mr[v]]
        join_leaf_orders!(tab, A, B, D)
        tab.side[A] .= 1
        tab.side[B] .= 2
        leaves[v] = vcat(A, B)
    end
    i, j = first(A), first(B)
    for jb in B, ia in A
        if tab.M[ia, jb] < tab.M[i, j]
            i, j = ia, jb
        end
    end
    order = trace_leaf_order(tab.S, i, j)
    pos = invperm(order)
    head = Vector{Int}(undef, length(ml))
    for v in eachindex(ml, mr)
        hl = ml[v] < 0 ? pos[-ml[v]] : head[ml[v]]
        hr = mr[v] < 0 ? pos[-mr[v]] : head[mr[v]]
        if hr < hl
            ml[v], mr[v] = mr[v], ml[v]
        end
        head[v] = min(hl, hr)
    end
    return order
end
"""
    branch_ordered_hclust(D::MatNum, linkage::Symbol, ::Val{:optimal})

Cluster `D` with `Clustering.hclust`, and order its leaves with the least sum of the distances between adjacent leaves that the tree permits.

# Algorithm

 1. Cluster `D` into `res` with `Clustering.hclust` under `linkage` and its branch order `:barjoseph`. The branch order flips merges and moves no merge, so the merges keep the order in which the linkage made them.
 2. Order the leaves of `res` into `order` with [`optimal_leaf_order!`](@ref) on the two columns of `res.merges` and on `D`.
 3. Return `Clustering.Hclust(res.merges, res.heights, order, res.linkage)`.

# Arguments

  - `D`: The distance matrix. The cost of an order is the sum of the entries of `D` itself, also when the linkage is `:ward`, which clusters the squares of `D`.
  - `linkage`: The linkage of `Clustering.hclust`.

# Returns

  - `res::Clustering.Hclust`: The dendrogram, with the leaf order of least cost.

# Related

  - [`optimal_leaf_order!`](@ref)
  - [`clusterise`](@ref)
"""
function branch_ordered_hclust(D::MatNum, linkage::Symbol, ::Val{:optimal})
    res = Clustering.hclust(D; linkage = linkage, branchorder = :barjoseph)
    order = optimal_leaf_order!(view(res.merges, :, 1), view(res.merges, :, 2), D)
    return Clustering.Hclust(res.merges, res.heights, order, res.linkage)
end
"""
    branch_ordered_hclust(D::MatNum, linkage::Symbol, ::Val{B}) where {B}

Cluster `D` with `Clustering.hclust` under `linkage` and its own branch order `B`.

`B = :barjoseph` gives the heuristic of `Clustering.orderbranches_barjoseph!`, and `B = :r` gives the order of the R function `hclust`. `Clustering.hclust` refuses any other value with an `ArgumentError`.

# Arguments

  - `D`: The distance matrix.
  - `linkage`: The linkage of `Clustering.hclust`.

# Returns

  - `res::Clustering.Hclust`: The dendrogram.

# Related

  - [`branch_ordered_hclust`](@ref)
  - [`clusterise`](@ref)
"""
function branch_ordered_hclust(D::MatNum, linkage::Symbol, ::Val{B}) where {B}
    return Clustering.hclust(D; linkage = linkage, branchorder = B)
end
"""
    clusterise(cle::ClustersEstimator{<:Any, <:Any, <:HClustAlgorithm, <:Any},
               X::MatNum; branchorder::Symbol = :optimal, dims::Int = 1,
               kwargs...)

Run hierarchical clustering and return the result as a [`Clusters`](@ref) object.

Estimates the similarity and distance matrices from `X`, makes a dendrogram under the linkage of `cle.alg`, and selects the number of clusters with `cle.onc`.

# Algorithm

 1. Estimate the similarity matrix `S` and the distance matrix `D` from `X` with [`cor_and_dist`](@ref), under `cle.de` and `cle.ce`.
 2. Cluster `D` into the dendrogram `res` under the linkage `cle.alg.linkage` with [`branch_ordered_hclust`](@ref) on `Val(branchorder)`.
 3. Select the number of clusters `k` with [`optimal_number_clusters`](@ref)`(cle.onc, res, D)`.
 4. Return `Clusters(; res = res, S = S, D = D, k = k)`. The field `P` keeps its default, `nothing`, because the clustering ran on `D` itself.

# Arguments

  - `cle`: The clustering estimator.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`.
  - `branchorder`: The order of the branches. `:optimal` orders the leaves with [`optimal_leaf_order!`](@ref). `Clustering.hclust` takes any other value, and it accepts `:barjoseph` and `:r`.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for [`cor_and_dist`](@ref).

# Returns

  - `res::Clusters`: The dendrogram, the similarity matrix, the distance matrix and the number of clusters.

# Related

  - [`Clusters`](@ref)
  - [`ClustersEstimator`](@ref)
"""
function clusterise(cle::ClustersEstimator{<:Any, <:Any, <:HClustAlgorithm, <:Any},
                    X::MatNum; branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    res = branch_ordered_hclust(D, cle.alg.linkage, Val(branchorder))
    k = optimal_number_clusters(cle.onc, res, D)
    return Clusters(; res = res, S = S, D = D, k = k)
end
"""
    validate_k_value(res::Clustering.Hclust, nodes::VecClN, k::Integer)

Test whether the dendrogram has a valid cut into `k` clusters.

Cuts the tree at `k`, then walks the `k - 1` tallest nodes. Each internal node among them must hold at least one of the `k` clusters wholly inside one of its two subtrees. A node that holds no such cluster makes `k` invalid.

# Only a tie in the heights makes the answer `false`

When every height is distinct, the `k - 1` tallest nodes are the merges that the cut removes. Each of them then holds a whole cluster below one of its children, and the answer is `true`. Two merges of equal height break that match. `sortperm` orders `nodes` and `Clustering.cutree` makes the cut, and the two can disagree inside a tie, so the walk can reach a merge that the cut keeps. A dendrogram with distinct heights rejects no `k`. A balanced tree of eight leaves, with four merges at height 1, two at height 2 and one at height 3, admits only `k` in 1, 2, 4 and 8.

`k = 1` is always valid, because the walk of step 3 is then empty.

# Algorithm

 1. Cut the tree at `k` with `Clustering.cutree`. The result is `idx`, one label per asset.
 2. Collect the assets of each label into `clusters`, one vector of asset indices per cluster.
 3. Walk the `k - 1` tallest entries of `nodes`, and skip a leaf among them.
 4. List the leaves below the `left` child and below the `right` child of the node with [`pre_order`](@ref). Call the two lists `ln` and `rn`.
 5. Count the clusters of step 2 that lie wholly inside `ln` or wholly inside `rn`. Answer `false` at once when the count is zero.
 6. Answer `true` when every node of step 3 holds at least one such cluster.

# Arguments

  - `res`: The dendrogram.
  - `nodes`: The nodes of the tree of `res`, sorted by descending height.
  - `k`: The number of clusters to test.

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

Take the number of clusters with the highest score at which the dendrogram has a valid cut.

The candidate is `argmax(arr)`, or `length(arr)` when no entry of `arr` is finite. When [`validate_k_value`](@ref) rejects a candidate, the function sets its score to `typemin(eltype(arr))` and takes the next candidate. When a rejection leaves no finite entry, the answer is `1`.

!!! warning

    The function does not check the scores, and it does not reject a `NaN`. In an array that holds a `NaN`, `argmax` returns the index of the first `NaN`, whatever the real scores are. An array of `NaN` alone has no finite entry, so it takes the `length(arr)` branch, and the answer is `1` when the tree rejects that candidate. The default measure of [`SecondOrderDifference`](@ref) gives no `NaN` for a cluster of two assets. The standard deviation of one value is undefined, so the measure divides by one, and the cluster contributes its one pairwise distance.

# Algorithm

 1. Rebuild the tree with [`to_tree`](@ref) and sort its nodes by descending height into `nodes`.
 2. Take the candidate `k`. It is `argmax(arr)`, or `length(arr)` when no entry of `arr` is finite.
 3. Ask [`validate_k_value`](@ref) whether the tree has a valid cut at `k`, and return `k` when it has.
 4. Return `1` when step 3 rejected `k` and no entry of `arr` is finite. A non-finite entry set to `typemin` stays non-finite, so step 2 would offer the same candidate again.
 5. Write `typemin(eltype(arr))` into `arr[k]` and repeat from step 2.

The loop ends after at most `length(arr)` rejections. Step 5 sets a rejected entry to `typemin`, so `argmax` does not offer it again while a finite entry is left. When no finite entry is left, step 2 takes the `length(arr)` branch, and step 4 answers `1` if the tree rejects it. A dendrogram with distinct heights rejects no candidate, so there the first candidate is the answer.

# Arguments

  - `res`: The dendrogram.
  - `arr`: The score of each candidate number of clusters. The function changes it in place. It sets the score of a rejected candidate to `typemin(eltype(arr))`, so that the next pass skips it. Both callers pass a local array of scores and do not read it after the call, so the change reaches no caller.

# Returns

  - `k::Integer`: A number of clusters at which the dendrogram has a valid cut.

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

Select the number of clusters at which to cut a dendrogram.

The `SecondOrderDifference` and `SilhouetteScore` methods score every candidate count up to the ceiling of `onc`. They take the count with the highest score at which the dendrogram has a valid cut. When [`validate_k_value`](@ref) rejects a count, the method tries the next highest score, so the answer is the best valid count and not always the best score.

Every method returns `k` alone. The non-hierarchical methods of the same name return the tuple `(res, k)`, because a flat partition cannot be cut again, so they return the partition that they made at `k`.

# Algorithm

The `Integer` method runs these steps. It replaces a stated count that is not valid with the nearest valid count, and it never refuses one.

 1. Read the stated count `onc.alg` into `k`, and the ceiling into `max_k`. The ceiling is `min(floor(Int, sqrt(N)), onc.max_k)`, where `N` is the number of assets. A `max_k` of `nothing` gives `floor(Int, sqrt(N))`.
 2. Lower `k` to `max_k` when it is larger.
 3. Rebuild the tree with [`to_tree`](@ref) and sort its nodes by descending height into `nodes`.
 4. Ask [`validate_k_value`](@ref) whether the tree has a valid cut at `k`, and return `k` when it has.
 5. Search upward from `k + 1` to `max_k` for the first valid count `ku`, at the distance `du = ku - k`. If the search finds none, `ku` stays at `k` and `du` at `0`.
 6. Search downward from `k - 1` to `1` for the first valid count `kl`, at the distance `dl = k - kl`. This search always finds one when `k > 1`, because `k = 1` is always a valid cut.
 7. Select the count. If one search alone found a count, take it. If both found one and `du != dl`, take the nearer. If both found one and `du == dl`, take `ku` when `max_k - ku > kl - 1` and `kl` otherwise. So a tie goes to the side with more candidate counts beyond the count it found.

The `SecondOrderDifference` and `SilhouetteScore` methods run the steps that their own algorithm types state. Both pass the array of scores to [`valid_k_clusters`](@ref), which takes the highest score at which the dendrogram has a valid cut. With two assets or fewer, the `SecondOrderDifference` method has no second-order difference to take and returns the number of assets.

# Arguments

  - `onc`: The estimator of the number of clusters.

      + `onc::OptimalNumberClusters{<:Any, <:Integer}`: Takes the stated `k`, lowered to the ceiling. If that `k` is not valid, it takes the nearest valid count above or below, as steps 5 to 7 state.
      + `onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference}`: Scores each count by the second-order difference of the within-cluster dispersions, then passes the scores to [`valid_k_clusters`](@ref). The dispersion of a cut is the sum over its clusters of `onc.alg.alg` applied to the pairwise distances of the cluster.
      + `onc::OptimalNumberClusters{<:Any, <:SilhouetteScore}`: Scores each count by `onc.alg.alg` applied to the vector of per-asset silhouettes, then passes the scores to [`valid_k_clusters`](@ref).

  - `res`: The dendrogram.

  - `D`: The distance matrix of the clustering.

# Returns

  - `k::Integer`: The selected number of clusters. The dendrogram always has a valid cut at it.

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
            C_list = Vector{eltype(D)}(undef, M * (M - 1) ÷ 2)
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
        gaps = @views W_list[1:(end - 2)] .+ W_list[3:end] .- 2 .* W_list[2:(end - 1)]
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
# The abstract type and the verb its `# Interfaces` section names (ADR 0154, issue #1131).
public AbstractPreorderBy, get_node_property
