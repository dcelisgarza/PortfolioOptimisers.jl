"""
    lance_williams_update(linkage::WeightedLinkage, dxi, dyi, dxy, nx::Integer, ny::Integer,
                          ni::Integer)
    lance_williams_update(linkage::CentroidLinkage, dxi, dyi, dxy, nx::Integer, ny::Integer,
                          ni::Integer)
    lance_williams_update(linkage::MedianLinkage, dxi, dyi, dxy, nx::Integer, ny::Integer,
                          ni::Integer)

Give the distance between the cluster `i` and the merge of the clusters `x` and `y`, under the update rule of a linkage.

Each member states its rule in its own docstring. The arithmetic is the arithmetic of the update rules of SciPy, in the same order, so the heights are equal to the heights of SciPy to the last bits. The centroid and the median rules square the distances inside and take the root of the result. A negative square, which only a distance that is not Euclidean can give, becomes zero.

# Arguments

  - `linkage`: The linkage.
  - `dxi`, `dyi`: The distances from `i` to `x` and from `i` to `y`.
  - `dxy`: The distance between `x` and `y`, which is the height of the merge.
  - `nx`, `ny`, `ni`: The sizes of the three clusters.

# Returns

  - `d`: The distance between `i` and the merge, of the type of the distances.

# Related

  - [`AbstractLinkage`](@ref)
  - [`generic_linkage_merges`](@ref)
  - [`nn_chain_merges`](@ref)
"""
function lance_williams_update(::WeightedLinkage, dxi, dyi, dxy, nx::Integer, ny::Integer,
                               ni::Integer)
    return (dxi + dyi) / 2
end
function lance_williams_update(::CentroidLinkage, dxi, dyi, dxy, nx::Integer, ny::Integer,
                               ni::Integer)
    v = (((nx * dxi * dxi) + (ny * dyi * dyi)) - (nx * ny * dxy * dxy) / (nx + ny)) /
        (nx + ny)
    return sqrt(max(v, zero(v)))
end
function lance_williams_update(::MedianLinkage, dxi, dyi, dxy, nx::Integer, ny::Integer,
                               ni::Integer)
    v = (dxi * dxi + dyi * dyi) / 2 - dxy * dxy / 4
    return sqrt(max(v, zero(v)))
end
"""
    older_first(a::Integer, b::Integer)

Order the two children of a merge so that the older child comes first.

The children are in the convention of `Clustering.Hclust`: `-i` is the leaf `i`, and a positive `v` is the merge `v`. A leaf is older than a merge, a leaf of lower index is older than a leaf of higher index, and a merge of lower index is older than a merge of higher index. This is the order of the SciPy identifiers, where the leaves are `0:(N - 1)` and the merges come after them, and SciPy puts the lower identifier first in each row.

# Arguments

  - `a`, `b`: The two children.

# Returns

  - `(first, second)::Tuple{Int, Int}`: The two children, the older one first.

# Related

  - [`generic_linkage_merges`](@ref)
  - [`nn_chain_merges`](@ref)
"""
function older_first(a::Integer, b::Integer)
    afirst = a < 0 ? (b > 0 || a > b) : (b > 0 && a < b)
    return afirst ? (a, b) : (b, a)
end
"""
    heap_swap!(heap::NamedTuple, i::Integer, j::Integer)

Swap the entries at the positions `i` and `j` of the binary heap of [`generic_linkage_merges`](@ref).

The heap holds one value for each key. `heap.vals[i]` is the value at position `i`, `heap.key[i]` is its key, and `heap.pos[k]` is the position of the key `k`. The function keeps the three vectors in step.

# Arguments

  - `heap`: The heap. The function changes it in place.
  - `i`, `j`: The two positions.

# Returns

  - `nothing`.

# Related

  - [`heap_sift_up!`](@ref)
  - [`heap_sift_down!`](@ref)
"""
function heap_swap!(heap::NamedTuple, i::Integer, j::Integer)
    (; vals, key, pos) = heap
    vals[i], vals[j] = vals[j], vals[i]
    key[i], key[j] = key[j], key[i]
    pos[key[i]] = i
    pos[key[j]] = j
    return nothing
end
"""
    heap_sift_up!(heap::NamedTuple, i::Integer)

Move the entry at position `i` of the binary heap up, until its parent holds a value that is not larger.

The parent of position `i` is position `i ÷ 2`. A tie stops the move, as in the heap of SciPy.

# Arguments

  - `heap`: The heap. The function changes it in place.
  - `i`: The position.

# Returns

  - `nothing`.

# Related

  - [`heap_swap!`](@ref)
  - [`heap_change!`](@ref)
"""
function heap_sift_up!(heap::NamedTuple, i::Integer)
    while i > 1 && heap.vals[i >> 1] > heap.vals[i]
        heap_swap!(heap, i, i >> 1)
        i >>= 1
    end
    return nothing
end
"""
    heap_sift_down!(heap::NamedTuple, i::Integer, n::Integer)

Move the entry at position `i` of the binary heap down, until no child of it holds a smaller value.

The heap holds the positions `1:n`. The children of position `i` are `2i` and `2i + 1`. Of two children with equal values, the move takes the first, and a tie with the entry stops the move, as in the heap of SciPy.

# Arguments

  - `heap`: The heap. The function changes it in place.
  - `i`: The position.
  - `n`: The number of entries in the heap.

# Returns

  - `nothing`.

# Related

  - [`heap_swap!`](@ref)
  - [`heap_change!`](@ref)
"""
function heap_sift_down!(heap::NamedTuple, i::Integer, n::Integer)
    vals = heap.vals
    c = 2 * i
    while c <= n
        if c + 1 <= n && vals[c + 1] < vals[c]
            c += 1
        end
        if !(vals[i] > vals[c])
            break
        end
        heap_swap!(heap, i, c)
        i = c
        c = 2 * i
    end
    return nothing
end
"""
    heap_change!(heap::NamedTuple, n::Integer, k::Integer, v)

Give the key `k` of the binary heap the value `v`, and restore the order of the heap.

A smaller value moves the entry up, and any other value moves it down.

# Arguments

  - `heap`: The heap. The function changes it in place.
  - `n`: The number of entries in the heap.
  - `k`: The key.
  - `v`: The new value.

# Returns

  - `nothing`.

# Related

  - [`heap_sift_up!`](@ref)
  - [`heap_sift_down!`](@ref)
  - [`generic_linkage_merges`](@ref)
"""
function heap_change!(heap::NamedTuple, n::Integer, k::Integer, v)
    i = heap.pos[k]
    old = heap.vals[i]
    heap.vals[i] = v
    if v < old
        heap_sift_up!(heap, i)
    else
        heap_sift_down!(heap, i, n)
    end
    return nothing
end
"""
    nearest_later_cluster(D::MatNum, sz::AbstractVector{<:Integer}, x::Integer)

Find the nearest cluster to the cluster `x` among the live clusters of a higher index.

A cluster is live when its size in `sz` is not zero. A tie goes to the lower index.

# Arguments

  - `D`: The distances between the clusters.
  - `sz`: The size of each cluster, zero for a cluster that a merge removed.
  - `x`: The cluster.

# Returns

  - `(y, d)`: The index `y` of the nearest cluster and its distance `d`. `y` is `0` and `d` is `typemax(eltype(D))` when no live cluster of a higher index exists.

# Related

  - [`generic_linkage_merges`](@ref)
"""
function nearest_later_cluster(D::MatNum, sz::AbstractVector{<:Integer}, x::Integer)
    y = 0
    d = typemax(eltype(D))
    for i in (x + 1):size(D, 1)
        if !iszero(sz[i]) && D[i, x] < d
            d = D[i, x]
            y = i
        end
    end
    return y, d
end
"""
    generic_linkage_merges(linkage::AbstractLinkage, D::MatNum)

Cluster `D` under any Lance-Williams linkage with the generic algorithm of [mullner2011](@cite).

The algorithm is correct for every update rule, also for a rule that is not reducible and gives inversions. It is a port of `fast_linkage` of `scipy.cluster.hierarchy`, with its binary heap, so the merges, their order and the rule for a tie are those of SciPy. The merges come out in the order in which they happen, and the heights are raw, so a merge can be lower than a merge below it.

# Algorithm

 1. Copy `D` into the working distances `dd`. Give each cluster `x < N` its nearest live cluster of a higher index `nb[x]` and its distance `md[x]` with [`nearest_later_cluster`](@ref). Put the `N - 1` values of `md` in a binary heap.
 2. For merge `k`, take the cluster `x` at the top of the heap and `y = nb[x]`. If `md[x]` is not the current distance between `x` and `y`, find the nearest cluster of `x` again, change its value in the heap, and take the top again. Stop after `N - k + 1` tries, as SciPy does.
 3. Remove `x` from the heap. Record the merge of `x` and `y` at the height `md[x]`, with the older child first ([`older_first`](@ref)). The cluster `y` now holds the merge, and `x` is removed.
 4. Update the distance from each live cluster `z` to `y` with [`lance_williams_update`](@ref).
 5. Move each candidate `nb[z] == x` with `z < x` to `y`. Lower `md[z]` to the new distance to `y` for each `z < y` where it is smaller. Find the nearest cluster of `y` again when `y < N`.
 6. Repeat from step 2 until one cluster remains.

# Arguments

  - `linkage`: The linkage. Its [`lance_williams_update`](@ref) updates the distances.
  - `D`: The distance matrix.

# Returns

  - `hmer::Clustering.HclustMerges`: The merges in the order in which they happen, with the raw heights.

# Related

  - [`AbstractLinkage`](@ref)
  - [`CentroidLinkage`](@ref)
  - [`MedianLinkage`](@ref)
  - [`linkage_merges`](@ref)

# References

  - $(ref_dict[:mullner2011]) Section 3.1, Figure 3.
  - $(ref_dict[:virtanen2020]) The kernel ports `fast_linkage` of `scipy.cluster.hierarchy`, which that work names and does not describe.
"""
function generic_linkage_merges(linkage::AbstractLinkage, D::MatNum)
    N = size(D, 1)
    dd = copyto!(Matrix{eltype(D)}(undef, size(D)), D)
    sz = ones(Int, N)
    nb = zeros(Int, N - 1)
    md = Vector{eltype(D)}(undef, N - 1)
    for x in 1:(N - 1)
        nb[x], md[x] = nearest_later_cluster(dd, sz, x)
    end
    heap = (; vals = copy(md), key = collect(1:(N - 1)), pos = collect(1:(N - 1)))
    for i in ((N - 1) >> 1):-1:1
        heap_sift_down!(heap, i, N - 1)
    end
    st = (; dd, sz, nb, md, heap)
    cid = collect(-1:-1:(-N))
    ml = Vector{Int}(undef, N - 1)
    mr = Vector{Int}(undef, N - 1)
    heights = similar(md)
    for k in 1:(N - 1)
        # The heap holds `N - k` keys before this merge.
        x, y, heights[k] = generic_linkage_pair!(st, N - k, N - k + 1)
        heap_swap!(heap, 1, N - k)
        heap_sift_down!(heap, 1, N - k - 1)
        ml[k], mr[k] = older_first(cid[x], cid[y])
        cid[y] = k
        lance_williams_merge!(linkage, dd, sz, x, y, heights[k])
        generic_linkage_neighbours!(st, N - k - 1, x, y)
    end
    return hclust_merges(ml, mr, heights)
end
"""
    generic_linkage_pair!(st::NamedTuple, hn::Integer, tries::Integer)

Find the next pair of clusters that the generic algorithm merges, step 2 of [`generic_linkage_merges`](@ref).

The cluster `x` at the top of the heap holds the least candidate distance `md[x]`, to its candidate `nb[x]`. A merge can make a candidate stale, so the function checks it against the current distance. A stale candidate is found again, its value in the heap changes, and the function takes the top again. It stops after `tries` tries, as SciPy does.

# Arguments

  - `st`: The state of the algorithm: the distances `dd`, the sizes `sz`, the candidates `nb` and `md`, and the `heap`. The function changes `nb`, `md` and `heap` in place.
  - `hn`: The number of entries in the heap.
  - `tries`: The most tries, `N - k + 1` for the merge `k`.

# Returns

  - `(x, y, dist)`: The two clusters and their distance, the height of the merge.

# Related

  - [`generic_linkage_merges`](@ref)
  - [`nearest_later_cluster`](@ref)
"""
function generic_linkage_pair!(st::NamedTuple, hn::Integer, tries::Integer)
    (; dd, sz, nb, md, heap) = st
    x = y = 0
    dist = zero(eltype(dd))
    for _ in 1:tries
        x, dist = heap.key[1], heap.vals[1]
        y = nb[x]
        if dist == dd[y, x]
            break
        end
        y, dist = nearest_later_cluster(dd, sz, x)
        nb[x] = y
        md[x] = dist
        heap_change!(heap, hn, x, dist)
    end
    return x, y, dist
end
"""
    lance_williams_merge!(linkage::AbstractLinkage, dd::MatNum,
                          sz::AbstractVector{<:Integer}, x::Integer, y::Integer, dist)

Merge the cluster `x` into the cluster `y`, and update the distances to `y` under the rule of `linkage`.

The distance from each live cluster `z` to the merge is [`lance_williams_update`](@ref) of the distances from `z` to `x` and to `y`. The cluster `y` then holds the merge, with the size `nx + ny`, and the size of `x` becomes zero. The two algorithms of the kernel share this step.

# Arguments

  - `linkage`: The linkage.
  - `dd`: The distances between the clusters. The function changes the row and the column of `y`.
  - `sz`: The size of each cluster. The function changes the sizes of `x` and `y`.
  - `x`, `y`: The two clusters of the merge.
  - `dist`: The distance between `x` and `y`.

# Returns

  - `nothing`.

# Related

  - [`generic_linkage_merges`](@ref)
  - [`nn_chain_merges`](@ref)
"""
function lance_williams_merge!(linkage::AbstractLinkage, dd::MatNum,
                               sz::AbstractVector{<:Integer}, x::Integer, y::Integer, dist)
    nx, ny = sz[x], sz[y]
    for z in axes(dd, 1)
        if !iszero(sz[z]) && z != x && z != y
            dd[z, y] = dd[y, z] = lance_williams_update(linkage, dd[z, x], dd[z, y], dist,
                                                        nx, ny, sz[z])
        end
    end
    sz[x] = 0
    sz[y] = nx + ny
    return nothing
end
"""
    generic_linkage_neighbours!(st::NamedTuple, hn::Integer, x::Integer, y::Integer)

Update the candidates of the generic algorithm after the merge of `x` into `y`, step 5 of [`generic_linkage_merges`](@ref).

# Algorithm

 1. Move each candidate `nb[z] == x` with `z < x` to `y`. This is a guess, which step 2 of the algorithm checks.
 2. For each live cluster `z < y` whose distance to `y` is less than `md[z]`, make `y` its candidate and change its value in the heap.
 3. When `y < N`, find the nearest live cluster of a higher index of `y` again with [`nearest_later_cluster`](@ref), and change its value in the heap.

# Arguments

  - `st`: The state of the algorithm, as [`generic_linkage_pair!`](@ref) states it. The function changes `nb`, `md` and `heap` in place.
  - `hn`: The number of entries in the heap.
  - `x`, `y`: The two clusters of the merge. `y` holds the merge.

# Returns

  - `nothing`.

# Related

  - [`generic_linkage_merges`](@ref)
"""
function generic_linkage_neighbours!(st::NamedTuple, hn::Integer, x::Integer, y::Integer)
    (; dd, sz, nb, md, heap) = st
    for z in 1:(x - 1)
        if !iszero(sz[z]) && nb[z] == x
            nb[z] = y
        end
    end
    for z in 1:(y - 1)
        if !iszero(sz[z]) && dd[z, y] < md[z]
            nb[z] = y
            md[z] = dd[z, y]
            heap_change!(heap, hn, z, dd[z, y])
        end
    end
    z, dz = nearest_later_cluster(dd, sz, y)
    if !iszero(z)
        nb[y] = z
        md[y] = dz
        heap_change!(heap, hn, y, dz)
    end
    return nothing
end
"""
    nn_chain_merges(linkage::AbstractLinkage, D::MatNum)

Cluster `D` under a reducible Lance-Williams linkage with the nearest-neighbour chain of [mullner2011](@cite).

The chain is correct only for a reducible linkage, which has no inversion. It is a port of `nn_chain` of `scipy.cluster.hierarchy`, so the merges, their order and the rule for a tie are those of SciPy.

# Algorithm

 1. Copy `D` into the working distances `dd`. Start a chain at the first live cluster when the chain is empty.
 2. Grow the chain with the nearest live cluster of its last cluster `x`. The cluster before `x` in the chain wins a tie, and else the lower index. Stop when the nearest cluster `y` of `x` is the cluster before `x`, so `x` and `y` are reciprocal nearest neighbours.
 3. Remove `x` and `y` from the chain. Record the merge of the lower index and the higher index at their distance. The cluster of the higher index now holds the merge.
 4. Update the distance from each live cluster to the merge with [`lance_williams_update`](@ref), and repeat from step 1 until one cluster remains.
 5. Sort the merges by height with a stable sort, and name the two children of each merge by the clusters that they hold, with the older child first. The sort gives the order of the merges, because the linkage is reducible.

# Arguments

  - `linkage`: The linkage. Its [`lance_williams_update`](@ref) updates the distances.
  - `D`: The distance matrix.

# Returns

  - `hmer::Clustering.HclustMerges`: The merges in the order of their heights.

# Related

  - [`AbstractLinkage`](@ref)
  - [`WeightedLinkage`](@ref)
  - [`linkage_merges`](@ref)

# References

  - $(ref_dict[:mullner2011]) Section 3.2, Figure 4.
  - $(ref_dict[:virtanen2020]) The kernel ports `nn_chain` of `scipy.cluster.hierarchy`, which that work names and does not describe.
"""
function nn_chain_merges(linkage::AbstractLinkage, D::MatNum)
    N = size(D, 1)
    dd = copyto!(Matrix{eltype(D)}(undef, size(D)), D)
    sz = ones(Int, N)
    chain = Vector{Int}(undef, N)
    len = 0
    zx = Vector{Int}(undef, N - 1)
    zy = Vector{Int}(undef, N - 1)
    heights = Vector{eltype(D)}(undef, N - 1)
    for k in 1:(N - 1)
        len, zx[k], zy[k], heights[k] = nn_chain_pair!(chain, len, dd, sz)
        lance_williams_merge!(linkage, dd, sz, zx[k], zy[k], heights[k])
    end
    return nn_chain_relabel(zx, zy, heights)
end
"""
    nn_chain_pair!(chain::AbstractVector{Int}, len::Integer, dd::MatNum,
                   sz::AbstractVector{<:Integer})

Grow the nearest-neighbour chain until its last two clusters are reciprocal nearest neighbours, steps 1 to 3 of [`nn_chain_merges`](@ref).

# Algorithm

 1. Start the chain at the first live cluster when it is empty.
 2. Find the nearest live cluster `y` of the last cluster `x` of the chain. The cluster before `x` in the chain wins a tie, and else the lower index.
 3. Stop when `y` is the cluster before `x`. Else put `y` at the end of the chain and repeat from step 2.

# Arguments

  - `chain`: The chain. The function changes it in place.
  - `len`: The length of the chain.
  - `dd`: The distances between the clusters.
  - `sz`: The size of each cluster, zero for a cluster that a merge removed.

# Returns

  - `(len, x, y, dist)`: The length of the chain without the two clusters, the lower and the higher index of the two clusters, and their distance.

# Related

  - [`nn_chain_merges`](@ref)
"""
function nn_chain_pair!(chain::AbstractVector{Int}, len::Integer, dd::MatNum,
                        sz::AbstractVector{<:Integer})
    if iszero(len)
        len = 1
        chain[1] = findfirst(!iszero, sz)
    end
    while true
        x = chain[len]
        prev = len > 1 ? chain[len - 1] : 0
        y, dist = nearest_live_cluster(dd, sz, x, prev)
        if len > 1 && y == prev
            a, b = minmax(x, y)
            return len - 2, a, b, dist
        end
        len += 1
        chain[len] = y
    end
end
"""
    nearest_live_cluster(dd::MatNum, sz::AbstractVector{<:Integer}, x::Integer,
                         prev::Integer)

Find the nearest live cluster to the cluster `x`, with the cluster `prev` first in a tie.

A cluster is live when its size in `sz` is not zero. The search starts from the distance to `prev`, so another cluster wins only when it is strictly nearer, and of two nearer clusters at one distance, the lower index wins. `prev = 0` starts the search from `typemax(eltype(dd))`.

# Arguments

  - `dd`: The distances between the clusters.
  - `sz`: The size of each cluster, zero for a cluster that a merge removed.
  - `x`: The cluster.
  - `prev`: The cluster that wins a tie, or `0`.

# Returns

  - `(y, d)`: The nearest cluster and its distance.

# Related

  - [`nn_chain_pair!`](@ref)
"""
function nearest_live_cluster(dd::MatNum, sz::AbstractVector{<:Integer}, x::Integer,
                              prev::Integer)
    y = prev
    d = iszero(prev) ? typemax(eltype(dd)) : dd[prev, x]
    for i in axes(dd, 1)
        if !iszero(sz[i]) && i != x && dd[i, x] < d
            d = dd[i, x]
            y = i
        end
    end
    return y, d
end
"""
    nn_chain_relabel(zx::AbstractVector{Int}, zy::AbstractVector{Int},
                     heights::AbstractVector)

Sort the merges of the nearest-neighbour chain by height, and name the children of each merge in the convention of `Clustering.Hclust`, step 5 of [`nn_chain_merges`](@ref).

The chain names each merge by the indices `zx < zy` of the two clusters that it joins. A union-find forest over the leaves `1:N` and the merges `N + k` replays the sorted merges, so each index becomes the cluster that it holds at that merge. This is `label` of `scipy.cluster.hierarchy`. A stable sort keeps the order of the chain in a tie, and a merge then comes after its children, because the linkage is reducible.

# Arguments

  - `zx`, `zy`: The lower and the higher index of each merge, in the order of the chain.
  - `heights`: The height of each merge, in the same order.

# Returns

  - `hmer::Clustering.HclustMerges`: The merges in the order of their heights, the older child of each merge first.

# Related

  - [`nn_chain_merges`](@ref)
  - [`cluster_root`](@ref)
"""
function nn_chain_relabel(zx::AbstractVector{Int}, zy::AbstractVector{Int},
                          heights::AbstractVector)
    N = length(heights) + 1
    o = sortperm(heights)
    parent = collect(1:(2 * N - 1))
    ml = Vector{Int}(undef, N - 1)
    mr = Vector{Int}(undef, N - 1)
    for (k, r) in pairs(o)
        a, b = minmax(cluster_root(parent, zx[r]), cluster_root(parent, zy[r]))
        parent[a] = parent[b] = N + k
        ml[k] = a <= N ? -a : a - N
        mr[k] = b <= N ? -b : b - N
    end
    return hclust_merges(ml, mr, heights[o])
end
"""
    cluster_root(parent::AbstractVector{Int}, a::Integer)

Find the root of the element `a` in a union-find forest.

# Arguments

  - `parent`: The parent of each element. A root is its own parent.
  - `a`: The element.

# Returns

  - `r::Int`: The root.

# Related

  - [`nn_chain_merges`](@ref)
  - [`cut_dendrogram`](@ref)
"""
function cluster_root(parent::AbstractVector{Int}, a::Integer)
    while parent[a] != a
        a = parent[a]
    end
    return a
end
"""
    linkage_merges(linkage::AbstractLinkage, D::MatNum)
    linkage_merges(linkage::WeightedLinkage, D::MatNum)

Cluster `D` under a linkage of the kernel of the library, and give the merges with their raw heights.

The default method runs [`generic_linkage_merges`](@ref), which is correct for every Lance-Williams linkage. [`WeightedLinkage`](@ref) is reducible, so its method runs the faster [`nn_chain_merges`](@ref).

# Arguments

  - `linkage`: The linkage.
  - `D`: The distance matrix.

# Returns

  - `hmer::Clustering.HclustMerges`: The merges, each after its two children, with the raw heights.

# Related

  - [`AbstractLinkage`](@ref)
  - [`branch_ordered_hclust`](@ref)
"""
function linkage_merges(linkage::AbstractLinkage, D::MatNum)
    return generic_linkage_merges(linkage, D)
end
function linkage_merges(linkage::WeightedLinkage, D::MatNum)
    return nn_chain_merges(linkage, D)
end
"""
    cophenetic_heights(heights::AbstractVector, ml::AbstractVector{<:Integer},
                       mr::AbstractVector{<:Integer})
    cophenetic_heights(res::Clustering.Hclust)

Give each merge of a dendrogram the largest merge height in its subtree.

This is the cophenetic height of the merge. It is the raw height on a tree with no inversion. On a tree with an inversion, a merge that is lower than a merge below it takes the height of that merge, so the cophenetic heights never fall from a child to its parent. Every reader that needs a monotone tree reads these heights, and `res.heights` keeps the raw heights.

# Mathematical definition

```math
\\begin{align}
c_v &= \\max\\left(h_v, c_{l(v)}, c_{r(v)}\\right)\\,.
\\end{align}
```

Where:

  - ``h_v``: the raw height of the merge ``v``.
  - ``l(v)``, ``r(v)``: the two children of ``v``. A leaf contributes no height.

# Arguments

  - `heights`: The raw height of each merge.
  - `ml`, `mr`: The left and the right child of each merge, in the convention of `Clustering.Hclust`. A merge comes after its children.
  - `res`: The dendrogram, which gives the three vectors above.

# Returns

  - `c::Vector`: The cophenetic height of each merge, in the order of the merges.

# Examples

```jldoctest
julia> using Clustering

julia> res = Clustering.Hclust([-1 -2; -3 1], [2.0, 1.5], [1, 2, 3], :centroid);

julia> cophenetic_heights(res)
2-element Vector{Float64}:
 2.0
 2.0
```

# Related

  - [`cut_dendrogram`](@ref)
  - [`cophenetic_order`](@ref)
  - [`AbstractLinkage`](@ref)
"""
function cophenetic_heights(heights::AbstractVector, ml::AbstractVector{<:Integer},
                            mr::AbstractVector{<:Integer})
    c = similar(heights)
    for v in eachindex(heights, ml, mr)
        h = heights[v]
        if ml[v] > 0
            h = max(h, c[ml[v]])
        end
        if mr[v] > 0
            h = max(h, c[mr[v]])
        end
        c[v] = h
    end
    return c
end
function cophenetic_heights(res::Clustering.Hclust)
    return cophenetic_heights(res.heights, view(res.merges, :, 1), view(res.merges, :, 2))
end
"""
    cophenetic_order(res::Clustering.Hclust)

Order the merges of a dendrogram by their cophenetic height, lowest first.

A tie goes to the earlier merge. So a parent always comes after its children, because the cophenetic height of a parent is at least the height of a child, and a parent is a later merge. A cut into `k` clusters does the first `N - k` merges of this order, and the last `k - 1` merges are the nodes above the cut. On a tree with no inversion, this is the order `sortperm(res.heights)` that `Clustering.cutree` reads.

# Arguments

  - `res`: The dendrogram.

# Returns

  - `order::Vector{Int}`: The indices of the merges.

# Related

  - [`cophenetic_heights`](@ref)
  - [`cut_dendrogram`](@ref)
"""
function cophenetic_order(res::Clustering.Hclust)
    return sortperm(cophenetic_heights(res))
end
"""
    cut_dendrogram(res::Clustering.Hclust, k::Integer)

Cut a dendrogram into `k` clusters, and label each asset with its cluster.

The cut does the first `N - k` merges of [`cophenetic_order`](@ref), so it removes the `k - 1` merges of largest cophenetic height, and a tie goes to the later merge. A parent comes after its children in that order, so the cut always gives exactly `k` clusters, also on a tree with an inversion. The labels are `1:k`, in the order of the lowest asset of each cluster.

On a tree with no inversion, the result is the result of `Clustering.cutree(res; k = k)`. On a tree with an inversion, `Clustering.cutree` replays a parent before its child and throws a `BoundsError`.

# Algorithm

 1. Make a union-find forest over the `N` assets.
 2. For each merge of the first `N - k` merges of [`cophenetic_order`](@ref), join the trees of one asset of each child.
 3. Walk the assets in order. Give the root of each asset a new label the first time that the walk reaches it, and give the asset the label of its root.

# Arguments

  - `res`: The dendrogram.
  - `k`: The number of clusters.

# Validation

  - `1 <= k <= N`. A `DomainError` names `k` otherwise.

# Returns

  - `idx::Vector{Int}`: One label per asset, over `1:k`.

# Examples

```jldoctest
julia> using Clustering

julia> res = Clustering.Hclust([-1 -2; -3 1], [2.0, 1.5], [1, 2, 3], :centroid);

julia> cut_dendrogram(res, 2)
3-element Vector{Int64}:
 1
 1
 2
```

# Related

  - [`cophenetic_order`](@ref)
  - [`cophenetic_heights`](@ref)
  - [`optimal_number_clusters`](@ref)
"""
function cut_dendrogram(res::Clustering.Hclust, k::Integer)
    N = length(res.order)
    @argcheck(one(k) <= k <= N,
              DomainError(k,
                          "a dendrogram of $N assets has a cut into `k` clusters for `1 <= k <= $N` alone."))
    # `leaf[v]` is one asset of the merge `v`, so a merge joins the trees of two assets.
    leaf = Vector{Int}(undef, N - 1)
    for v in eachindex(leaf)
        a = res.merges[v, 1]
        leaf[v] = a < 0 ? -a : leaf[a]
    end
    parent = collect(1:N)
    for v in view(cophenetic_order(res), 1:(N - k))
        a, b = res.merges[v, 1], res.merges[v, 2]
        ra = cluster_root(parent, a < 0 ? -a : leaf[a])
        rb = cluster_root(parent, b < 0 ? -b : leaf[b])
        parent[rb] = ra
    end
    label = zeros(Int, N)
    idx = Vector{Int}(undef, N)
    next = 0
    for i in 1:N
        r = cluster_root(parent, i)
        if iszero(label[r])
            next += 1
            label[r] = next
        end
        idx[i] = label[r]
    end
    return idx
end
"""
    tall_first_nodes(res::Clustering.Hclust)

Give the nodes of the tree of a dendrogram, from the tallest to the shortest.

[`validate_k_value`](@ref) walks the first `k - 1` of them. A node is tall by its cophenetic height. A tie between two nodes of equal cophenetic height and equal raw height goes to the earlier node, and a leaf is earlier than a merge. The walk then reaches a merge that the cut keeps when a tie of equal heights spans the cut, and that is how [`validate_k_value`](@ref) rejects such a cut. A merge whose raw height is less than its cophenetic height sits above an inversion. It comes before the merges of its tie, and the later one first, because that is the order of the cut. So an inversion alone rejects no cut.

On a tree with no inversion, the order is `sortperm` of the raw heights, from the tallest, with the earlier node first in a tie.

# Arguments

  - `res`: The dendrogram.

# Returns

  - `nodes::Vector{ClusterNode}`: The `2N - 1` nodes of [`to_tree`](@ref), in the order above.

# Related

  - [`validate_k_value`](@ref)
  - [`valid_k_clusters`](@ref)
  - [`cophenetic_heights`](@ref)
"""
function tall_first_nodes(res::Clustering.Hclust)
    nodes = to_tree(res)[2]
    N = length(res.order)
    c = cophenetic_heights(res)
    function rank(nd)
        if is_leaf(nd)
            return (zero(eltype(c)), false, -nd.id)
        end
        v = nd.id - N
        inverted = res.heights[v] < c[v]
        return (c[v], inverted, inverted ? nd.id : -nd.id)
    end
    return sort(nodes; by = rank, rev = true)
end

export cophenetic_heights, cut_dendrogram
