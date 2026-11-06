"""
    DendroConstruct(Zi::MatNum, LabelVec1::VecNum,
                    LabelVec2::VecNum,
                    LinkageDist::Num_VecNum)

Append one merge to a linkage matrix.

The function finds the merge from the change between two label vectors, and it appends one row for it to the linkage matrix. [`build_link_and_dendro`](@ref) and [`HierarchyConstruct4s`](@ref) call it once for each merge of the dendrogram.

# Algorithm

 1. Take `indx`, the vertices whose label is different in `LabelVec1` and `LabelVec2`. The merge moved those vertices.
 2. Read the labels that `LabelVec1` gave them, drop the repeats and sort them. A merge joins two labels, so exactly two labels remain.
 3. Append one row to `Zi`, which holds those two labels and then `LinkageDist`.

# Arguments

  - `Zi`: `i × 3` linkage matrix at iteration `i`, in the format of the Matlab output. Each row holds the two merged labels and the height of the merge.
  - `LabelVec1`: `N × 1` label vector of the vertices, for the last valid iteration.
  - `LabelVec2`: `N × 1` label vector of the vertices, for the trial iteration.
  - `LinkageDist`: Height of the current merge, which the function writes into the third column.

# Returns

  - `Z::MatNum`: `(i + 1)×3` linkage matrix at iteration `i + 1`, in the format of the Matlab output. [`turn_into_Hclust_merges`](@ref) converts it to the [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) convention.

# Related

  - [`HierarchyConstruct4s`](@ref)
  - [`turn_into_Hclust_merges`](@ref)
"""
function DendroConstruct(Zi::MatNum, LabelVec1::VecNum, LabelVec2::VecNum,
                         LinkageDist::Num_VecNum)
    indx = LabelVec1 .!= LabelVec2
    Z = vcat(Zi, hcat(transpose(sort!(unique(LabelVec1[indx]))), LinkageDist))
    return Z
end
"""
    LinkageFunction(d::MatNum, labelvec::VecNum)

Find the pair of clusters of smallest union diameter in a bubble.

The function scores every pair of clusters that `labelvec` names by the diameter of their union under the distance matrix `d`, and it returns the pair of smallest score. The diameter is the largest non-zero distance inside the union, so the score is a complete linkage. [`build_link_and_dendro`](@ref) and [`HierarchyConstruct4s`](@ref) call it to select each merge.

# Mathematical definition

```math
\\begin{align}
\\delta(I,\\, J) &= \\underset{u,\\,v \\in \\mathcal{C}_{I} \\cup \\mathcal{C}_{J},\\; d_{u,\\,v} \\neq 0}{\\max}\\; d_{u,\\,v}\\,, \\\\
(I^{\\star},\\, J^{\\star}) &= \\underset{I < J}{\\arg\\min}\\; \\delta(I,\\, J)\\,.
\\end{align}
```

Where:

  - ``d_{u,\\,v}``: Distance between vertices ``u`` and ``v``, the entry of the input matrix.
  - ``\\mathcal{C}_{I}``: Vertices that carry label ``I``.
  - ``\\delta(I,\\, J)``: Diameter of the union of the two clusters.

The function scores the union, not the cut between the two clusters, so a distance between two members of one cluster can set ``\\delta``. A pair whose union has no non-zero distance scores ``0``, the smallest possible score.

# Algorithm

 1. Take `lvec`, the sorted distinct labels of `labelvec`.
 2. For each pair `(r, c)` of labels with `r < c`, select the vertices that have either label.
 3. Take the largest non-zero entry of the distance submatrix on those vertices, and record `(lvec[r], lvec[c], value)` as a row of `Links`. Record `0` as the value when the submatrix has no non-zero entry.
 4. Take the row of smallest value, which gives the pair `PairLink` and the score `dvu`.

# Arguments

  - `d`: `Nv×Nv` distance matrix of the vertices of a bubble, a cluster or the graph. Row and column `i` are the same vertex, and entry `i` of `labelvec` names its cluster.
  - `labelvec`: `Nv×1` label vector of the same vertices.

# Returns

  - `PairLink::Vector{Int}`: `2 × 1` vector of the two cluster labels of the selected pair.
  - `dvu::Number`: Diameter of the union of that pair, the smallest such value over every pair.

# Related

  - [`build_link_and_dendro`](@ref)
  - [`HierarchyConstruct4s`](@ref)
"""
function LinkageFunction(d::MatNum, labelvec::VecNum)
    lvec = sort!(unique(labelvec))
    nl = length(lvec)
    Links = Matrix{promote_type(eltype(lvec), eltype(d))}(undef, nl * (nl - 1) ÷ 2, 3)
    k = 0
    for r in 1:(nl - 1)
        vecr = labelvec .== lvec[r]
        for c in (r + 1):nl
            vecc = labelvec .== lvec[c]
            x1 = vecr .|| vecc
            dd = d[x1, x1]
            de = dd[dd .!= 0]
            k += 1
            Links[k, 1] = lvec[r]
            Links[k, 2] = lvec[c]
            Links[k, 3] = if !isempty(de)
                maximum(de)
            else
                0
            end
        end
    end
    dvu, imn = findmin(view(Links, :, 3))
    PairLink = Links[imn, 1:2]
    return PairLink, dvu
end
"""
    build_link_and_dendro(
        rg::AbstractRange,
        dpm::MatNum,
        LabelVec::VecNum,
        LabelVec1::VecNum,
        LabelVec2::VecNum,
        V::VecNum,
        nc::Number,
        Z::MatNum
    )

Merge the clusters of a bubble or a cluster one pair at a time, and append each merge to the linkage matrix.

At each step the function merges the pair of clusters that [`LinkageFunction`](@ref) selects, and it appends the row of the merge to the linkage matrix with [`DendroConstruct`](@ref). [`HierarchyConstruct4s`](@ref) calls it for each bubble and for each cluster.

# Algorithm

 1. Take the pair of smallest union diameter with [`LinkageFunction`](@ref) over `dpm` and `LabelVec`.
 2. Give both members of the pair the label `maximum(LabelVec1) + 1`, so the merged cluster takes a label no vertex carries yet.
 3. Write the merged labels back into `LabelVec2` at the vertices `V`.
 4. Append one row to `Z` with [`DendroConstruct`](@ref), at the height `1 / nc`.
 5. Subtract one from `nc`, and copy `LabelVec2` into `LabelVec1`.
 6. Repeat from step 1 once for each element of `rg`.

# Arguments

  - `rg`: Range whose length sets the number of merges. The function does not read its values.
  - `dpm`: `Nv×Nv` distance matrix of the vertices of the bubble or cluster, in the order of `V`.
  - `LabelVec`: `Nv×1` label vector of those vertices. Step 2 changes it in place.
  - `LabelVec1`: `N × 1` label vector of every vertex, for the last valid iteration.
  - `LabelVec2`: `N × 1` label vector of every vertex, for the trial iteration. Step 3 changes it in place.
  - `V`: `Nv×1` vector of the vertices of the bubble or cluster, which index the rows of `LabelVec1`.
  - `nc::Number`: Merge counter. Step 4 writes the height `1 / nc` and step 5 lowers it by one, so the heights of a run rise towards `1`.
  - `Z`: Current linkage matrix, with three columns.

# Returns

  - `Z::MatNum`: Linkage matrix after every merge of the range, one row longer per merge.
  - `nc::Number`: Merge counter, lowered by the number of merges.
  - `LabelVec1::VecNum`: `N × 1` label vector for the next iteration, with the merged labels.

# Related

  - [`LinkageFunction`](@ref)
  - [`DendroConstruct`](@ref)
  - [`HierarchyConstruct4s`](@ref)
"""
function build_link_and_dendro(rg::AbstractRange, dpm::MatNum, LabelVec::VecNum,
                               LabelVec1::VecNum, LabelVec2::VecNum, V::VecNum, nc::Number,
                               Z::MatNum)
    LabelVecPrev = similar(LabelVec2)
    for _ in rg
        PairLink, dvu = LinkageFunction(dpm, LabelVec)  # Look for the pair of clusters which produces the best linkage
        LabelVec[LabelVec .== PairLink[1].||LabelVec .== PairLink[2]] .= maximum(LabelVec1) +
                                                                         1  # Merge the cluster pair by updating the label vector with a same label.
        LabelVec2[V] = LabelVec
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, 1 / nc)
        nc -= 1
        LabelVec1 = copyto!(LabelVecPrev, LabelVec2)
    end
    return Z, nc, LabelVec1
end
"""
    HierarchyConstruct4s(
        Rpm::MatNum,
        Dpm::MatNum,
        Tc::VecNum,
        Mv::MatNum
    )

Build the dendrogram of the Direct Bubble Hierarchical Tree (DBHT), first inside each cluster and then across the clusters.

The function first merges the vertices inside each discrete cluster, bubble by bubble, on the bubble tree of the Planar Maximally Filtered Graph (PMFG). It then merges the clusters into one hierarchy.

# Algorithm

 1. Give every vertex its own label in `LabelVec1`, and build `E`, the `N × maximum(Tc)` indicator of the discrete clustering `Tc`.
 2. For each cluster `k`, restrict `Mv` to the vertices of that cluster, which gives `Mc`. Assign each of them to exactly one bubble with [`BubbleMember`](@ref), which gives `Mvv`. Set the merge counter `nc` to the vertex count of the cluster minus one.
 3. For each bubble of the cluster that holds more than one vertex, merge its vertices with [`build_link_and_dendro`](@ref) over `length(V) - 1` steps, on the distance submatrix of that bubble.
 4. Merge the bubbles of the cluster with [`build_link_and_dendro`](@ref) over `length(Bub) - 1` steps, on the distance submatrix of the whole cluster. Steps 3 and 4 share one `nc`, so the heights of a cluster rise across both steps.
 5. Repeat steps 2 to 4 for each cluster, which leaves one label for each cluster.
 6. Merge the clusters over `length(kvec) - 1` steps. At each step, take the pair of smallest union diameter with [`LinkageFunction`](@ref) over all of `Dpm`, and give both sides a new label.
 7. Write the height of that merge from `dcl`, not from the score of step 6. `dcl` starts at `1` for every vertex, and each merge sets both sides to the sum of the two, so the height counts the clusters that the merge joins. Thus every merge across clusters is above every merge inside a cluster, whose height is never more than `1`.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG. Only [`BubbleMember`](@ref) reads it.
  - `Dpm`: `N × N` matrix of the shortest path lengths of the PMFG. Each linkage score comes from it.
  - `Tc`: `N × 1` cluster membership vector. `Tc[n] = k` means that vertex `n` belongs to the `k`-th discrete cluster.
  - `Mv`: `N × Nb` bubble membership matrix. `Mv[n, bi] = 1` means that vertex `n` is a vertex of bubble `bi`.

# Returns

  - `Z::MatNum`: `(N-1)×3` linkage matrix in the format of the Matlab output. Each row holds the two merged labels and the height of the merge. [`turn_into_Hclust_merges`](@ref) converts it to the [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) convention.

# Related

  - [`BubbleMember`](@ref)
  - [`build_link_and_dendro`](@ref)
  - [`turn_into_Hclust_merges`](@ref)
  - [`DBHT`](@ref)
"""
function HierarchyConstruct4s(Rpm::MatNum, Dpm::MatNum, Tc::VecNum, Mv::MatNum)
    N = size(Dpm, 1)
    kvec = sort!(unique(Tc))
    LabelVec1 = collect(1:N)
    LabelVec2 = similar(LabelVec1)
    E = SparseArrays.sparse(LabelVec1, Tc, ones(Int, N), N, maximum(Tc))
    Z = Matrix{Float64}(undef, 0, 3)

    # Intra-cluster hierarchy construction
    for n in eachindex(kvec)
        Mc = vec(E[:, kvec[n]]) ⊙ Mv   # Get the list of bubbles which coincide with nth cluster
        Mvv = BubbleMember(Rpm, Mv, Mc) # Assign each vertex in the nth cluster to a specific bubble
        Bub = findall(vec(sum(Mvv; dims = 1) .> 0)) # Get the list of bubbles which contain the vertices of nth cluster
        nc = count(==(kvec[n]), Tc) - 1

        # Apply the linkage within the bubbles.
        for m in eachindex(Bub)
            V = vec(findall(Mvv[:, Bub[m]] .!= 0)) # Retrieve the list of vertices assigned to mth bubble
            if length(V) > 1
                dpm = Dpm[V, V] # Retrieve the distance matrix for the vertices in V
                LabelVec = LabelVec1[V] # Initiate the label vector which labels for the clusters
                copyto!(LabelVec2, LabelVec1)
                Z, nc, LabelVec1 = build_link_and_dendro(1:(length(V) - 1), dpm, LabelVec,
                                                         LabelVec1, LabelVec2, V, nc, Z)
            end
        end

        V = findall(E[:, kvec[n]] .!= 0)
        dpm = Dpm[V, V]

        # Perform linkage merging between the bubbles
        LabelVec = LabelVec1[V] # Initiate the label vector which labels for the clusters.
        copyto!(LabelVec2, LabelVec1)
        Z, nc, LabelVec1 = build_link_and_dendro(1:(length(Bub) - 1), dpm, LabelVec,
                                                 LabelVec1, LabelVec2, V, nc, Z)
    end

    # Inter-cluster hierarchy construction
    copyto!(LabelVec2, LabelVec1)
    dcl = ones(Int, length(LabelVec1))
    for _ in 1:(length(kvec) - 1)
        PairLink, dvu = LinkageFunction(Dpm, LabelVec1)
        LabelVec2[LabelVec1 .== PairLink[1].||LabelVec1 .== PairLink[2]] .= maximum(LabelVec1) +
                                                                            1
        dvu = unique(dcl[LabelVec1 .== PairLink[1]]) +
              unique(dcl[LabelVec1 .== PairLink[2]])
        dcl[LabelVec1 .== PairLink[1].||LabelVec1 .== PairLink[2]] .= dvu
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, dvu)
        copyto!(LabelVec1, LabelVec2)
    end

    return Z
end
"""
    turn_into_Hclust_merges(Z::MatNum)

Convert a linkage matrix in the Matlab convention to the convention of [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

The function writes the linkage matrix of [`HierarchyConstruct4s`](@ref) in the index convention of [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust), and it adds the size of the cluster of each merge.

At this function the result passes to [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust), so every consumer downstream reads the convention below. [`DBHTs`](@ref) loads the first two columns into `mleft` and `mright` and the third into `heights`. [`Clusters`](@ref) and each function that cuts a dendrogram read them on that convention.

Both conventions give one merge to each row, and they name the two sides of a row in different ways.

| The side is                  | Matlab writes             | `Clustering.Hclust` writes |
|:---------------------------- |:------------------------- |:-------------------------- |
| a leaf, vertex `a`           | `a`, which is at most `N` | `-a`                       |
| the cluster built by row `j` | `j + N`                   | `j`                        |

Thus a row names only rows above it, and the size of the cluster that it builds is the sum of the sizes of its two sides.

# Algorithm

 1. Set `N` to `size(Z, 1) + 1`, the leaf count, and append a fourth column of zeros to hold the cluster sizes.
 2. For each row `i`, read each of its first two entries `a`. When `a` is at most `N`, it names a leaf. Write `-a` in its place and add `1` to the size of row `i`.
 3. Otherwise `a` names the cluster that row `j = a - N` built. Write `j` in its place and add the size of row `j` to the size of row `i`. Row `j` is above row `i`, so its size is already final.

# Arguments

  - `Z`: `(N-1)×3` linkage matrix in the Matlab convention. Each row is one merge, with the two merged labels and the height of the merge.

# Returns

  - `Z::MatNum`: `(N-1)×4` linkage matrix in the format of [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust). Columns one and two hold the two sides in the convention of the table above. Column three holds the heights with no change, and column four holds the leaf count of the cluster that each row builds.

# Related

  - [`HierarchyConstruct4s`](@ref)
  - [`DendroConstruct`](@ref)
  - [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust)
"""
function turn_into_Hclust_merges(Z::MatNum)
    N = size(Z, 1) + 1
    Z = hcat(Z, zeros(eltype(Z), N - 1))

    for i in axes(Z, 1)

        # Cluster indices.
        a = Int(Z[i, 1])
        b = Int(Z[i, 2])

        # If the cluster index is less than N, it represents a leaf,
        # so only one add one to the count.
        if a <= N
            Z[i, 1] = -a
            Z[i, 4] += 1
        else
            # Clusters in index Z[i, 1:2] are combined to form cluster i + N.
            # If a cluster has index a > N, it's a combined cluster.
            # The index of the child is j = a - N, so we need to go to index j
            # which is being combined by cluster a, get the count at index j
            # and add it to the count at index i, which contains cluster a.
            j = a - N
            Z[i, 1] = j
            Z[i, 4] += Z[j, 4]
        end

        if b <= N
            Z[i, 2] = -b
            Z[i, 4] += 1
        else
            # Do the same with the other side of the cluster, to wherever that side leads.
            j = b - N
            Z[i, 2] = j
            Z[i, 4] += Z[j, 4]
        end
    end
    return Z
end
