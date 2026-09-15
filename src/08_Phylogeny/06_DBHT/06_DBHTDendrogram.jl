"""
    DendroConstruct(Zi::MatNum, LabelVec1::VecNum,
                    LabelVec2::VecNum,
                    LinkageDist::Num_VecNum)

Construct the linkage matrix by continually adding rows to the matrix.

This function appends a new row to the linkage matrix at each iteration, recording the merge of clusters as indicated by changes in the label vectors. It is used internally for building dendrograms in DBHT and related hierarchical clustering routines.

# Algorithm

 1. Take `indx`, the vertices whose label differs between `LabelVec1` and `LabelVec2`. Those are the vertices the merge moved.
 2. Read the labels `LabelVec1` gave them, drop the repeats and sort them. A merge joins two labels, so exactly two survive.
 3. Append one row to `Zi`: those two labels, followed by `LinkageDist`.

# Arguments

  - `Zi`: `i × 3` linkage matrix at iteration `i` in the same format as the output from Matlab. Each row holds the two merged labels and the height of the merge.
  - `LabelVec1`: `N × 1` label vector for the vertices in the bubble for the previous valid iteration.
  - `LabelVec2`: `N × 1` label vector for the vertices in the bubble for the trial iteration.
  - `LinkageDist`: Height of the current merge, written into the third column.

# Returns

  - `Z::MatNum`: `(i + 1)×3` linkage matrix at iteration `i + 1` in the same format as the output from Matlab. [`turn_into_Hclust_merges`](@ref) converts it to the [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) convention.

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

This function scores every pair of clusters that `labelvec` names by the diameter of their union under the distance matrix `d`, and returns the pair of smallest score. The diameter is the largest non-zero distance inside the union, so the score is a complete linkage. Used internally for hierarchical linkage construction in DBHT dendrogram routines.

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

The union is scored, not the cut between the two clusters, so a distance between two members of the **same** cluster can set ``\\delta``. A pair whose union carries no non-zero distance scores ``0``, which is the smallest score there is.

# Algorithm

 1. Take `lvec`, the sorted distinct labels of `labelvec`.
 2. For each pair `(r, c)` of labels with `r < c`, select the vertices that carry either label.
 3. Take the largest non-zero entry of the distance submatrix on those vertices, and record `(lvec[r], lvec[c], value)` as a row of `Links`. Record `0` as the value when the submatrix carries no non-zero entry.
 4. Take the row of smallest value, giving the pair `PairLink` and the score `dvu`.

# Arguments

  - `d`: `Nv×Nv` distance matrix for the vertices assigned to a bubble. Row and column `i` are the same vertex, and entry `i` of `labelvec` names its cluster.
  - `labelvec`: `Nv×1` label vector for the vertices in the bubble.

# Returns

  - `PairLink::Vector{Int}`: `2 × 1` vector of the two cluster labels of the selected pair.
  - `dvu::Number`: Diameter of the union of that pair, the smallest such value over every pair.

# Related

  - [`build_link_and_dendro`](@ref)
  - [`HierarchyConstruct4s`](@ref)
"""
function LinkageFunction(d::MatNum, labelvec::VecNum)
    lvec = sort!(unique(labelvec))
    Links = Matrix{Int}(undef, 0, 3)
    for r in 1:(length(lvec) - 1)
        vecr = labelvec .== lvec[r]
        for c in (r + 1):length(lvec)
            vecc = labelvec .== lvec[c]
            x1 = vecr .|| vecc
            dd = d[x1, x1]
            de = dd[dd .!= 0]
            Link1 = if !isempty(de)
                hcat(lvec[r], lvec[c], vec(maximum(de; dims = 1)))
            else
                hcat(lvec[r], lvec[c], 0)
            end
            Links = vcat(Links, Link1)
        end
    end
    dvu, imn = findmin(Links[:, 3])
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

Iteratively construct the linkage matrix for a bubble or cluster.

This function iterates over the vertices in a bubble or cluster, merging the pair of clusters with the best linkage at each step (as determined by [`LinkageFunction`](@ref)), and appending the corresponding row to the linkage matrix using [`DendroConstruct`](@ref). Used internally for building dendrograms in DBHT and related hierarchical clustering routines.

# Algorithm

 1. Take the pair of smallest union diameter with [`LinkageFunction`](@ref) over `dpm` and `LabelVec`.
 2. Give both members of the pair the label `maximum(LabelVec1) + 1`, so the merged cluster takes a label no vertex carries yet.
 3. Write the merged labels back into `LabelVec2` at the vertices `V`.
 4. Append one row to `Z` with [`DendroConstruct`](@ref), at the height `1 / nc`.
 5. Subtract one from `nc`, and copy `LabelVec2` into `LabelVec1`.
 6. Repeat from step 1 once for each element of `rg`.

# Arguments

  - `rg`: Range whose **length** sets the number of merges. Its values are not read.
  - `dpm`: `Nv×Nv` distance matrix for the vertices assigned to the bubble or cluster, in the order of `V`.
  - `LabelVec`: `Nv×1` label vector of those vertices. It is mutated by step 2.
  - `LabelVec1`: `N × 1` label vector over every vertex, for the previous valid iteration.
  - `LabelVec2`: `N × 1` label vector over every vertex, for the trial iteration. It is mutated by step 3.
  - `V`: `Nv×1` vector of the indices of the vertices in the bubble or cluster, indexing the rows of `LabelVec1`.
  - `nc::Number`: Merge counter. Step 4 writes the height `1 / nc` and step 5 lowers it by one, so the heights of a run rise towards `1`.
  - `Z`: Current linkage matrix, with three columns.

# Returns

  - `Z::MatNum`: Linkage matrix after every merge of the range, one row longer per merge.
  - `nc::Number`: Merge counter, lowered by the number of merges.
  - `LabelVec1::VecNum`: `N × 1` label vector for the next iteration, carrying the merged labels.

# Related

  - [`LinkageFunction`](@ref)
  - [`DendroConstruct`](@ref)
  - [`HierarchyConstruct4s`](@ref)
"""
function build_link_and_dendro(rg::AbstractRange, dpm::MatNum, LabelVec::VecNum,
                               LabelVec1::VecNum, LabelVec2::VecNum, V::VecNum, nc::Number,
                               Z::MatNum)
    for _ in rg
        PairLink, dvu = LinkageFunction(dpm, LabelVec)  # Look for the pair of clusters which produces the best linkage
        LabelVec[LabelVec .== PairLink[1].||LabelVec .== PairLink[2]] .= maximum(LabelVec1) +
                                                                         1  # Merge the cluster pair by updating the label vector with a same label.
        LabelVec2[V] = LabelVec
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, 1 / nc)
        nc -= 1
        LabelVec1 = copy(LabelVec2)
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

Constructs the intra- and inter-cluster hierarchy by utilizing the Bubble Hierarchy structure of a Maximal Planar Graph, specifically a Planar Maximally Filtered Graph (PMFG).

This function builds a hierarchical clustering (dendrogram) by first constructing intra-cluster linkages within each cluster (using the bubble structure), and then merging clusters to form the global hierarchy. It is a core step in the DBHT (Direct Bubble Hierarchical Tree) clustering pipeline.

# Algorithm

 1. Give every vertex its own label in `LabelVec1`, and build `E`, the `N × maximum(Tc)` indicator of the discrete clustering `Tc`.
 2. For each cluster `k`, restrict `Mv` to the vertices of that cluster, giving `Mc`, and assign each of them to exactly one bubble with [`BubbleMember`](@ref), giving `Mvv`. Set the merge counter `nc` to the vertex count of the cluster less one.
 3. For each bubble of the cluster that holds more than one vertex, merge its vertices with [`build_link_and_dendro`](@ref) over `length(V) - 1` steps, on the distance submatrix of that bubble.
 4. Merge the bubbles of the cluster with [`build_link_and_dendro`](@ref) over `length(Bub) - 1` steps, on the distance submatrix of the whole cluster. Steps 3 and 4 share one `nc`, so the heights of a cluster rise across both.
 5. Repeat steps 2 to 4 for each cluster, which leaves one label per cluster.
 6. Merge the clusters over `length(kvec) - 1` steps: take the pair of smallest union diameter with [`LinkageFunction`](@ref) over the whole of `Dpm`, and give both sides a fresh label.
 7. Write the height of that merge from `dcl` and not from the score of step 6. `dcl` starts at `1` for every vertex and each merge sets both sides to the sum of the two, so the height counts the clusters the merge joins. This is what puts every inter-cluster merge above every intra-cluster one, whose heights never exceed `1`.

# Arguments

  - `Rpm`: `N × N` sparse weighted adjacency matrix of the PMFG. It is read by [`BubbleMember`](@ref) alone.
  - `Dpm`: `N × N` shortest path lengths matrix of the PMFG. Every linkage score is read from it.
  - `Tc`: `N × 1` cluster membership vector. `Tc[n] = k` indicates cluster membership of vertex `n` to the `k`-th discrete cluster.
  - `Mv`: `N × Nb` bubble membership matrix. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.

# Returns

  - `Z::MatNum`: `(N-1)×3` linkage matrix in the same format as the output from Matlab. Each row holds the two merged labels and the height of the merge. [`turn_into_Hclust_merges`](@ref) converts it to the [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) convention.

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
    E = SparseArrays.sparse(LabelVec1, Tc, ones(Int, N), N, maximum(Tc))
    Z = Matrix{Float64}(undef, 0, 3)

    # Intra-cluster hierarchy construction
    for n in eachindex(kvec)
        Mc = vec(E[:, kvec[n]]) ⊙ Mv   # Get the list of bubbles which coincide with nth cluster
        Mvv = BubbleMember(Rpm, Mv, Mc) # Assign each vertex in the nth cluster to a specific bubble
        Bub = findall(vec(sum(Mvv; dims = 1) .> 0)) # Get the list of bubbles which contain the vertices of nth cluster
        nc = sum(Tc .== kvec[n]) - 1

        # Apply the linkage within the bubbles.
        for m in eachindex(Bub)
            V = vec(findall(Mvv[:, Bub[m]] .!= 0)) # Retrieve the list of vertices assigned to mth bubble
            if length(V) > 1
                dpm = Dpm[V, V] # Retrieve the distance matrix for the vertices in V
                LabelVec = LabelVec1[V] # Initiate the label vector which labels for the clusters
                LabelVec2 = copy(LabelVec1)
                Z, nc, LabelVec1 = build_link_and_dendro(1:(length(V) - 1), dpm, LabelVec,
                                                         LabelVec1, LabelVec2, V, nc, Z)
            end
        end

        V = findall(E[:, kvec[n]] .!= 0)
        dpm = Dpm[V, V]

        # Perform linkage merging between the bubbles
        LabelVec = LabelVec1[V] # Initiate the label vector which labels for the clusters.
        LabelVec2 = copy(LabelVec1)
        Z, nc, LabelVec1 = build_link_and_dendro(1:(length(Bub) - 1), dpm, LabelVec,
                                                 LabelVec1, LabelVec2, V, nc, Z)
    end

    # Inter-cluster hierarchy construction
    LabelVec2 = copy(LabelVec1)
    dcl = ones(Int, length(LabelVec1))
    for _ in 1:(length(kvec) - 1)
        PairLink, dvu = LinkageFunction(Dpm, LabelVec1)
        LabelVec2[LabelVec1 .== PairLink[1].||LabelVec1 .== PairLink[2]] .= maximum(LabelVec1) +
                                                                            1
        dvu = unique(dcl[LabelVec1 .== PairLink[1]]) +
              unique(dcl[LabelVec1 .== PairLink[2]])
        dcl[LabelVec1 .== PairLink[1].||LabelVec1 .== PairLink[2]] .= dvu
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, dvu)
        LabelVec1 = copy(LabelVec2)
    end

    return Z
end
"""
    turn_into_Hclust_merges(Z::MatNum)

Convert a Matlab-style linkage matrix to a format compatible with [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

This function transforms a linkage matrix produced by DBHT or similar hierarchical clustering routines into the format required by [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust), including proper indexing and cluster size tracking.

**This is the seam to [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust), so the convention below is the one every consumer downstream reads.** [`DBHTs`](@ref) loads the first two columns into `mleft` and `mright` and the third into `heights`, and [`Clusters`](@ref) and everything that cuts a dendrogram reads them back on that convention.

Both conventions number one merge per row, and they differ in how a row names its two sides.

| The side is                  | Matlab writes             | `Clustering.Hclust` writes |
|:---------------------------- |:------------------------- |:-------------------------- |
| a leaf, vertex `a`           | `a`, which is at most `N` | `-a`                       |
| the cluster built by row `j` | `j + N`                   | `j`                        |

A row therefore names only rows above it, and the size of the cluster it builds is the sum of the sizes of its two sides.

# Algorithm

 1. Set `N` to `size(Z, 1) + 1`, which is the leaf count, and append a fourth column of zeros to hold the cluster sizes.
 2. For each row `i` and for each of its first two entries `a`: when `a` is at most `N` it names a leaf, so write `-a` in its place and add `1` to the size of row `i`.
 3. Otherwise `a` names the cluster that row `j = a - N` built, so write `j` in its place and add the size of row `j` to the size of row `i`. Row `j` is above row `i`, so its size is already final.

# Arguments

  - `Z`: `(N-1)×3` Matlab-style linkage matrix, where each row represents a merge step with cluster indices and linkage heights.

# Returns

  - `Z::MatNum`: `(N-1)×4` linkage matrix in [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) format. Columns one and two hold the two sides on the convention of the table above, column three keeps the heights unchanged, and column four holds the leaf count of the cluster each row builds.

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
