"""
    DBHTs(D::MatNum, S::MatNum; branchorder::Symbol = :optimal,
          root::DBHTRootMethod = UniqueRoot(),
          sim::Option{<:AbstractSimilarityMatrixAlgorithm} = nothing)

Perform Direct Bubble Hierarchical Tree clustering, a deterministic clustering algorithm [DBHTs](@cite). This version uses a graph-theoretic filtering technique called Triangulated Maximally Filtered Graph (TMFG).

This function implements the full DBHT clustering pipeline: it constructs a Planar Maximally Filtered Graph (PMFG) from the similarity matrix, extracts the clique and bubble hierarchies, assigns clusters, and builds a hierarchical clustering (dendrogram) compatible with [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

# Algorithm

 1. Check that `D` and `S` are non-empty and of equal size.
 2. Build the PMFG from `S` with [`PMFG_T2s`](@ref), giving the weighted adjacency `Rpm`, and check its edge count with [`assert_pmfg_weights`](@ref).
 3. Copy the sparsity pattern of `Rpm` into `Apm` and fill it with the dissimilarities of `D`, so the structure comes from the similarities and the lengths from the distances.
 4. Take the shortest path lengths `Dpm` on `Apm` with [`distance_wei`](@ref).
 5. Build the clique and bubble hierarchies from `Rpm` with [`CliqHierarchyTree2s`](@ref), giving `Hb`, `Mb`, `CliqList` and `Sb`.
 6. Lift the clique membership `Mb` to the vertex membership `Mv`: column `n` marks every vertex of every 3-clique that bubble `n` holds.
 7. Assign the clusters with [`BubbleCluster8s`](@ref), giving `Adjv` and the discrete membership `T8`.
 8. Build the linkage matrix `Z` with [`HierarchyConstruct4s`](@ref), and convert it with [`turn_into_Hclust_merges`](@ref).
 9. Load the two merge columns and the heights into a `Clustering.HclustMerges`, and order its branches through the branch `branchorder` selects.
10. Wrap the merges in a `Clustering.Hclust` tagged `:DBHT`.

# Arguments

  - `D`: `N × N` dissimilarity matrix (e.g., a distance matrix). It must be symmetric, and the symmetry is a caller contract that this function does not check.
  - `S`: `N × N` non-negative similarity matrix. It must be symmetric, on the same unchecked contract.
  - `branchorder`: Ordering method for the dendrogram branches. `:optimal` and `:barjoseph` both call `Clustering.orderbranches_barjoseph!`, and `:r` calls `Clustering.orderbranches_r!`. Any other value is **not** refused: it leaves the branches in the order [`HierarchyConstruct4s`](@ref) built them.
  - `root`: Root selection method for the clique hierarchy.
  - `sim`: Similarity matrix algorithm that produced `S`. It is forwarded to [`assert_pmfg_weights`](@ref) and read for nothing else, so that a refusal names the configuration rather than the matrix. A caller that holds only the matrices leaves it `nothing`.

# Validation

  - `!isempty(S)`, raising `IsEmptyError`.
  - `!isempty(D)`, raising `IsEmptyError`.
  - `size(S) == size(D)`, raising `DimensionMismatch`.
  - The PMFG built from `S` keeps its `3N - 6` edges, by [`assert_pmfg_weights`](@ref). An exactly zero similarity is an absent edge.

Symmetry is **not** among them. A caller that derives both matrices from a correlation matrix gets
it by construction, and a caller that assembles either by hand carries the contract itself.

# Returns

  - `T8::Vector{Int}`: `N × 1` cluster membership vector. `T8[n] = k` puts vertex `n` in the `k`-th discrete cluster.
  - `Rpm::SparseMatrixCSC{<:Number, Int}`: `N × N` adjacency matrix of the Planar Maximally Filtered Graph (PMFG).
  - `Adjv::SparseMatrixCSC{Int, Int}`: Bubble cluster membership matrix from [`BubbleCluster8s`](@ref).
  - `Dpm::Matrix{<:Number}`: `N × N` shortest path length matrix of the PMFG.
  - `Mv::SparseMatrixCSC{Int, Int}`: `N × Nb` bubble membership matrix. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
  - `Z::Matrix{<:Number}`: `(N-1)×3` linkage matrix in Matlab format.
  - `Z_hclust::Clustering.Hclust`: Dendrogram in [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) format.

# Related

  - [`DBHT`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`BubbleCluster8s`](@ref)
  - [`HierarchyConstruct4s`](@ref)
  - [`turn_into_Hclust_merges`](@ref)
  - [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust)
"""
function DBHTs(D::MatNum, S::MatNum; branchorder::Symbol = :optimal,
               root::DBHTRootMethod = UniqueRoot(),
               sim::Option{<:AbstractSimilarityMatrixAlgorithm} = nothing)
    @argcheck(!isempty(S), IsEmptyError)
    @argcheck(!isempty(D), IsEmptyError)
    @argcheck(size(S) == size(D), DimensionMismatch)
    Rpm = PMFG_T2s(S)[1]
    assert_pmfg_weights(Rpm, sim)
    Apm = copy(Rpm)
    Apm[Apm .!= 0] = D[Apm .!= 0]
    Dpm = distance_wei(Apm)[1]

    H1, Hb, Mb, CliqList, Sb = CliqHierarchyTree2s(Rpm, root)

    Mb = Mb[1:size(CliqList, 1), :]

    sRpm = size(Rpm, 1)
    Mv = SparseArrays.spzeros(Int, sRpm, 0)

    nMb = size(Mb, 2)
    for n in axes(Mb, 2)
        vc = SparseArrays.spzeros(Int, sRpm)
        vc[sort!(unique(CliqList[Mb[:, n] .!= 0, :]))] .= 1
        Mv = hcat(Mv, vc)
    end

    Adjv, T8 = BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)

    Z = HierarchyConstruct4s(Rpm, Dpm, T8, Mv)
    Z = turn_into_Hclust_merges(Z)

    n = size(Z, 1)
    hmer = Clustering.HclustMerges{eltype(D)}(n + 1)
    resize!(hmer.mleft, n) .= Int.(Z[:, 1])
    resize!(hmer.mright, n) .= Int.(Z[:, 2])
    resize!(hmer.heights, n) .= Z[:, 3]

    if branchorder == :barjoseph || branchorder == :optimal
        Clustering.orderbranches_barjoseph!(hmer, D)
    elseif branchorder == :r
        Clustering.orderbranches_r!(hmer)
    end

    Z_hclust = Clustering.Hclust(hmer, :DBHT)

    return T8, Rpm, Adjv, Dpm, Mv, Z, Z_hclust
end
"""
    clusterise(cle::ClustersEstimator{<:Any, <:Any, <:DBHT, <:Any}, X::MatNum;
               branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)

Perform Direct Bubble Hierarchical Tree (DBHT) clustering using a `ClustersEstimator` configured with a `DBHT` algorithm.

This method computes the similarity and distance matrices from the input data matrix `X` using the estimator's configured estimators and algorithms, applies the DBHT clustering pipeline, and returns a [`Clusters`](@ref) result containing the hierarchical clustering, similarity and distance matrices, and the optimal number of clusters.

# Algorithm

 1. Take the correlation matrix `S` and the distance matrix `D` from `X` with `cle.ce` and `cle.de`, through [`cor_and_dist`](@ref).
 2. Check that `D` lies in the domain `cle.alg.sim` needs, with [`assert_similarity_domain`](@ref).
 3. Map `D` to the non-negative similarity `S` with [`distance_to_similarity`](@ref), through the branch `cle.alg.sim` selects.
 4. Run the pipeline with [`DBHTs`](@ref), and keep its last output alone, the `Clustering.Hclust` dendrogram `res`.
 5. Take the number of clusters `k` from `res` and `D` with [`optimal_number_clusters`](@ref), through the branch `cle.onc` selects.
 6. Wrap `res`, `S`, `D` and `k` in a [`Clusters`](@ref).

# Arguments

  - `cle`: A `ClustersEstimator` whose algorithm is a [`DBHT`](@ref) instance.
  - `X`: Data matrix (`observations × assets` or `assets × observations` depending on `dims`).
  - `branchorder`: Symbol specifying the dendrogram branch ordering method. Accepts `:optimal` (default), `:barjoseph`, or `:r`.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimators.

# Returns

  - `clr::Clusters`: DBHT clustering result.

# Related

  - [`DBHT`](@ref)
  - [`Clusters`](@ref)
  - [`DBHTs`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`ClustersEstimator`](@ref)
"""
function clusterise(cle::ClustersEstimator{<:Any, <:Any, <:DBHT, <:Any}, X::MatNum;
                    branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    assert_similarity_domain(cle.alg.sim, cle.de, D)
    S = distance_to_similarity(cle.alg.sim; S = S, D = D)
    res = DBHTs(D, S; branchorder = branchorder, root = cle.alg.root, sim = cle.alg.sim)[end]
    k = optimal_number_clusters(cle.onc, res, D)
    return Clusters(; res = res, S = S, D = D, k = k)
end
