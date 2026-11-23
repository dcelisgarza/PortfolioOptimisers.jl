"""
    DBHTs(D::MatNum, S::MatNum;
          branchorder::Option{<:AbstractBranchOrder} = OptimalLeafOrder(),
          root::DBHTRootMethod = UniqueRoot(),
          sim::Option{<:AbstractSimilarityMatrixAlgorithm} = nothing)

Run the Direct Bubble Hierarchical Tree clustering of [DBHTs](@cite), a deterministic clustering algorithm.

This function filters the similarity matrix with a Triangulated Maximally Filtered Graph (TMFG). [DBHTs](@cite) builds a planar maximally filtered graph instead, and [`DBHT`](@ref) states the difference. The `root` and `branchorder` keywords come from this library alone.

The function runs the full DBHT pipeline. It builds the TMFG from the similarity matrix, builds the clique and bubble hierarchies, assigns the clusters, and builds a dendrogram as a [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

# Algorithm

 1. Check that `D` and `S` are non-empty and of equal size.
 2. Build the TMFG from `S` with [`PMFG_T2s`](@ref), which gives the weighted adjacency `Rpm`. Check its edge count with [`assert_pmfg_weights`](@ref).
 3. Copy the sparsity pattern of `Rpm` into `Apm` and fill it with the dissimilarities of `D`, so the structure comes from the similarities and the lengths from the distances.
 4. Take the shortest path lengths `Dpm` on `Apm` with [`distance_wei`](@ref).
 5. Build the clique and bubble hierarchies from `Rpm` with [`CliqHierarchyTree2s`](@ref), which gives `Hb`, `Mb`, `CliqList` and `Sb`.
 6. Lift the clique membership `Mb` to the vertex membership `Mv`. Column `n` of `Mv` marks every vertex of every 3-clique that bubble `n` holds.
 7. Assign the clusters with [`BubbleCluster8s`](@ref), which gives `Adjv` and the discrete membership `T8`.
 8. Build the linkage matrix `Z` with [`HierarchyConstruct4s`](@ref), and convert it with [`turn_into_Hclust_merges`](@ref).
 9. Load the two merge columns and the heights into a `Clustering.HclustMerges`, and order its branches under `branchorder` with [`order_branches!`](@ref).
10. Wrap the merges in a `Clustering.Hclust` tagged `:DBHT`.

# Arguments

  - `D`: `N × N` dissimilarity matrix, such as a distance matrix. It must be symmetric. The caller holds that contract, and this function does not check it.
  - `S`: `N × N` non-negative similarity matrix. It must be symmetric, on the same contract that the function does not check.
  - `branchorder`: The branch order of the dendrogram, a member of [`AbstractBranchOrder`](@ref), or `nothing` for the tree order. [`order_branches!`](@ref) states what each member does. The merges come in the order that [`HierarchyConstruct4s`](@ref) built them, and `nothing` keeps them so.
  - `root`: Root selection method for the clique hierarchy.
  - `sim`: Similarity matrix algorithm that made `S`. The function passes it to [`assert_pmfg_weights`](@ref) and reads it for nothing else, so that a refusal names the configuration and not the matrix. A caller that holds only the matrices leaves it `nothing`.

# Validation

  - `!isempty(S)`, raising `IsEmptyError`.
  - `!isempty(D)`, raising `IsEmptyError`.
  - `size(S) == size(D)`, raising `DimensionMismatch`.
  - The TMFG that the function builds from `S` keeps its `3N - 6` edges, by [`assert_pmfg_weights`](@ref). An exactly zero similarity is an absent edge.

The function does not check the symmetry of `D` or `S`. A caller that derives both matrices from a correlation matrix gets symmetric matrices. A caller that makes either matrix by hand holds the contract itself.

# Returns

  - `T8::Vector{Int}`: `N × 1` cluster membership vector. `T8[n] = k` puts vertex `n` in the `k`-th discrete cluster.
  - `Rpm::SparseMatrixCSC{<:Number, Int}`: `N × N` adjacency matrix of the Triangulated Maximally Filtered Graph (TMFG).
  - `Adjv::SparseMatrixCSC{Int, Int}`: Non-discrete cluster membership matrix from [`BubbleCluster8s`](@ref). It is `0 × 0` when the graph has one converging bubble or none.
  - `Dpm::Matrix{<:Number}`: `N × N` shortest path length matrix of the TMFG.
  - `Mv::SparseMatrixCSC{Int, Int}`: `N × Nb` bubble membership matrix. `Mv[n, bi] = 1` means that vertex `n` is a vertex of bubble `bi`.
  - `Z::Matrix{<:Number}`: `(N-1)×4` linkage matrix in the convention of [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust), as [`turn_into_Hclust_merges`](@ref) writes it. The branch order of step 9 does not change it.
  - `Z_hclust::Clustering.Hclust`: Dendrogram in [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) format.

# Related

  - [`DBHT`](@ref)
  - [`CliqHierarchyTree2s`](@ref)
  - [`BubbleCluster8s`](@ref)
  - [`HierarchyConstruct4s`](@ref)
  - [`turn_into_Hclust_merges`](@ref)
  - [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust)
"""
function DBHTs(D::MatNum, S::MatNum;
               branchorder::Option{<:AbstractBranchOrder} = OptimalLeafOrder(),
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

    sRpm = size(Rpm, 1)::Int
    nMb = size(Mb, 2)
    Mv = SparseArrays.spzeros(Int, sRpm, nMb)
    for n in axes(Mb, 2)
        Mv[sort!(unique(CliqList[@view(Mb[:, n]) .!= 0, :])), n] .= 1
    end

    Adjv, T8 = BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)

    Z = HierarchyConstruct4s(Rpm, Dpm, T8, Mv)
    Z = turn_into_Hclust_merges(Z)

    n = size(Z, 1)
    hmer = Clustering.HclustMerges{eltype(D)}(n + 1)
    resize!(hmer.mleft, n) .= Int.(@view(Z[:, 1]))
    resize!(hmer.mright, n) .= Int.(@view(Z[:, 2]))
    resize!(hmer.heights, n) .= Z[:, 3]

    order_branches!(hmer, D, branchorder)

    Z_hclust = Clustering.Hclust(hmer, :DBHT)

    return T8, Rpm, Adjv, Dpm, Mv, Z, Z_hclust
end
"""
    clusterise(cle::ClustersEstimator{<:Any, <:Any, <:DBHT, <:Any}, X::MatNum;
               dims::Int = 1, kwargs...)

Run Direct Bubble Hierarchical Tree (DBHT) clustering with a `ClustersEstimator` whose algorithm is a `DBHT`.

The method computes the similarity and distance matrices of the data matrix `X` with the estimators that `cle` holds, and it runs the DBHT pipeline on them. It returns a [`Clusters`](@ref) result that holds the dendrogram, the similarity and distance matrices, and the optimal number of clusters.

# Algorithm

 1. Take the correlation matrix `S` and the distance matrix `D` from `X` with `cle.ce` and `cle.de`, through [`cor_and_dist`](@ref). Refuse each asset that the reader could not estimate with [`assert_distance_reader_rows`](@ref).
 2. Check with [`assert_similarity_domain`](@ref) that `D` lies in the domain that `cle.alg.sim` needs.
 3. Map `D` to the non-negative similarity `S` with [`distance_to_similarity`](@ref), through the branch that `cle.alg.sim` selects.
 4. Run the pipeline with [`DBHTs`](@ref) under `cle.alg.root` and `cle.alg.branchorder`, and keep only its last output, the `Clustering.Hclust` dendrogram `res`.
 5. Take the number of clusters `k` from `res` and `D` with [`optimal_number_clusters`](@ref), through the branch that `cle.onc` selects.
 6. Wrap `res`, `S`, `D` and `k` in a [`Clusters`](@ref).

# Arguments

  - `cle`: A `ClustersEstimator` whose algorithm is a [`DBHT`](@ref).
  - `X`: Data matrix, `observations × assets` when `dims` is `1` and `assets × observations` when `dims` is `2`.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments that the method passes to [`cor_and_dist`](@ref).

# Validation

  - No diagonal entry of `D` is `NaN`. Raises a [`DistanceReaderRefusalError`](@ref) that names the columns of the assets that the reader could not estimate. A fit catches it and drops those assets.

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
                    dims::Int = 1, kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    assert_distance_reader_rows(D)
    assert_similarity_domain(cle.alg.sim, cle.de, D)
    S = distance_to_similarity(cle.alg.sim; S = S, D = D)
    res = DBHTs(D, S; branchorder = cle.alg.branchorder, root = cle.alg.root,
                sim = cle.alg.sim)[end]
    k = optimal_number_clusters(cle.onc, res, D)
    return Clusters(; res = res, S = S, D = D, k = k)
end
