"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all clustering estimator types.

A subtype holds the configuration that clusters the assets of a universe. [`clusterise`](@ref) runs it and returns an [`AbstractClusteringResult`](@ref). [`ClustersEstimator`](@ref) and [`NetworkClustersEstimator`](@ref) are the two subtypes.

# Related

  - [`AbstractClustersAlgorithm`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClustersEstimator <: AbstractPhylogenyEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all clustering algorithm types.

A subtype selects the method that clusters a distance matrix. The hierarchical algorithms and the non-hierarchical algorithms are the two families.

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClustersAlgorithm <: AbstractPhylogenyAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the clustering algorithm `alg` unchanged.

A caller that rebuilds an estimator with [`factory`](@ref) reaches this method for a clustering algorithm. A clustering algorithm has no field that depends on a prior, so the method keeps every field and replaces none.

# Algorithm

 1. Return `alg` itself. The method rebuilds no field of it, and it discards `args` and `kwargs`.

# Arguments

  - `alg`: Clustering algorithm.
  - `args...`: Optional arguments. The method does not read them.
  - `kwargs...`: Optional keyword arguments. The method does not read them.

# Returns

  - `alg::AbstractClustersAlgorithm`: The original clustering algorithm.

# Related

  - [`AbstractClustersAlgorithm`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::AbstractClustersAlgorithm, args...; kwargs...)
    return alg
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all hierarchical clustering algorithm types.

A subtype builds a dendrogram, a `Clustering.Hclust`. [`optimal_number_clusters`](@ref) then cuts the dendrogram at the selected number of clusters. [`HClustAlgorithm`](@ref) and [`DBHT`](@ref) are the subtypes.

# Related

  - [`AbstractClustersAlgorithm`](@ref)
  - [`HClustAlgorithm`](@ref)
  - [`DBHT`](@ref)
"""
abstract type AbstractHierarchicalClusteringAlgorithm <: AbstractClustersAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all non-hierarchical clustering algorithm types.

A subtype makes a flat partition, a `Clustering.ClusteringResult`, at one number of clusters. A partition has no dendrogram to cut, so [`optimal_number_clusters`](@ref) runs the algorithm once for each candidate count. [`KMeansAlgorithm`](@ref) is the subtype.

# Related

  - [`AbstractClustersAlgorithm`](@ref)
  - [`KMeansAlgorithm`](@ref)
"""
abstract type AbstractNonHierarchicalClusteringAlgorithm <: AbstractClustersAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all optimal number of clusters estimator types.

A subtype holds the configuration that selects the number of clusters. [`OptimalNumberClusters`](@ref) is the subtype.

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
"""
abstract type AbstractOptimalNumberClustersEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all optimal number of clusters algorithm types.

A subtype is a score that [`optimal_number_clusters`](@ref) calculates for each candidate count, and the count with the largest score wins. [`SecondOrderDifference`](@ref) and [`SilhouetteScore`](@ref) are the subtypes.

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
"""
abstract type AbstractOptimalNumberClustersAlgorithm <: AbstractAlgorithm end
"""
    const Int_ONC = Union{<:Integer, <:AbstractOptimalNumberClustersAlgorithm}

Alias for an integer or optimal number of clusters algorithm.

Matches an integer, which states the number of clusters, or an [`AbstractOptimalNumberClustersAlgorithm`](@ref), which selects the number from the data.

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
"""
const Int_ONC = Union{<:Integer, <:AbstractOptimalNumberClustersAlgorithm}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all clustering result types.

A subtype holds the output of a clustering estimator. [`Clusters`](@ref) is the subtype.

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractClustersAlgorithm`](@ref)
"""
abstract type AbstractClusteringResult <: AbstractPhylogenyResult end
"""
    const ClTypes = Union{<:Clustering.ClusteringResult, <:Clustering.Hclust}

Alias for clustering result types from the Clustering.jl package.

Matches a `Clustering.ClusteringResult` or a `Clustering.Hclust`. The `res` field of [`Clusters`](@ref) accepts either, so one result type holds a flat partition or a dendrogram.

# Related

  - [`Clusters`](@ref)
  - [`ClustersEstimator`](@ref)
"""
const ClTypes = Union{<:Clustering.ClusteringResult, <:Clustering.Hclust}
"""
$(DocStringExtensions.TYPEDEF)

Holds a clustering of the asset universe and the matrices that the clustering used.

`res` holds either of the two outputs of [`clusterise`](@ref). A hierarchical algorithm makes a `Clustering.Hclust`, and a non-hierarchical algorithm makes a `Clustering.ClusteringResult`. `k` is the number of clusters that an [`AbstractOptimalNumberClustersEstimator`](@ref) selected. [`assignments`](@ref) reads one cluster label per asset from `res` and `k`. On the hierarchical branch it cuts the tree at `k`, and on the other branch it reads the labels that `res` holds.

# The matrix that the clustering used

`S` is the similarity matrix of the universe, and `D` is its distance matrix. A [`ClustersEstimator`](@ref) clusters `D` and sets `P` to `nothing`. A [`NetworkClustersEstimator`](@ref) makes a pseudo-distance matrix from the structure of the network and clusters that matrix. It stores the matrix in `P`, so `res` is the clustering of `P`. The result also keeps `D`, so a caller can read the distances of the universe next to the clustering of the network.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Clusters(;
        res::ClTypes,
        S::MatNum,
        D::MatNum,
        P::Option{<:MatNum} = nothing,
        k::Integer
    ) -> Clusters

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:S])
  - $(val_dict[:D])
  - $(val_dict[:S_D])
  - $(val_dict[:S_P])
  - $(val_dict[:ck])

# Related

  - [`AbstractClusteringResult`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`NetworkClustersEstimator`](@ref)
  - [`clusterise`](@ref)
  - [`assignments`](@ref)
"""
@concrete struct Clusters <: AbstractClusteringResult
    """
    $(field_dict[:clres])
    """
    res
    """
    $(field_dict[:S])
    """
    S
    """
    $(field_dict[:D])
    """
    D
    """
    $(field_dict[:clP])
    """
    P
    """
    $(field_dict[:ck])
    """
    k
    function Clusters(res::ClTypes, S::MatNum, D::MatNum, P::Option{<:MatNum}, k::Integer)
        @argcheck(!isempty(S), IsEmptyError)
        @argcheck(!isempty(D), IsEmptyError)
        @argcheck(size(S) == size(D), DimensionMismatch)
        if !isnothing(P)
            @argcheck(!isempty(P), IsEmptyError)
            @argcheck(size(S) == size(P), DimensionMismatch)
        end
        @argcheck(one(k) <= k,
                  DomainError(k,
                              "`Clusters.k` is $k, and a clustering holds at least one cluster. State an integer greater than or equal to `1`."))
        return new{typeof(res), typeof(S), typeof(D), typeof(P), typeof(k)}(res, S, D, P, k)
    end
end
function Clusters(; res::ClTypes, S::MatNum, D::MatNum, P::Option{<:MatNum} = nothing,
                  k::Integer)::Clusters
    return Clusters(res, S, D, P, k)
end
"""
    clusterise(cle::AbstractClusteringResult, args...; kwargs...)

Return the clustering result `cle` unchanged.

A consumer calls `clusterise` on a clustering estimator or on a precomputed result. For a result, this method runs no clustering, so the library never clusters a result again.

# Algorithm

 1. Return `cle` itself. The method runs no clustering, and it discards `args` and `kwargs`.

# Arguments

  - `cle`: Clustering result.
  - `args...`: Positional arguments. The method does not read them.
  - `kwargs...`: Keyword arguments. The method does not read them.

# Returns

  - `cle::AbstractClusteringResult`: The input, unchanged.

# Related

  - [`AbstractClusteringResult`](@ref)
  - [`Clusters`](@ref)
  - [`ClustersEstimator`](@ref)
"""
function clusterise(cle::AbstractClusteringResult, args...; kwargs...)
    return cle
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the number of clusters at which the within-cluster dispersion has its largest second-order difference.

This score is the two-difference gap statistic. The second-order difference finds the elbow of the curve of the dispersion over the cluster counts. The gap statistic of [tibshirani2001](@cite) needs a Monte Carlo simulation for this, and the second-order difference does not.

# Mathematical definition

```math
\\begin{align}
c^{\\star} &= \\underset{c}{\\arg\\max} \\left(W_{c+2} + W_{c} - 2 W_{c+1}\\right)\\,,\\\\
&\\mathrm{s.t.} \\quad 1 \\leq c \\leq \\sqrt{N}\\,,\\\\
W_{c} &= \\sum_{j=1}^{c} g\\left(\\left\\{d_{uv} : u, v \\in \\mathcal{C}_{j},\\, u < v\\right\\}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:c_star_clusters])
  - ``W_{c}``: Within-cluster dispersion of a cut into ``c`` clusters.
  - ``\\mathcal{C}_{j}``: Set of assets in the ``j``-th cluster of that cut.
  - ``d_{uv}``: Entry of the distance matrix that the clustering used.
  - ``g``: Vector-to-scalar measure `alg`. It reduces the pairwise distances of one cluster to one number. A cluster of one asset has no pairwise distance and contributes ``0``.
  - $(math_dict[:N])

The selected ``c`` is the left end of the triple, not its centre. Equation 12.15 of [cajas2025](@cite) prints it so, and the code maximises the equation as printed. The equation admits ``c = 1``, and step 3 of the algorithm below excludes it.

# The measure defines the dispersion

Equation 12.15 of [cajas2025](@cite) calls ``W_{c}`` the within-cluster dispersion and gives no other definition. The book takes the equation from [yue2008](@cite). No copy of that work is available, so its definition of ``W_{c}`` is not known.

The gap statistic of [tibshirani2001](@cite), Equation 2, takes ``W_{k} = \\sum_{r} D_{r} / (2 n_{r})``. ``D_{r}`` is the sum of the pairwise distances of cluster ``r`` over the ordered pairs. That is ``(n_{r} - 1) / 2`` times the mean of the distances, so a large cluster has more weight. `alg = MeanValue()` takes the mean of the pairwise distances of a cluster. The default `alg = StandardisedValue()` divides that mean by the corrected standard deviation of the same distances.

The three definitions can select three different counts. On the sample `randn(StableRNG(987654321), 400, 40)`, `MeanValue()` selects 6 and `StandardisedValue()` selects 4.

!!! note

    A cluster of two assets has one pairwise distance, and one value has no corrected standard deviation. [`StandardisedValue`](@ref) divides by ``1`` in that case, so such a cluster contributes its mean pairwise distance to ``W_{c}``. The gap series stays finite, and the selected ``c`` is a real maximiser. The two measures still differ on every larger cluster.

# Algorithm

[`optimal_number_clusters`](@ref) does these steps on both branches. Its methods are in `03_Hierarchical.jl` and in `04_NonHierarchicalClustering.jl`.

 1. Take the ceiling `min(floor(Int, sqrt(N)), max_k)` from the [`OptimalNumberClusters`](@ref) that holds the score. Add ``2`` to it and cap the sum at ``N``. The result is `c1`, the number of candidate counts. The difference at ``c`` also reads the dispersions at ``c + 1`` and ``c + 2``, so the method builds two more counts.
 2. Cluster the universe at every count from ``1`` to `c1`. The cuts are `cluster_lvls`.
 3. Set `W_list[1]` to `typemin`, so that the difference at ``c = 1`` cannot win.
 4. Reduce every cut from ``2`` to `c1` to one dispersion with `alg`, and write it into `W_list`. On the hierarchical branch, `alg` runs once for each cluster over the pairwise distances of that cluster, and the method sums the results. On the non-hierarchical branch, `alg` runs once over the per-point costs of the k-means run.
 5. Take the second-order difference of `W_list`. The result is `gaps`, one entry for each candidate count up to the ceiling. If `c1` is ``2`` or less, there is no such difference. This happens only when ``N \\leq 2``, where the ceiling is ``1``, and both branches return the ceiling.
 6. Select the count. The hierarchical branch gives `gaps` to [`valid_k_clusters`](@ref). That function moves down from the largest entry until the dendrogram accepts the count. The non-hierarchical branch has no dendrogram to reject a count, so it takes the `argmax` of `gaps`. If no entry of the scores is finite, both branches take the length of the score array in place of the `argmax`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SecondOrderDifference(;
        alg::Num_VecToScaM = StandardisedValue()
    ) -> SecondOrderDifference

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `alg`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> SecondOrderDifference()
SecondOrderDifference
  alg ┼ StandardisedValue
      │   mv ┼ MeanValue
      │      │   w ┴ nothing
      │   sv ┼ StdValue
      │      │           w ┼ nothing
      │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
  - [`SilhouetteScore`](@ref)
  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`StandardisedValue`](@ref)
  - [`optimal_number_clusters`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 12.2.1.2, Equation 12.15. $(ref_dict[:no_original_source])
  - $(ref_dict[:yue2008])
  - $(ref_dict[:tibshirani2001]) Equations 1-3, for the gap statistic that needs the simulation.
"""
@propagatable @concrete struct SecondOrderDifference <:
                               AbstractOptimalNumberClustersAlgorithm
    """
    $(field_dict[:vsalg])
    """
    @fprop alg
    function SecondOrderDifference(alg::Num_VecToScaM)
        return new{typeof(alg)}(alg)
    end
end
function SecondOrderDifference(;
                               alg::Num_VecToScaM = StandardisedValue())::SecondOrderDifference
    return SecondOrderDifference(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the number of clusters from the silhouettes of the assets.

The silhouette of an asset compares its mean distance to its own cluster with its mean distance to the nearest other cluster. For each candidate count, `alg` reduces the vector of silhouettes to one number. The count with the largest number wins.

# Mathematical definition

```math
\\begin{align}
s_{i} &= \\frac{b_{i} - a_{i}}{\\max\\left(a_{i},\\, b_{i}\\right)}\\,,\\\\
a_{i} &= \\sum_{j \\in \\mathcal{C}_{I},\\, j \\neq i} \\frac{d_{ij}}{\\left|\\mathcal{C}_{I}\\right| - 1}\\,,\\\\
b_{i} &= \\underset{J \\neq I}{\\min} \\sum_{j \\in \\mathcal{C}_{J}} \\frac{d_{ij}}{\\left|\\mathcal{C}_{J}\\right|}\\,,\\\\
c^{\\star} &= \\underset{c}{\\arg\\max} \\; g\\left(\\boldsymbol{s}\\right)\\,, \\quad 1 \\leq c \\leq \\sqrt{N}\\,.
\\end{align}
```

Where:

  - ``s_{i}``: Silhouette of asset ``i``, in ``\\left[-1,\\, 1\\right]``. It is ``1`` when ``a_{i} = 0`` and ``-1`` when ``b_{i} = 0``.
  - ``a_{i}``: Mean distance from asset ``i`` to the other members of its own cluster ``\\mathcal{C}_{I}``.
  - ``b_{i}``: Smallest mean distance from asset ``i`` to the members of any other cluster.
  - ``d_{ij}``: Entry of the distance matrix that the clustering used.
  - ``\\boldsymbol{s}``: Vector of silhouettes over all assets, one per asset.
  - ``g``: Vector-to-scalar measure `alg`.
  - $(math_dict[:c_star_clusters])
  - $(math_dict[:N])

# The default reduction is the source's standardised score

`alg = StandardisedValue()` divides the mean of ``\\boldsymbol{s}`` by its corrected standard deviation. This is the quality measure of [lopezdeprado2019](@cite), term for term. [lopezdeprado2019](@cite) writes ``q = \\mathrm{E}[\\{s_{i}\\}] / \\sqrt{\\mathrm{V}[\\{s_{i}\\}]}`` and does not say which variance it takes. Equation 12.14 of [cajas2025](@cite) takes the corrected variance. At a fixed ``N`` the two forms differ by a constant factor, so they select the same ``c``.

`alg = MeanValue()` gives the rule of [rousseeuw1987](@cite), which selects the count with the largest mean silhouette. The mean and the standardised score can select different counts. On the sample `randn(StableRNG(987654321), 400, 40)`, the mean selects 6 and the standardised score selects 4.

Neither [rousseeuw1987](@cite) nor [lopezdeprado2019](@cite) bounds ``c`` by ``\\sqrt{N}``, and the second tries every count from ``2`` to ``N - 1``. The bound is the ceiling of [`OptimalNumberClusters`](@ref), which applies to both scores.

The reduction runs over one vector of length ``N``, not over one cluster at a time as in [`SecondOrderDifference`](@ref). So a cluster of one or two assets needs no special case here.

# Algorithm

[`optimal_number_clusters`](@ref) does these steps on both branches. Its methods are in `03_Hierarchical.jl` and in `04_NonHierarchicalClustering.jl`.

 1. Take the ceiling `min(floor(Int, sqrt(N)), max_k)` from the [`OptimalNumberClusters`](@ref) that holds the score. The result is `c1`, the largest candidate count. The score at ``c`` reads only the cut at ``c``, so the method builds no count above the ceiling.
 2. Cluster the universe at every count from ``1`` to `c1`. The cuts are `cluster_lvls`.
 3. Set `W_list[1]` to `typemin`, so that a single cluster cannot win.
 4. Calculate the silhouettes of the assets for every cut from ``2`` to `c1` with `Clustering.silhouettes`. The result for one cut is `sl`, with one entry for each asset.
 5. Reduce each `sl` to one score with `alg`, and write it into `W_list`.
 6. Select the count. The hierarchical branch gives `W_list` to [`valid_k_clusters`](@ref). That function moves down from the largest entry until the dendrogram accepts the count. The non-hierarchical branch has no dendrogram to reject a count, so it takes the `argmax` of `W_list`. If no entry of the scores is finite, both branches take the length of the score array in place of the `argmax`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SilhouetteScore(;
        alg::Num_VecToScaM = StandardisedValue()
    ) -> SilhouetteScore

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `alg`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> SilhouetteScore()
SilhouetteScore
  alg ┼ StandardisedValue
      │   mv ┼ MeanValue
      │      │   w ┴ nothing
      │   sv ┼ StdValue
      │      │           w ┼ nothing
      │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
  - [`SecondOrderDifference`](@ref)
  - [`VectorToScalarMeasure`](@ref)
  - [`StandardisedValue`](@ref)
  - [`optimal_number_clusters`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:rousseeuw1987]) Section 2, pages 55-56, for ``s_{i}``, and page 59 for the mean silhouette width.
  - $(ref_dict[:lopezdeprado2019]) Section 3, for the quality measure ``q``.
  - $(ref_dict[:cajas2025]) Section 12.2.1.1, Equation 12.14, for the corrected variance in ``q``.
"""
@propagatable @concrete struct SilhouetteScore <: AbstractOptimalNumberClustersAlgorithm
    """
    $(field_dict[:vsalg])
    """
    @fprop alg
    function SilhouetteScore(alg::Num_VecToScaM)
        return new{typeof(alg)}(alg)
    end
end
function SilhouetteScore(; alg::Num_VecToScaM = StandardisedValue())::SilhouetteScore
    return SilhouetteScore(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the number of clusters of a dendrogram or a flat partition, under a ceiling.

`alg` is an [`AbstractOptimalNumberClustersAlgorithm`](@ref), which selects the count from the data, or an `Integer`, which states the count.

# The ceiling and a stated `k`

The ceiling is `min(floor(Int, sqrt(N)), max_k)` on every branch. A `max_k` above ``\\sqrt{N}`` has no effect, and `max_k = nothing` sets the ceiling to `floor(Int, sqrt(N))`. The selection lowers a stated `k` above the ceiling to the ceiling.

On the hierarchical branch, [`validate_k_value`](@ref) also tests a stated `k` against the tree. If no node of the dendrogram supports a cut into `k` clusters, [`optimal_number_clusters`](@ref) searches up and down for the nearest valid count and takes the nearer one. On the same branch, the two scores apply the same test through [`valid_k_clusters`](@ref). That function moves down the score array from the largest entry until a count passes.

The ceiling ``\\sqrt{N}`` is the bound of Equation 12.15 of [cajas2025](@cite). There, it applies only to the two-difference gap statistic. The book takes that equation from a work of which no copy is available, see [`SecondOrderDifference`](@ref). The cap `max_k` and the search of the dendrogram for a valid `k` are additions of this library.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimalNumberClusters(;
        max_k::Option{<:Integer} = nothing,
        alg::Int_ONC = SecondOrderDifference()
    ) -> OptimalNumberClusters

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:max_k])
  - $(val_dict[:kalg])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `alg`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> OptimalNumberClusters(; max_k = 10)
OptimalNumberClusters
  max_k ┼ Int64: 10
    alg ┼ SecondOrderDifference
        │   alg ┼ StandardisedValue
        │       │   mv ┼ MeanValue
        │       │      │   w ┴ nothing
        │       │   sv ┼ StdValue
        │       │      │           w ┼ nothing
        │       │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`SecondOrderDifference`](@ref)
  - [`SilhouetteScore`](@ref)
  - [`optimal_number_clusters`](@ref)
  - [`valid_k_clusters`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 12.2.1.2, Equation 12.15. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct OptimalNumberClusters <:
                               AbstractOptimalNumberClustersEstimator
    """
    $(field_dict[:max_k])
    """
    max_k
    """
    $(field_dict[:kalg])
    """
    @fprop alg
    function OptimalNumberClusters(max_k::Option{<:Integer}, alg::Int_ONC)
        if !isnothing(max_k)
            @argcheck(one(max_k) <= max_k,
                      DomainError(max_k,
                                  "`OptimalNumberClusters.max_k` is $max_k, and the search needs at least one cluster to choose from. State an integer greater than or equal to `1`, or `nothing` for the default cap."))
        end
        if isa(alg, Integer)
            @argcheck(one(alg) <= alg,
                      DomainError(alg,
                                  "`OptimalNumberClusters.alg` is $alg, and an integer here fixes the number of clusters, so it is at least `1`. State an integer greater than or equal to `1`, or an algorithm that chooses it."))
        end
        return new{typeof(max_k), typeof(alg)}(max_k, alg)
    end
end
function OptimalNumberClusters(; max_k::Option{<:Integer} = nothing,
                               alg::Int_ONC = SecondOrderDifference())::OptimalNumberClusters
    return OptimalNumberClusters(max_k, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Builds a dendrogram that merges the two nearest clusters at each step, until one cluster remains.

`linkage` is the criterion for the distance between two clusters that have more than one asset.

  - `:single` takes the smallest distance between the members of the two clusters.
  - `:complete` takes the largest distance between the members of the two clusters.
  - `:average` takes the mean distance over all pairs with one member in each cluster.
  - `:ward` merges the pair of clusters that increases the total within-cluster variance least.
  - `:ward_presquared` is `:ward` for a matrix that holds squared distances.

[`clusterise`](@ref) gives the symbol to `Clustering.hclust` unchanged, so every criterion of that function is available.

Figure 2 of [mullner2011](@cite) states each criterion in two forms. One form updates the distance after a merge, and the other is a closed form over the members. The formulas of `:ward` hold for Euclidean distances between points. `Clustering.hclust` squares the distances for `:ward`, applies the update to the squares, and takes the square root of the merge heights. This is the form of that figure. Equation 12.12 of [cajas2025](@cite) prints the same update without the squares.

[mullner2011](@cite) is a survey. No copy is available of the works that first state the four criteria, or of the work that first states the update that they share.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HClustAlgorithm(;
        linkage::Symbol = :ward
    ) -> HClustAlgorithm

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> HClustAlgorithm(; linkage = :average)
HClustAlgorithm
  linkage ┴ Symbol: :average
```

# Related

  - [`AbstractHierarchicalClusteringAlgorithm`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`DBHT`](@ref)
  - [`clusterise`](@ref)

# References

  - $(ref_dict[:mullner2011]) Figure 2. $(ref_dict[:no_original_source])
"""
@concrete struct HClustAlgorithm <: AbstractHierarchicalClusteringAlgorithm
    """
    Linkage criterion of `Clustering.hclust`, see [`Clustering.jl`](https://juliastats.org/Clustering.jl/stable/hclust.html). The constructor accepts any `Symbol` and checks none, because `Clustering.hclust` owns the set of criteria. `HClustAlgorithm(; linkage = :nonsense)` builds. [`clusterise`](@ref) then throws `ArgumentError: Unsupported cluster linkage nonsense` from `Clustering.hclust`.
    """
    linkage
    function HClustAlgorithm(linkage::Symbol)
        return new{typeof(linkage)}(linkage)
    end
end
function HClustAlgorithm(; linkage::Symbol = :ward)::HClustAlgorithm
    return HClustAlgorithm(linkage)
end
"""
$(DocStringExtensions.TYPEDEF)

Clusters the assets of a universe from a return matrix.

Four of the fields are steps, in the order that they run.

  - `ce` estimates the correlation matrix `S`.
  - `de` makes the distance matrix.
  - `alg` clusters the distance matrix.
  - `onc` selects the number of clusters.

[`clusterise`](@ref) runs the steps and returns a [`Clusters`](@ref). On the non-hierarchical branch, `onc` runs `alg` once for each candidate count.

# The fields that take observation weights

`ce` and `de` read the observations of the return matrix. A [`VariationInfoDistance`](@ref) in `de` reads the return matrix itself, not the correlation matrix. So `ce` and `de` carry the `@fprop` tag, and the weights of an [`ObsWeights`](@ref) reach them. `alg` and `onc` read only the `assets × assets` distance matrix. They have no observation to weight, so they carry no `@fprop` tag.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ClustersEstimator(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        alg::AbstractClustersAlgorithm = HClustAlgorithm(),
        onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters(),
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> ClustersEstimator

Keywords correspond to the struct's fields.

## Validation

  - `x_src in ALL_RETURNS_SOURCES`. [`assert_returns_source`](@ref) throws an `ArgumentError` otherwise.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).
  - `de`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> ClustersEstimator()
ClustersEstimator
     ce ┼ PortfolioOptimisersCovariance
        │   ce ┼ Covariance
        │      │    me ┼ SimpleExpectedReturns
        │      │       │   w ┴ nothing
        │      │    ce ┼ GeneralCovariance
        │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │      │       │    w ┴ nothing
        │      │   alg ┼ FullMoment()
        │      │     w ┴ nothing
        │   mp ┼ MatrixProcessing
        │      │     pdm ┼ Posdef
        │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │      │      dn ┼ nothing
        │      │      dt ┼ nothing
        │      │     alg ┼ nothing
        │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
     de ┼ Distance
        │   power ┼ nothing
        │     alg ┴ CanonicalDistance()
    alg ┼ HClustAlgorithm
        │   linkage ┴ Symbol: :ward
    onc ┼ OptimalNumberClusters
        │   max_k ┼ nothing
        │     alg ┼ SecondOrderDifference
        │         │   alg ┼ StandardisedValue
        │         │       │   mv ┼ MeanValue
        │         │       │      │   w ┴ nothing
        │         │       │   sv ┼ StdValue
        │         │       │      │           w ┼ nothing
        │         │       │      │   corrected ┴ Bool: true
  x_src ┴ nothing
```

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractHierarchicalClusteringAlgorithm`](@ref)
  - [`AbstractNonHierarchicalClusteringAlgorithm`](@ref)
  - [`AbstractOptimalNumberClustersEstimator`](@ref)
  - [`Clusters`](@ref)
  - [`clusterise`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct ClustersEstimator <: AbstractClustersEstimator
    """
    $(field_dict[:ce])
    """
    @fprop ce
    """
    $(field_dict[:de])
    """
    @fprop de
    """
    $(field_dict[:clalg])
    """
    alg
    """
    $(field_dict[:onc])
    """
    onc
    """
    $(field_dict[:x_src_graph])
    """
    x_src
    function ClustersEstimator(ce::StatsBase.CovarianceEstimator,
                               de::AbstractDistanceEstimator,
                               alg::AbstractClustersAlgorithm,
                               onc::AbstractOptimalNumberClustersEstimator,
                               x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, ALL_RETURNS_SOURCES, :ClustersEstimator)
        return new{typeof(ce), typeof(de), typeof(alg), typeof(onc), typeof(x_src)}(ce, de,
                                                                                    alg,
                                                                                    onc,
                                                                                    x_src)
    end
end
function ClustersEstimator(;
                           ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                           de::AbstractDistanceEstimator = Distance(;
                                                                    alg = CanonicalDistance()),
                           alg::AbstractClustersAlgorithm = HClustAlgorithm(),
                           onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters(),
                           x_src::Option{<:AbstractReturnsSource} = nothing)::ClustersEstimator
    return ClustersEstimator(ce, de, alg, onc, x_src)
end
function returns_source(cle::ClustersEstimator)
    return cle.x_src
end
function reader_frames(cle::ClustersEstimator, path::AbstractString, ::Bool)
    return source_frames(cle.x_src, path, true)
end
"""
    const ClE_Cl = Union{<:AbstractClustersEstimator, <:AbstractClusteringResult}

Alias for a clustering estimator or result.

Matches an [`AbstractClustersEstimator`](@ref) or an [`AbstractClusteringResult`](@ref). The library dispatches on it where a function accepts a clustering estimator or a precomputed result.

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
const ClE_Cl = Union{<:AbstractClustersEstimator, <:AbstractClusteringResult}

export Clusters, clusterise, SecondOrderDifference, SilhouetteScore, OptimalNumberClusters,
       HClustAlgorithm, ClustersEstimator
