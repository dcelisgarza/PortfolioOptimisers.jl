"""
$(DocStringExtensions.TYPEDEF)

Partitions assets into `k` groups by Lloyd's algorithm, with no dendrogram.

Runs [`Clustering.kmeans`](https://juliastats.org/Clustering.jl/stable/api/#Clustering.kmeans) over the columns of the distance matrix. Each asset is the point of its distances to every asset, so two assets fall in one cluster when their distances to the rest of the universe are similar. The result is a flat partition with no tree, so no later step can cut it at a different `k`.

`Clustering.kmeans` alternates two steps over all the points: it assigns each point to its nearest centre, and it moves each centre to the mean of its points. [lloyd1982](@cite) states that alternation for a quantiser of one real variable, in Method I of Section VI. [macqueen1967](@cite) names *k-means* and states the problem for points in ``N`` dimensions, but it moves a mean after each point, one point at a time. The work that first states the alternation over all the points in ``N`` dimensions was not read.

The algorithm draws random starting centres, so the type has the fields `rng` and `seed`. [`resolve_rng`](@ref) combines them when the algorithm runs, so a stated `seed` makes a run reproducible and the caller does not make a generator.

# `kwargs` reaches `Clustering.kmeans` unchanged

The call splats `kwargs` as it is. The constructor checks only that a `weights` entry is a non-empty `AbstractVector`, and it does not check the length. `Clustering.kmeans` needs one weight per point, which here is one weight per asset.

[`factory`](@ref) never writes observation weights into `kwargs`. An observation weight has no meaning here, because every step after `ce` reads an `assets x assets` matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KMeansAlgorithm(;
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        kwargs::NamedTuple = (;)
    ) -> KMeansAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - If `kwargs` holds `weights`, it must be a non-empty `AbstractVector`. The constructor does not check its length. `Clustering.kmeans` raises a `DimensionMismatch` for a wrong length when it runs.

# Examples

```jldoctest
julia> KMeansAlgorithm()
KMeansAlgorithm
     rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
    seed ┼ nothing
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractNonHierarchicalClusteringAlgorithm`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`clusterise`](@ref)
  - [`get_k_clusters_from_alg`](@ref)
  - [`resolve_rng`](@ref)
  - [`Clustering.kmeans`](https://juliastats.org/Clustering.jl/stable/api/#Clustering.kmeans)

# References

  - $(ref_dict[:lloyd1982]) Section VI, Method I. $(ref_dict[:no_original_source])
  - $(ref_dict[:macqueen1967]) Sections 1 and 2.1, for the name and the problem in ``N`` dimensions.
"""
@concrete struct KMeansAlgorithm <: AbstractNonHierarchicalClusteringAlgorithm
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
    """
    Keyword arguments for [`Clustering.kmeans`](https://juliastats.org/Clustering.jl/stable/api/#Clustering.kmeans).
    """
    kwargs
    function KMeansAlgorithm(rng::Random.AbstractRNG, seed::Option{<:Integer},
                             kwargs::NamedTuple)
        if haskey(kwargs, :weights)
            @argcheck(isa(kwargs.weights, AbstractVector),
                      ArgumentError("kwargs.weights must be an AbstractVector of point weights, one element per asset. Got\nkwargs.weights => $(typeof(kwargs.weights))"))
            @argcheck(!isempty(kwargs.weights), IsEmptyError)
        end
        return new{typeof(rng), typeof(seed), typeof(kwargs)}(rng, seed, kwargs)
    end
end
function KMeansAlgorithm(; rng::Random.AbstractRNG = Random.default_rng(),
                         seed::Option{<:Integer} = nothing,
                         kwargs::NamedTuple = (;))::KMeansAlgorithm
    return KMeansAlgorithm(rng, seed, kwargs)
end
"""
    get_k_clusters_from_alg(alg, D, k)

Partition the assets into `k` clusters with a non-hierarchical algorithm.

A new subtype of [`AbstractNonHierarchicalClusteringAlgorithm`](@ref) needs a struct and one method of this function. The method returns the result that `# Returns` states.

# Algorithm

The [`KMeansAlgorithm`](@ref) method runs these steps.

 1. Combine `alg.rng` and `alg.seed` into `rng` with [`resolve_rng`](@ref). If `seed` is stated, `resolve_rng` copies `alg.rng` and seeds the copy, so `alg.rng` does not advance and two calls of one estimator draw the same starts. If `seed` is `nothing`, `rng` is `alg.rng` itself, so its state advances and the next call draws different starts.
 2. Call `Clustering.kmeans(D, k; rng = rng, alg.kwargs...)` and return the partition. Lloyd's algorithm runs inside that call, and `Clustering.jl` owns its steps.

# Arguments

  - `alg`: The non-hierarchical clustering algorithm.
  - `D`: The distance matrix, `assets x assets`. Its columns are the points to cluster.
  - `k`: The number of clusters to make.

# Returns

  - `res::Clustering.ClusteringResult`: The partition. It has at least the fields that [`optimal_number_clusters`](@ref) reads, which are `assignments` and, under [`SecondOrderDifference`](@ref), `costs`.

# Related

  - [`AbstractNonHierarchicalClusteringAlgorithm`](@ref)
  - [`KMeansAlgorithm`](@ref)
  - [`optimal_number_clusters`](@ref)
"""
function get_k_clusters_from_alg(alg::KMeansAlgorithm, D::MatNum, k::Integer)
    rng = resolve_rng(alg.rng, alg.seed)
    return Clustering.kmeans(D, k; rng = rng, alg.kwargs...)
end
"""
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:Integer},
                             alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference},
                             alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SilhouetteScore},
                             alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)

Select the number of clusters for a non-hierarchical algorithm, and return the partition at that number.

The `SecondOrderDifference` and `SilhouetteScore` methods cluster the distance matrix once per candidate count and score each partition. Each method returns the partition at the selected count together with that count. A flat partition cannot be cut again at another count, so the method returns the partition that it already made.

# No validity test

[`valid_k_clusters`](@ref) has no counterpart here. It rejects a count at which the dendrogram has no valid cut. A flat partition has no dendrogram, so each method takes the `argmax` of the scores as it is.

The dispersion under [`SecondOrderDifference`](@ref) is also a different quantity from the dispersion of the hierarchical branch. Here it is `onc.alg.alg` applied to the per-point costs of k-means, and not to the pairwise distances inside each cluster. The vector of costs has one entry per asset at every count, so a cluster of two assets never reduces it to one value. These are the dispersions at the counts 2 to 6 on a universe of 20 assets with two pairs of near-copies:

  - The hierarchical dispersions rise, 11.27, 26.04, 47.89, 112.96 and 114.11, and select 3.
  - The k-means dispersions fall, 6.58, 2.83, 2.57, 2.17 and 2.06, and select 2.

# Algorithm

The `Integer` method runs these steps.

 1. Read the stated count `onc.alg` into `k`, and the ceiling into `max_k`. The ceiling is `min(floor(Int, sqrt(N)), onc.max_k)`, where `N` is the number of assets. A `max_k` of `nothing` gives `floor(Int, sqrt(N))`.
 2. Lower `k` to `max_k` when it is larger.
 3. Cluster `D` once at `k` into `res` with [`get_k_clusters_from_alg`](@ref).
 4. Return `res` and `k`.

The `SecondOrderDifference` and `SilhouetteScore` methods run the steps that their own algorithm types state, with one difference. No dendrogram rejects a count, so each method takes the `argmax` as it is. Each method then returns `cluster_lvls[k]`, the run that it made at the selected count, together with `k`. It does not run the algorithm again.

# Arguments

  - `onc`: The estimator of the number of clusters.

      + `onc::OptimalNumberClusters{<:Any, <:Integer}`: Takes the stated `k`, lowered to the ceiling.
      + `onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference}`: Scores each count by the second-order difference of `onc.alg.alg` applied to the per-point costs of the run at that count, and takes the `argmax`.
      + `onc::OptimalNumberClusters{<:Any, <:SilhouetteScore}`: Scores each count by `onc.alg.alg` applied to the vector of per-asset silhouettes, and takes the `argmax`.

  - `alg`: The non-hierarchical clustering algorithm, for example [`KMeansAlgorithm`](@ref).

  - `D`: The distance matrix.

# Returns

  - `res::Clustering.ClusteringResult`: The partition at the selected count.
  - `k::Integer`: The selected number of clusters.

The method returns the two as a tuple. The hierarchical methods of the same name in `03_Hierarchical.jl` return `k` alone, because the caller can cut the dendrogram again at any count.

# Related

  - [`OptimalNumberClusters`](@ref)
  - [`KMeansAlgorithm`](@ref)
  - [`get_k_clusters_from_alg`](@ref)
"""
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:Integer},
                                 alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)
    k = onc.alg
    max_k = onc.max_k
    N = size(D, 1)
    if isnothing(max_k)
        max_k = floor(Int, sqrt(N))
    end
    max_k = min(floor(Int, sqrt(N)), max_k)
    if k > max_k
        k = max_k
    end
    res = get_k_clusters_from_alg(alg, D, k)
    return res, k
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference},
                                 alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)
    N = size(D, 1)
    max_k = isnothing(onc.max_k) ? floor(Int, sqrt(N)) : onc.max_k
    max_k = min(floor(Int, sqrt(N)), max_k)
    c1 = min(max_k + 2, N)
    cluster_lvls = [get_k_clusters_from_alg(alg, D, k) for k in 1:c1]
    measure_alg = onc.alg.alg
    W_list = Vector{eltype(D)}(undef, c1)
    W_list[1] = typemin(eltype(D))
    for i in 2:c1
        costs = cluster_lvls[i].costs
        W_list[i] = vec_to_real_measure(measure_alg, costs)
    end
    k = if c1 > 2
        gaps = @views W_list[1:(end - 2)] .+ W_list[3:end] .- 2 .* W_list[2:(end - 1)]
        all(!isfinite, gaps) ? length(gaps) : argmax(gaps)
    else
        # `c1 <= 2` only when `N <= 2`, where the ceiling is `1`. `c1` itself can be `2`.
        max_k
    end
    return cluster_lvls[k], k
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SilhouetteScore},
                                 alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)
    N = size(D, 1)
    max_k = isnothing(onc.max_k) ? floor(Int, sqrt(N)) : onc.max_k
    c1 = min(floor(Int, sqrt(N)), max_k)
    cluster_lvls = [get_k_clusters_from_alg(alg, D, k) for k in 1:c1]
    measure_alg = onc.alg.alg
    W_list = Vector{eltype(D)}(undef, c1)
    W_list[1] = typemin(eltype(D))
    for i in 2:c1
        sl = Clustering.silhouettes(cluster_lvls[i], D)
        W_list[i] = vec_to_real_measure(measure_alg, sl)
    end
    k = all(!isfinite, W_list) ? length(W_list) : argmax(W_list)
    return cluster_lvls[k], k
end
"""
    clusterise(cle::ClustersEstimator{<:Any, <:Any,
                                      <:AbstractNonHierarchicalClusteringAlgorithm, <:Any},
               X::MatNum; dims::Int = 1, kwargs...)

Cluster assets with a non-hierarchical algorithm and return a [`Clusters`](@ref) result.

Estimates the similarity and distance matrices from `X`, then passes `D` to [`optimal_number_clusters`](@ref), which selects `k` and returns the partition at `k`.

# Algorithm

 1. Estimate the similarity matrix `S` and the distance matrix `D` from `X` with [`cor_and_dist`](@ref), under `cle.de` and `cle.ce`.
 2. Pass `D` and the algorithm `cle.alg` to [`optimal_number_clusters`](@ref)`(cle.onc, cle.alg, D)`, which returns the partition `res` and its count `k`. No other clustering runs, because `res` is the partition at the selected count.
 3. Return `Clusters(; res = res, S = S, D = D, k = k)`. The field `P` keeps its default, `nothing`, because the clustering ran on `D` itself.

# Arguments

  - `cle`: The clustering estimator, with a non-hierarchical algorithm.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for [`cor_and_dist`](@ref).

# Returns

  - `res::Clusters`: The partition, the similarity matrix, the distance matrix and the number of clusters.

# Related

  - [`Clusters`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`AbstractNonHierarchicalClusteringAlgorithm`](@ref)
"""
function clusterise(cle::ClustersEstimator{<:Any, <:Any,
                                           <:AbstractNonHierarchicalClusteringAlgorithm,
                                           <:Any}, X::MatNum; dims::Int = 1, kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    res, k = optimal_number_clusters(cle.onc, cle.alg, D)
    return Clusters(; res = res, S = S, D = D, k = k)
end
"""
    Clustering.assignments(clr::Clusters{<:Clustering.Hclust, <:Any, <:Any, <:Any})
    Clustering.assignments(clr::Clusters{<:Clustering.ClusteringResult, <:Any, <:Any,
                                         <:Any})

Label every asset of a [`Clusters`](@ref) result with the cluster it belongs to.

The function has one method for each clustering family. The two methods differ only in the source of the labels. A dendrogram has no labels until a cut makes them, and a flat partition already holds them. `03_Hierarchical.jl` declares the method for a `Clustering.Hclust`, and this file declares the method for a `Clustering.ClusteringResult`.

# Algorithm

 1. If `clr.res` is a `Clustering.Hclust`, it is a dendrogram with no labels. Cut it at `clr.k` with `Clustering.cutree`, which gives one label per asset.
 2. If `clr.res` is a `Clustering.ClusteringResult`, one run at one count made it, and it holds its own labels. Read `clr.res.assignments`.

# Arguments

  - `clr`: The clustering result to label.

# Returns

  - `idx::AbstractVector{<:Integer}`: One label per asset, over `1:clr.k`, in the order of the asset axis of the universe.

# Related

  - [`Clusters`](@ref)
  - [`clusterise`](@ref)
  - [`optimal_number_clusters`](@ref)
"""
function Clustering.assignments(clr::Clusters{<:Clustering.ClusteringResult, <:Any, <:Any,
                                              <:Any})
    return clr.res.assignments
end

export ClustersEstimator, Clusters, KMeansAlgorithm
