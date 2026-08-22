"""
$(DocStringExtensions.TYPEDEF)

Partitions assets into `k` groups by Lloyd's algorithm, with no dendrogram.

Runs [`Clustering.kmeans`](https://juliastats.org/Clustering.jl/stable/api/#Clustering.kmeans) over the **columns of the distance matrix**, so each asset is the point given by its distances to every asset, and two assets cluster together when they sit at similar distances from the rest of the universe. The result is a flat partition: there is no tree, so nothing downstream can cut it at a different `k`.

`rng` and `seed` are here because the algorithm is randomised. [`resolve_rng`](@ref) combines them at the point of use, so a stated `seed` makes a run reproducible without the caller building a generator.

# `kwargs` reaches `Clustering.kmeans` unchanged

Whatever `kwargs` holds is splatted into the call. The constructor checks only that a `weights` entry is a non-empty `AbstractVector`; it does not check the length, and `Clustering.kmeans` wants **one weight per point**, which here means one per asset.

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

  - If `kwargs` contains `weights`, it must be a non-empty `AbstractVector`. Its **length is not checked**; `Clustering.kmeans` raises the `DimensionMismatch` at the point of use.

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

  - $(ref_dict[:lloyd1982])
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

Partition the assets into `k` clusters with the given non-hierarchical algorithm.

The whole extension contract of [`AbstractNonHierarchicalClusteringAlgorithm`](@ref): a new member is a struct and one method of this function.

# Arguments

  - `alg`: Non-hierarchical clustering algorithm.
  - `D`: Distance matrix, `assets x assets`. Its **columns** are the points to cluster.
  - `k`: Number of clusters to produce.

# Returns

  - `res::Clustering.ClusteringResult`: The partition, carrying at least the fields [`optimal_number_clusters`](@ref) reads — `assignments` and, under [`SecondOrderDifference`](@ref), `costs`.

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

Run a non-hierarchical algorithm at every candidate `k` and keep the best one.

Clusters the distance matrix once per candidate count, scores the results, and returns the winning clustering together with its `k`. Both come back because a flat partition cannot be re-cut: unlike the hierarchical branch, the clustering *is* the choice of `k`.

# No validity test, and no tree to run one against

[`valid_k_clusters`](@ref) has no counterpart here. It rejects a count the dendrogram cannot be cut at, and a flat partition has no dendrogram, so the argmax is taken as it stands.

The dispersion under [`SecondOrderDifference`](@ref) is also a different quantity from the hierarchical branch's: it is `onc.alg.alg` applied to the k-means **per-point costs**, not to within-cluster pairwise distances. That vector has one entry per asset whatever the cut, so a cut never reduces a one-value vector here.

# Arguments

  - `onc`: Optimal number of clusters estimator.

      + `onc::OptimalNumberClusters{<:Any, <:Integer}`: Uses a fixed `k` directly, clamped to `max_k`.
      + `onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference}`: Scores each count by the two-difference gap statistic of `onc.alg.alg` applied to that run's per-point costs, and takes the argmax.
      + `onc::OptimalNumberClusters{<:Any, <:SilhouetteScore}`: Scores each count by `onc.alg.alg` applied to the vector of per-asset silhouettes, and takes the argmax.

  - `alg`: Non-hierarchical clustering algorithm (e.g., [`KMeansAlgorithm`](@ref)).

  - `D`: Pairwise distance matrix.

# Returns

  - `(res, k)`: The clustering result and optimal number of clusters.

# Related

  - [`OptimalNumberClusters`](@ref)
  - [`KMeansAlgorithm`](@ref)
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
    c1 = min(min(floor(Int, sqrt(N)), max_k) + 2, N)
    cluster_lvls = [get_k_clusters_from_alg(alg, D, k) for k in 1:c1]
    measure_alg = onc.alg.alg
    W_list = Vector{eltype(D)}(undef, c1)
    W_list[1] = typemin(eltype(D))
    for i in 2:c1
        costs = cluster_lvls[i].costs
        W_list[i] = vec_to_real_measure(measure_alg, costs)
    end
    k = if c1 > 2
        gaps = W_list[1:(end - 2)] + W_list[3:end] - 2 * W_list[2:(end - 1)]
        all(!isfinite, gaps) ? length(gaps) : argmax(gaps)
    else
        c1
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

Estimates the similarity and distance matrices from `X`, then hands `D` to [`optimal_number_clusters`](@ref), which chooses `k` and returns the clustering for it. `P` is left as `nothing`, because the clustering ran on `D` itself.

# Arguments

  - `cle`: Clustering estimator configured with a non-hierarchical algorithm.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimators.

# Returns

  - `res::Clusters`: Clustering result containing the result, similarity and distance matrices, and number of clusters.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Return the cluster assignments for a non-hierarchical [`Clusters`](@ref) result.

# Related

  - [`Clusters`](@ref)
"""
function Clustering.assignments(clr::Clusters{<:Clustering.ClusteringResult, <:Any, <:Any,
                                              <:Any})
    return clr.res.assignments
end

export ClustersEstimator, Clusters, KMeansAlgorithm
