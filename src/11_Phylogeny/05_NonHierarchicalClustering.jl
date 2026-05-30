"""
$(DocStringExtensions.TYPEDEF)

K-means clustering algorithm configuration for non-hierarchical clustering in `PortfolioOptimisers.jl`.

`KMeansAlgorithm` is a composable clustering algorithm type that specifies the use of the k-means algorithm (via [`Clustering.kmeans`](https://juliastats.org/Clustering.jl/stable/api/#Clustering.kmeans)) for constructing non-hierarchical clusterings from a distance matrix.

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

  - If `kwargs` contains `weights`, it must be a non-empty `AbstractVector`.

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
  - [`Clustering.kmeans`](https://juliastats.org/Clustering.jl/stable/api/#Clustering.kmeans)
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
            @argcheck(isa(kwargs.weights, AbstractVector), TypeError)
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
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`KMeansAlgorithm`](@ref) with observation weights `w` added to the `kwargs` field.

# Related

  - [`KMeansAlgorithm`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::KMeansAlgorithm, w::StatsBase.AbstractWeights)::KMeansAlgorithm
    return KMeansAlgorithm(; rng = alg.rng, seed = alg.seed,
                           kwargs = (; alg.kwargs..., weights = w))
end
"""
    _get_k_clusters_from_alg(alg, D, k)

Assign observations to `k` clusters using the specified clustering algorithm and distance matrix.

Internal function used by non-hierarchical clustering estimators.

# Arguments

  - `alg`: Clustering algorithm (e.g., [`KMeansAlgorithm`](@ref)).
  - `D`: Pairwise distance matrix.
  - `k`: Number of clusters.

# Returns

  - Cluster assignments.

# Related

  - [`KMeansAlgorithm`](@ref)
"""
function _get_k_clusters_from_alg(alg::KMeansAlgorithm, D::MatNum, k::Integer)
    if !isnothing(alg.seed)
        Random.seed!(alg.rng, alg.seed)
    end
    return Clustering.kmeans(D, k; alg.kwargs...)
end
"""
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:Integer},
                             alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference},
                             alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)
    optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SilhouetteScore},
                             alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)

Select the optimal number of clusters for a non-hierarchical clustering algorithm.

Dispatches on `onc.alg` to apply the configured selection strategy and returns both the best clustering result and the optimal `k`.

# Arguments

  - `onc`: Optimal number of clusters estimator.

      + `onc::OptimalNumberClusters{<:Any, <:Integer}`: Uses a fixed `k` directly, clamped to `max_k`.
      + `onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference}`: Selects `k` by maximising the second-order difference of per-cluster costs.
      + `onc::OptimalNumberClusters{<:Any, <:SilhouetteScore}`: Selects `k` by maximising the mean silhouette score.

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
    res = _get_k_clusters_from_alg(alg, D, k)
    return res, k
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference},
                                 alg::AbstractNonHierarchicalClusteringAlgorithm, D::MatNum)
    N = size(D, 1)
    max_k = isnothing(onc.max_k) ? floor(Int, sqrt(N)) : onc.max_k
    c1 = min(min(floor(Int, sqrt(N)), max_k) + 2, N)
    cluster_lvls = [_get_k_clusters_from_alg(alg, D, k) for k in 1:c1]
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
    cluster_lvls = [_get_k_clusters_from_alg(alg, D, k) for k in 1:c1]
    measure_alg = onc.alg.alg
    W_list = Vector{eltype(D)}(undef, c1)
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

Run non-hierarchical clustering and return the result as a [`Clusters`](@ref) object.

Computes the similarity and distance matrices from `X`, selects the optimal number of clusters, and returns a [`Clusters`](@ref) result.

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
