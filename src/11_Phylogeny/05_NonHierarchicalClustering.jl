"""
"""
struct KMeansAlgorithm{T1, T2, T3} <: AbstractNonHierarchicalClusteringAlgorithm
    rng::T1
    seed::T2
    kwargs::T3
    function KMeansAlgorithm(rng::Random.AbstractRNG, seed::Option{<:Integer},
                             kwargs::NamedTuple)
        if haskey(kwargs, :weights)
            @argcheck(!isempty(kwargs.weights), IsEmptyError)
        end
        return new{typeof(rng), typeof(seed), typeof(kwargs)}(rng, seed, kwargs)
    end
end
function KMeansAlgorithm(; rng::Random.AbstractRNG = Random.default_rng(),
                         seed::Option{<:Integer} = nothing, kwargs::NamedTuple = (;))
    return KMeansAlgorithm(rng, seed, kwargs)
end
function factory(alg::KMeansAlgorithm, w::Option{<:StatsBase.AbstractWeights} = nothing)
    kwargs = if !isnothing(w)
        (; alg.kwargs..., weights = w)
    else
        alg.kwargs
    end
    return KMeansAlgorithm(; rng = alg.rng, seed = alg.seed, kwargs = kwargs)
end
"""
"""
struct NonHierarchicalClustering{T1, T2, T3, T4} <: AbstractNonHierarchicalClusteringResult
    clustering::T1
    S::T2
    D::T3
    k::T4
    function NonHierarchicalClustering(clustering::Clustering.Hclust, S::MatNum, D::MatNum,
                                       k::Integer)
        @argcheck(!isempty(S), IsEmptyError)
        @argcheck(!isempty(D), IsEmptyError)
        @argcheck(size(S) == size(D), DimensionMismatch)
        @argcheck(one(k) <= k, DomainError)
        return new{typeof(clustering), typeof(S), typeof(D), typeof(k)}(clustering, S, D, k)
    end
end
function NonHierarchicalClustering(; clustering::Clustering.Hclust, S::MatNum, D::MatNum,
                                   k::Integer)
    return NonHierarchicalClustering(clustering, S, D, k)
end
"""
"""
struct NonHierarchicalClusteringEstimator{T1, T2, T3, T4} <:
       AbstractNonHierarchicalClusteringEstimator
    ce::T1
    de::T2
    alg::T3
    onc::T4
    function NonHierarchicalClusteringEstimator(ce::StatsBase.CovarianceEstimator,
                                                de::AbstractDistanceEstimator,
                                                alg::AbstractHierarchicalClusteringAlgorithm,
                                                onc::AbstractOptimalNumberClustersEstimator)
        return new{typeof(ce), typeof(de), typeof(alg), typeof(onc)}(ce, de, alg, onc)
    end
end
function NonHierarchicalClusteringEstimator(;
                                            ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                            de::AbstractDistanceEstimator = Distance(;
                                                                                     alg = CanonicalDistance()),
                                            alg::AbstractHierarchicalClusteringAlgorithm = HClustAlgorithm(),
                                            onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())
    return NonHierarchicalClusteringEstimator(ce, de, alg, onc)
end
function factory(cle::NonHierarchicalClusteringEstimator,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)
    return NonHierarchicalClusteringEstimator(; ce = cle.ce, de = cle.de,
                                              alg = factory(cle.alg, w), onc = cle.onc)
end
function _get_k_clusters_from_alg(alg::KMeansAlgorithm, dist::MatNum, k::Integer)
    if !isnothing(alg.seed)
        Random.seed!(alg.rng, alg.seed)
    end
    return Clustering.kmeans(dist, k; alg.kwargs...)
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:Integer},
                                 alg::AbstractNonHierarchicalClusteringAlgorithm,
                                 dist::MatNum)
    k = onc.alg
    max_k = onc.max_k
    N = size(dist, 1)
    if isnothing(max_k)
        max_k = floor(Int, sqrt(N))
    end
    max_k = min(floor(Int, sqrt(N)), max_k)
    if k > max_k
        k = max_k
    end
    clustering = _get_k_clusters_from_alg(alg, dist, k)
    return clustering, k
end
function optimal_number_clusters(onc::OptimalNumberClusters{<:Any, <:SecondOrderDifference},
                                 alg::AbstractNonHierarchicalClusteringAlgorithm,
                                 dist::MatNum)
    N = size(dist, 1)
    max_k = isnothing(onc.max_k) ? floor(Int, sqrt(N)) : onc.max_k
    c1 = min(min(floor(Int, sqrt(N)), max_k) + 2, N)
    cluster_lvls = [_get_k_clusters_from_alg(alg, dist, k) for k in 1:c1]
    measure_alg = onc.alg.alg
    W_list = Vector{eltype(dist)}(undef, c1)
    W_list[1] = typemin(eltype(dist))
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
                                 alg::AbstractNonHierarchicalClusteringAlgorithm,
                                 dist::MatNum)
    N = size(dist, 1)
    max_k = isnothing(onc.max_k) ? floor(Int, sqrt(N)) : onc.max_k
    c1 = min(floor(Int, sqrt(N)), max_k)
    cluster_lvls = [_get_k_clusters_from_alg(alg, dist, k) for k in 1:c1]
    measure_alg = onc.alg.alg
    W_list = Vector{eltype(dist)}(undef, c1)
    for i in 2:c1
        sl = Clustering.silhouettes(cluster_lvls[i], dist)
        W_list[i] = vec_to_real_measure(measure_alg, sl)
    end
    k = all(!isfinite, W_list) ? length(W_list) : argmax(W_list)
    return cluster_lvls[k], k
end
function clusterise(cle::NonHierarchicalClusteringEstimator, X::MatNum; dims::Int = 1,
                    kwargs...)
    S, D = cor_and_dist(cle.de, cle.ce, X; dims = dims, kwargs...)
    clustering, k = optimal_number_clusters(cle.onc, cle.alg, D)
    return NonHierarchicalClustering(; clustering = clustering, S = S, D = D, k = k)
end
function get_clustering_indices(clr::NonHierarchicalClustering)
    return clr.clustering.assignments
end

export NonHierarchicalClusteringEstimator, NonHierarchicalClustering, KMeansAlgorithm
