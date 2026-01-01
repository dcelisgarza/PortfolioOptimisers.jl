struct KMeansAlgorithm{T1, T2} <: AbstractNonHierarchicalClusteringAlgorithm
    w::T1
    kwargs::T2
    function KMeansAlgorithm(w::Option{<:StatsBase.AbstractWeights}, kwargs::NamedTuple)
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(w), typeof(kwargs)}(w, kwargs)
    end
end
function KMeansAlgorithm(; w::Option{<:StatsBase.AbstractWeights} = nothing,
                         kwargs::NamedTuple = (;))
    return KMeansAlgorithm(w, kwargs)
end
function factory(alg::KMeansAlgorithm, w::Option{<:StatsBase.AbstractWeights} = nothing)
    return KMeansAlgorithm(; w = ifelse(isnothing(alg.w), alg.w, w), kwargs = alg.kwargs)
end
struct NonHierarchicalClustering{T1, T2} <: AbstractNonHierarchicalClusteringResult
    clustering::T1
    k::T2
    function NonHierarchicalClustering(clustering::Clustering.ClusteringResult, k::Integer)
        @argcheck(one(k) <= k, DomainError)
        return new{typeof(clustering), typeof(k)}(clustering, k)
    end
end
function NonHierarchicalClustering(; clustering::Clustering.ClusteringResult, k::Integer)
    return NonHierarchicalClustering(clustering, k)
end
struct NonHierarchicalClusteringEstimator{T1, T2} <: AbstractClusteringEstimator
    alg::T1
    onc::T2
    function NonHierarchicalClusteringEstimator(alg::AbstractNonHierarchicalClusteringAlgorithm,
                                                onc::AbstractOptimalNumberClustersEstimator)
        return new{typeof(alg), typeof(onc)}(alg, onc)
    end
end
function NonHierarchicalClusteringEstimator(;
                                            alg::AbstractNonHierarchicalClusteringAlgorithm = KMeansAlgorithm(),
                                            onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())
    return NonHierarchicalClusteringEstimator(alg, onc)
end
function factory(cle::NonHierarchicalClusteringEstimator,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)
    return NonHierarchicalClusteringEstimator(; alg = factory(cle.alg, w), onc = cle.onc)
end
function optimal_number_clusters(cle::NonHierarchicalClusteringEstimator{<:KMeansAlgorithm,
                                                                         <:OptimalNumberClusters{<:Any,
                                                                                                 <:Integer}},
                                 X::MatNum)
    onc = cle.onc
    k = onc.alg
    max_k = onc.max_k
    N = size(X, 1)
    if isnothing(max_k)
        max_k = floor(Int, sqrt(N))
    end
    max_k = min(floor(Int, sqrt(N)), max_k)
    if k > max_k
        k = max_k
    end
    clustering = kmeans(X, k; weights = cle.alg.w, cle.alg.kwargs...)
    return clustering, k
end
function optimal_number_clusters(cle::NonHierarchicalClusteringEstimator{<:KMeansAlgorithm,
                                                                         <:OptimalNumberClusters{<:Any,
                                                                                                 <:SecondOrderDifference}},
                                 X::MatNum)
    N = size(X, 2)
    max_k = isnothing(cle.onc.max_k) ? floor(Int, sqrt(N)) : cle.onc.max_k
    c1 = min(min(floor(Int, sqrt(N)), max_k) + 2, N)
    cluster_lvls = [Clustering.kmeans(X, i; weights = cle.alg.w, cle.alg.kwargs...)
                    for i in 1:c1]
    measure_alg = cle.onc.alg.alg
    W_list = Vector{eltype(X)}(undef, c1)
    W_list[1] = typemin(eltype(X))
    for i in 2:c1
        lvl = cluster_lvls[i].costs
        costs = lvl.costs
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
function optimal_number_clusters(cle::NonHierarchicalClusteringEstimator{<:KMeansAlgorithm,
                                                                         <:OptimalNumberClusters{<:Any,
                                                                                                 <:SilhouetteScore}},
                                 X::MatNum)
    N = size(X, 2)
    max_k = isnothing(cle.onc.max_k) ? floor(Int, sqrt(N)) : cle.onc.max_k
    c1 = min(floor(Int, sqrt(N)), max_k)
    cluster_lvls = [Clustering.kmeans(X, i; weights = cle.alg.w, cle.alg.kwargs...)
                    for i in 1:c1]
    measure_alg = cle.onc.alg.alg
    W_list = Vector{eltype(X)}(undef, c1)
    metric = ifelse(isnothing(cle.onc.alg.metric), Distances.SqEuclidean(),
                    cle.onc.alg.metric)
    for i in 2:c1
        lvl = cluster_lvls[i]
        assignments = lvl.assignments
        sl = Clustering.silhouettes(assignments, X; metric = metric)
        W_list[i] = vec_to_real_measure(measure_alg, sl)
    end
    k = all(!isfinite, W_list) ? length(W_list) : argmax(W_list)
    return cluster_lvls[k], k
end
function clusterise(cle::NonHierarchicalClusteringEstimator{<:KMeansAlgorithm}, X::MatNum;
                    dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2), DomainError)
    if dims == 2
        X = transpose(X)
    end
    clustering, k = optimal_number_clusters(cle, X)
    return NonHierarchicalClustering(; clustering = clustering, k = k)
end
function get_clustering_indices(clr::NonHierarchicalClustering)
    return clr.clustering.assignments
end

export NonHierarchicalClusteringEstimator, NonHierarchicalClustering, KMeansAlgorithm
