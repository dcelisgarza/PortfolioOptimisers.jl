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
                                            alg::AbstractNonHierarchicalClusteringAlgorithm = HClustAlgorithm(),
                                            onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())
    return NonHierarchicalClusteringEstimator(alg, onc)
end

export NonHierarchicalClustering
