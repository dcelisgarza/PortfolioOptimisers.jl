struct ClusteringEstimator{T1 <: StatsBase.CovarianceEstimator,
                           T2 <: PortfolioOptimisersUnionDistanceMetric,
                           T3 <: ClusteringAlgorithm,
                           T4 <: Union{<:Integer, NumberClustersHeuristic}} <:
       AbstractClusteringEstimator
    ce::T1
    de::T2
    alg::T3
    nch::T4
end
function ClusteringEstimator(;
                             ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                             de::PortfolioOptimisersUnionDistanceMetric = CanonicalDistance(),
                             alg::ClusteringAlgorithm = HierarchicalClustering(),
                             nch::Union{<:Integer, NumberClustersHeuristic} = SecondOrderDifference())
    return ClusteringEstimator{typeof(ce), typeof(de), typeof(alg), typeof(nch)}(ce, de,
                                                                                 alg, nch)
end
function clusterise(cle::ClusteringEstimator, X::AbstractMatrix{<:Real};
                    branchorder::Symbol = :optimal, dims::Int = 1)
    return _clusterise(cle.alg, X; ce = cle.ce, de = cle.de, nch = cle.nch,
                       branchorder = branchorder, dims = dims)
end

export ClusteringEstimator, clusterise
