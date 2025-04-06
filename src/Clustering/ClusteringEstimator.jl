struct ClusteringEstimator{T1 <: StatsBase.CovarianceEstimator, T2 <: DistanceEstimator,
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
                             de::DistanceEstimator = CanonicalDistance(),
                             alg::ClusteringAlgorithm = HierarchicalClustering(),
                             nch::Union{<:Integer, NumberClustersHeuristic} = SecondOrderDifference())
    return ClusteringEstimator{typeof(ce), typeof(de), typeof(alg), typeof(nch)}(ce, de,
                                                                                 alg, nch)
end

export ClusteringEstimator
