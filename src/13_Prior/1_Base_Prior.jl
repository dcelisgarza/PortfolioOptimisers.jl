abstract type AbstractPriorEstimator <: AbstractEstimator end
abstract type AbstractLowOrderPriorEstimator <: AbstractPriorEstimator end
abstract type AbstractLowOrderPriorEstimator_1_0 <: AbstractLowOrderPriorEstimator end
abstract type AbstractLowOrderPriorEstimator_2_1 <: AbstractLowOrderPriorEstimator end
abstract type AbstractLowOrderPriorEstimator_2_2 <: AbstractLowOrderPriorEstimator end
abstract type AbstractLowOrderPriorEstimator_1o2_1o2 <: AbstractLowOrderPriorEstimator end
const AbstractLowOrderPriorEstimatorMap_2_1 = Union{<:AbstractLowOrderPriorEstimator_1_0,
                                                    <:AbstractLowOrderPriorEstimator_1o2_1o2}
const AbstractLowOrderPriorEstimatorMap_2_2 = Union{<:AbstractLowOrderPriorEstimator_2_1,
                                                    <:AbstractLowOrderPriorEstimator_2_2,
                                                    <:AbstractLowOrderPriorEstimator_1o2_1o2}
const AbstractLowOrderPriorEstimatorMap_1o2_1o2 = Union{<:AbstractLowOrderPriorEstimator_1_0,
                                                        <:AbstractLowOrderPriorEstimator_2_1,
                                                        <:AbstractLowOrderPriorEstimator_2_2,
                                                        <:AbstractLowOrderPriorEstimator_1o2_1o2}
abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end
abstract type AbstractPriorResult <: AbstractResult end
function prior(pr::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
    return prior(pr, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
function prior_view(pr::AbstractPriorEstimator, args...; kwargs...)
    return pr
end
function prior(pr::AbstractPriorResult, args...; kwargs...)
    return pr
end
function clusterise(cle::ClusteringEstimator, pr::AbstractPriorResult; kwargs...)
    return clusterise(cle, pr.X; kwargs...)
end
function phylogeny_matrix(necle::Union{<:Network, <:ClusteringEstimator},
                          pr::AbstractPriorResult; kwargs...)
    return phylogeny_matrix(necle, pr.X; kwargs...)
end
function centrality_vector(necte::Union{<:Network, <:Centrality},
                           cent::AbstractCentralityAlgorithm, pr::AbstractPriorResult;
                           kwargs...)
    return centrality_vector(necte, pr.X; kwargs...)
end
function average_centrality(ne::Network, cent::AbstractCentralityAlgorithm,
                            w::AbstractVector, pr::AbstractPriorResult; kwargs...)
    return dot(centrality_vector(ne, cent, pr.X; kwargs...), w)
end
function average_centrality(cte::Centrality, w::AbstractVector, pr::AbstractPriorResult;
                            kwargs...)
    return average_centrality(cte.ne, cte.cent, w, pr.X; kwargs...)
end

export prior
