abstract type AbstractPriorEstimator <: AbstractEstimator end
abstract type AbstractLowOrderPriorEstimator <: AbstractPriorEstimator end
"""
Only uses asset returns
"""
abstract type AbstractLowOrderPriorEstimator_A <: AbstractLowOrderPriorEstimator end
"""
Uses asset and factor returns.
"""
abstract type AbstractLowOrderPriorEstimator_F <: AbstractLowOrderPriorEstimator end
"""
Uses asset returns and optionally factor returns.
"""
abstract type AbstractLowOrderPriorEstimator_AF <: AbstractLowOrderPriorEstimator end

const AbstractLowOrderPriorEstimator_A_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                  AbstractLowOrderPriorEstimator_AF}
const AbstractLowOrderPriorEstimator_F_AF = Union{AbstractLowOrderPriorEstimator_F,
                                                  AbstractLowOrderPriorEstimator_AF}
const AbstractLowOrderPriorEstimator_A_F_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                    AbstractLowOrderPriorEstimator_F,
                                                    AbstractLowOrderPriorEstimator_AF}

"""
"""
abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end
"""
"""
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
function phylogeny_matrix(necle::Union{<:AbstractNetworkEstimator,
                                       <:AbstractClusteringEstimator,
                                       <:AbstractClusteringResult}, pr::AbstractPriorResult;
                          kwargs...)
    return phylogeny_matrix(necle, pr.X; kwargs...)
end
function centrality_vector(necte::CentralityEstimator, pr::AbstractPriorResult; kwargs...)
    return centrality_vector(necte, pr.X; kwargs...)
end
function centrality_vector(ne::Union{<:AbstractNetworkEstimator,
                                     <:AbstractClusteringEstimator,
                                     <:AbstractClusteringResult},
                           cent::AbstractCentralityAlgorithm, pr::AbstractPriorResult;
                           kwargs...)
    return centrality_vector(ne, cent, pr.X; kwargs...)
end
function average_centrality(ne::Union{<:AbstractPhylogenyEstimator,
                                      <:AbstractPhylogenyResult},
                            cent::AbstractCentralityAlgorithm, w::AbstractVector,
                            pr::AbstractPriorResult; kwargs...)
    return dot(centrality_vector(ne, cent, pr.X; kwargs...).X, w)
end
function average_centrality(cte::CentralityEstimator, w::AbstractVector,
                            pr::AbstractPriorResult; kwargs...)
    return average_centrality(cte.ne, cte.cent, w, pr.X; kwargs...)
end

export prior
