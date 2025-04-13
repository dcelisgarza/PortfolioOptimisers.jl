function ucs_cluster_factory(::Nothing,
                             prior_ucs::Union{<:AbstractUncertaintySetResult,
                                              <:AbstractUncertaintySetEstimator}, ::Any)
    return prior_ucs
end
function ucs_cluster_factory(risk_ucs::Union{<:AbstractUncertaintySetResult,
                                             <:AbstractUncertaintySetEstimator}, ::Any,
                             ::Any)
    return risk_ucs
end
function ucs_cluster_factory(::Nothing, ::Nothing, ::Any)
    return nothing
end
function ucs_cluster_factory(::Nothing,
                             prior_ucs::BoxUncertaintySetResult{<:AbstractMatrix,
                                                                <:AbstractMatrix},
                             cluster::AbstractVector)
    return BoxUncertaintySetResult(; lb = view(prior_ucs.lb, cluster, cluster),
                                   ub = view(prior_ucs.ub, cluster, cluster))
end
function ucs_cluster_factory(::Nothing,
                             prior_ucs::EllipseUncertaintySetResult{<:AbstractMatrix,
                                                                    <:Any},
                             cluster::AbstractVector)
    idx = fourth_moment_index_factory(floor(Int, sqrt(size(prior_ucs.sigma, 1))), cluster)
    return EllipseUncertaintySetResult(; sigma = prior_ucs.sigma[idx, idx], k = prior_ucs.k)
end
function ucs_cluster_factory(::Nothing,
                             prior_ucs::BoxUncertaintySetResult{<:AbstractVector,
                                                                <:AbstractVector},
                             cluster::AbstractVector)
    return BoxUncertaintySetResult(; lb = view(prior_ucs.lb, cluster),
                                   ub = view(prior_ucs.ub, cluster))
end
function ucs_cluster_factory(::Nothing,
                             prior_ucs::EllipseUncertaintySetResult{<:AbstractVector,
                                                                    <:Any},
                             cluster::AbstractVector)
    return EllipseUncertaintySetResult(; sigma = prior_ucs.sigma[cluster, cluster],
                                       k = prior_ucs.k)
end
function ucs_cluster_factory(risk_ucs::BoxUncertaintySetResult{<:AbstractMatrix,
                                                               <:AbstractMatrix}, ::Any,
                             cluster::AbstractVector)
    return BoxUncertaintySetResult(; lb = view(risk_ucs.lb, cluster, cluster),
                                   ub = view(risk_ucs.ub, cluster, cluster))
end
function ucs_cluster_factory(risk_ucs::EllipseUncertaintySetResult{<:AbstractMatrix, <:Any},
                             ::Any, cluster::AbstractVector)
    idx = fourth_moment_index_factory(floor(Int, sqrt(size(risk_ucs.sigma, 1))), cluster)
    return EllipseUncertaintySetResult(; sigma = risk_ucs.sigma[idx, idx], k = risk_ucs.k)
end
function ucs_cluster_factory(risk_ucs::BoxUncertaintySetResult{<:AbstractVector,
                                                               <:AbstractVector}, ::Any,
                             cluster::AbstractVector)
    return BoxUncertaintySetResult(; lb = view(risk_ucs.lb, cluster),
                                   ub = view(risk_ucs.ub, cluster))
end
function ucs_cluster_factory(risk_ucs::EllipseUncertaintySetResult{<:AbstractVector, <:Any},
                             ::Any, cluster::AbstractVector)
    return EllipseUncertaintySetResult(; sigma = risk_ucs.sigma[cluster, cluster],
                                       k = risk_ucs.k)
end
