function risk_measure_nothing_vec_factory(::Nothing, ::Nothing, cluster::AbstractVector)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_vec_factory(risk_variable::AbstractVector{<:Real}, ::Any,
                                          cluster::AbstractVector)
    return view(risk_variable, cluster)
end
function risk_measure_nothing_vec_factory(::Nothing, prior_variable::AbstractVector{<:Real},
                                          cluster::AbstractVector)
    return view(prior_variable, cluster)
end
function risk_measure_nothing_matrix_factory(::Nothing, ::Nothing, cluster::AbstractVector)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_matrix_factory(risk_variable::AbstractMatrix{<:Real}, ::Any,
                                             cluster::AbstractVector)
    return view(risk_variable, cluster, cluster)
end
function risk_measure_nothing_matrix_factory(::Nothing, prior_variable::AbstractMatrix,
                                             cluster::AbstractVector)
    return view(prior_variable, cluster, cluster)
end
function fourth_moment_cluster_index_factory(N::Integer, cluster::AbstractVector)
    idx = Vector{Int}(undef, 0)
    sizehint!(idx, length(cluster)^2)
    for c ∈ cluster
        append!(idx, (((c - 1) * N + 1):(c * N))[cluster])
    end
    return idx
end

function ucs_cluster_factory(::Nothing,
                             prior_ucs::Union{<:UncertaintySet, <:UncertaintySetEstimator},
                             ::Any)
    return prior_ucs
end
function ucs_cluster_factory(risk_ucs::Union{<:UncertaintySet, <:UncertaintySetEstimator},
                             ::Any, ::Any)
    return risk_ucs
end
function ucs_cluster_factory(::Nothing, ::Nothing, ::Any)
    return nothing
end
function ucs_cluster_factory(::Nothing,
                             prior_ucs::BoxUncertaintySet{<:AbstractMatrix,
                                                          <:AbstractMatrix},
                             cluster::AbstractVector)
    return BoxUncertaintySet(; lb = view(prior_ucs.lb, cluster, cluster),
                             ub = view(prior_ucs.ub, cluster, cluster))
end
function ucs_cluster_factory(::Nothing,
                             prior_ucs::EllipseUncertaintySet{<:AbstractMatrix, <:Any},
                             cluster::AbstractVector)
    idx = fourth_moment_cluster_index_factory(floor(Int, sqrt(size(prior_ucs.sigma, 1))),
                                              cluster)
    return EllipseUncertaintySet(; sigma = prior_ucs.sigma[idx, idx], k = prior_ucs.k)
end
function ucs_cluster_factory(::Nothing,
                             prior_ucs::BoxUncertaintySet{<:AbstractVector,
                                                          <:AbstractVector},
                             cluster::AbstractVector)
    return BoxUncertaintySet(; lb = view(prior_ucs.lb, cluster),
                             ub = view(prior_ucs.ub, cluster))
end
function ucs_cluster_factory(::Nothing,
                             prior_ucs::EllipseUncertaintySet{<:AbstractVector, <:Any},
                             cluster::AbstractVector)
    return EllipseUncertaintySet(; sigma = prior_ucs.sigma[cluster, cluster],
                                 k = prior_ucs.k)
end
function ucs_cluster_factory(risk_ucs::BoxUncertaintySet{<:AbstractMatrix,
                                                         <:AbstractMatrix}, ::Any,
                             cluster::AbstractVector)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, cluster, cluster),
                             ub = view(risk_ucs.ub, cluster, cluster))
end
function ucs_cluster_factory(risk_ucs::EllipseUncertaintySet{<:AbstractMatrix, <:Any},
                             ::Any, cluster::AbstractVector)
    idx = fourth_moment_cluster_index_factory(floor(Int, sqrt(size(risk_ucs.sigma, 1))),
                                              cluster)
    return EllipseUncertaintySet(; sigma = risk_ucs.sigma[idx, idx], k = risk_ucs.k)
end
function ucs_cluster_factory(risk_ucs::BoxUncertaintySet{<:AbstractVector,
                                                         <:AbstractVector}, ::Any,
                             cluster::AbstractVector)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, cluster),
                             ub = view(risk_ucs.ub, cluster))
end
function ucs_cluster_factory(risk_ucs::EllipseUncertaintySet{<:AbstractVector, <:Any},
                             ::Any, cluster::AbstractVector)
    return EllipseUncertaintySet(; sigma = risk_ucs.sigma[cluster, cluster], k = risk_ucs.k)
end
