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
function uncertainty_set_factory(::NoUncertaintySet, ::NoUncertaintySet, ::Any)
    return NoUncertaintySet()
end
function uncertainty_set_factory(::NoUncertaintySet,
                                 prior_uncertainty_set::BoxUncertaintySet{<:AbstractMatrix,
                                                                          <:AbstractMatrix},
                                 cluster::AbstractVector)
    return BoxUncertaintySet(; lo = view(prior_uncertainty_set.lo, cluster, cluster),
                             hi = view(prior_uncertainty_set.hi, cluster, cluster))
end
function uncertainty_set_factory(::NoUncertaintySet,
                                 prior_uncertainty_set::EllipseUncertaintySet{<:AbstractMatrix,
                                                                              <:Any},
                                 cluster::AbstractVector)
    idx = fourth_moment_cluster_index_factory(floor(Int,
                                                    sqrt(size(prior_uncertainty_set.sigma,
                                                              1))), cluster)
    return EllipseUncertaintySet(; sigma = prior_uncertainty_set.sigma[idx, idx],
                                 k = prior_uncertainty_set.k)
end
function uncertainty_set_factory(::NoUncertaintySet,
                                 prior_uncertainty_set::BoxUncertaintySet{<:AbstractVector,
                                                                          <:AbstractVector},
                                 cluster::AbstractVector)
    return BoxUncertaintySet(; lo = view(prior_uncertainty_set.lo, cluster),
                             hi = view(prior_uncertainty_set.hi, cluster))
end
function uncertainty_set_factory(::NoUncertaintySet,
                                 prior_uncertainty_set::EllipseUncertaintySet{<:AbstractVector,
                                                                              <:Any},
                                 cluster::AbstractVector)
    return EllipseUncertaintySet(; sigma = prior_uncertainty_set.sigma[cluster, cluster],
                                 k = prior_uncertainty_set.k)
end
function uncertainty_set_factory(risk_uncertainty_set::BoxUncertaintySet{<:AbstractMatrix,
                                                                         <:AbstractMatrix},
                                 ::Any, cluster::AbstractVector)
    return BoxUncertaintySet(; lo = view(risk_uncertainty_set.lo, cluster, cluster),
                             hi = view(risk_uncertainty_set.hi, cluster, cluster))
end
function uncertainty_set_factory(risk_uncertainty_set::EllipseUncertaintySet{<:AbstractMatrix,
                                                                             <:Any}, ::Any,
                                 cluster::AbstractVector)
    idx = fourth_moment_cluster_index_factory(floor(Int,
                                                    sqrt(size(risk_uncertainty_set.sigma,
                                                              1))), cluster)
    return EllipseUncertaintySet(; sigma = risk_uncertainty_set.sigma[idx, idx],
                                 k = risk_uncertainty_set.k)
end
function uncertainty_set_factory(risk_uncertainty_set::BoxUncertaintySet{<:AbstractVector,
                                                                         <:AbstractVector},
                                 ::Any, cluster::AbstractVector)
    return BoxUncertaintySet(; lo = view(risk_uncertainty_set.lo, cluster),
                             hi = view(risk_uncertainty_set.hi, cluster))
end
function uncertainty_set_factory(risk_uncertainty_set::EllipseUncertaintySet{<:AbstractVector,
                                                                             <:Any}, ::Any,
                                 cluster::AbstractVector)
    return EllipseUncertaintySet(; sigma = risk_uncertainty_set.sigma[cluster, cluster],
                                 k = risk_uncertainty_set.k)
end
