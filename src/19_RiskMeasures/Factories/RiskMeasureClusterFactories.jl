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
function fourth_moment_index_factory(N::Integer, cluster::AbstractVector)
    idx = Vector{Int}(undef, 0)
    sizehint!(idx, length(cluster)^2)
    for c ∈ cluster
        append!(idx, (((c - 1) * N + 1):(c * N))[cluster])
    end
    return idx
end
