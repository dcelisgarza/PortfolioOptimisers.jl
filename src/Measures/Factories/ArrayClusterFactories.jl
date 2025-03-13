function cluster_real_or_vector_factory(x::Real, ::AbstractVector)
    return x
end
function cluster_real_or_vector_factory(x::AbstractVector{<:Real}, cluster::AbstractVector)
    return view(x, cluster)
end
