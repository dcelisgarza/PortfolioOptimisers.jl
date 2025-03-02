struct OWA_NCRRA{T1 <: Real} <: AbstractOrderedWeightArrays
    g::T1
end
function OWA_NCRRA(; g::Real = 0.5)
    @smart_assert(zero(g) < g < one(g))
    return OWA_NCRRA{typeof(g)}(g)
end
function ncrra_weights(weights::AbstractMatrix{<:Real}, g::Real)
    k = size(weights, 2)
    phis = Vector{eltype(weights)}(undef, k - 1)
    e = 1
    for i ∈ eachindex(phis)
        e *= g + i - 1
        phis[i] = e / factorial(i + 1)
    end

    phis ./= sum(phis)
    a = weights * phis

    w = similar(a)
    w[1] = a[1]
    for i ∈ 2:length(a)
        w[i] = maximum(a[1:i])
    end

    return w
end
function owa_l_moment_crm(method::OWA_NCRRA, weights::AbstractMatrix{<:Real})
    return ncrra_weights(weights, method.g)
end

export OWA_NCRRA
