abstract type AdjacencyConstraint end
struct NoAdj <: AdjacencyConstraint end
struct SDP{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <: AdjacencyConstraint
    A::T1
    p::T2
end
function SDP(; A::AbstractMatrix{<:Real}, p::Real = 0.05)
    @smart_assert(!isempty(A))
    @smart_assert(p >= zero(p))
    return SDP{typeof(A), typeof(p)}(A, p)
end
struct IP{T1 <: AbstractMatrix{<:Real}, T2 <: Union{<:Integer, <:AbstractVector{<:Integer}},
          T3 <: Real} <: AdjacencyConstraint
    A::T1
    B::T2
    scale::T3
end
function IP(; A::AbstractMatrix{<:Real},
            B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1, scale::Real = 100_000.0)
    @smart_assert(!isempty(A))
    A = unique(A + I; dims = 1)
    if isa(B, AbstractVector)
        @smart_assert(!isempty(B) && size(A, 1) == length(B) && all(B .> zero(B)))
    else
        @smart_assert(B > zero(B))
    end
    return IP{typeof(A), typeof(B), typeof(scale)}(A, B, scale)
end
