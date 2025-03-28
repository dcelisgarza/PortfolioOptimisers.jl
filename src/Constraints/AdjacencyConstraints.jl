abstract type AdjacencyConstraintModel end
struct SemiDefiniteAdjacency{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <:
       AdjacencyConstraintModel
    A::T1
    p::T2
end
function SemiDefiniteAdjacency(; A::AbstractMatrix{<:Real}, p::Real = 0.05)
    @smart_assert(!isempty(A))
    @smart_assert(p >= zero(p))
    return SemiDefiniteAdjacency{typeof(A), typeof(p)}(A, p)
end
struct IntegerAdjacency{T1 <: AbstractMatrix{<:Real},
                        T2 <: Union{<:Integer, <:AbstractVector{<:Integer}}, T3 <: Real} <:
       AdjacencyConstraintModel
    A::T1
    B::T2
    scale::T3
end
function IntegerAdjacency(; A::AbstractMatrix{<:Real},
                          B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                          scale::Real = 100_000.0)
    @smart_assert(!isempty(A))
    A = unique(A + I; dims = 1)
    if isa(B, AbstractVector)
        @smart_assert(!isempty(B) && size(A, 1) == length(B) && all(B .> zero(B)))
    else
        @smart_assert(B > zero(B))
    end
    return IntegerAdjacency{typeof(A), typeof(B), typeof(scale)}(A, B, scale)
end
struct AdjacencyConstraint
    # Union{<:NetworkEstimator, <:ClusteringEstimator}
end
struct CentralityConstraint end
#! Similar to linear constraint but using centrality vectors
function centrality_constraints(::Union{<:CentralityConstraint,
                                        <:AbstractVector{<:CentralityConstraint}},
                                sets::DataFrame; kwargs...)
    return nothing
end
function centrality_constraints(::Nothing, args...; kwargs...)
    return nothing
end
