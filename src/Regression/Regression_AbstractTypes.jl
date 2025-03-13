abstract type RegressionMethod end
struct LoadingsMatrix{T1 <: Union{Nothing, <:AbstractVector}, T2 <: AbstractMatrix}
    b::T1
    M::T2
end
function LoadingsMatrix(; b::Union{Nothing, <:AbstractVector}, M::AbstractMatrix)
    @smart_assert(!isempty(M))
    if isa(b, AbstractVector)
        @smart_assert(!isempty(b))
        @smart_assert(length(b) == size(M, 1))
    end
    return LoadingsMatrix{typeof(b), typeof(M)}(b, M)
end
function regression end

export regression, LoadingsMatrix
