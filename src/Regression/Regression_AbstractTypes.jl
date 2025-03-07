abstract type RegressionMethod end
struct LoadingsMatrix{T1 <: AbstractVector, T2 <: AbstractMatrix}
    b::T1
    M::T2
end
function LoadingsMatrix(; b::AbstractVector = Vector{Float64}(undef, 0),
                        M::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    if !isempty(b) && !isempty(M)
        @smart_assert(length(b) == size(M, 1))
    end
    return LoadingsMatrix{typeof(b), typeof(M)}(b, M)
end
function regression end

export regression, LoadingsMatrix
