abstract type RegressionMethod end
struct LoadingsMatrix{T1 <: AbstractVector, T2 <: AbstractMatrix}
    c::T1
    M::T2
end
function LoadingsMatrix(; c::AbstractVector = Vector{Float64}(undef, 0),
                        M::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    if !isempty(c) && !isempty(M)
        @smart_assert(length(c) == size(M, 1))
    end
    return LoadingsMatrix{typeof(c), typeof(M)}(c, M)
end
function regression end

export regression, LoadingsMatrix
