⊗(A::AbstractArray, B::AbstractArray) = reshape(kron(B, A), (length(A), length(B)))
outer_prod(A::AbstractArray, B::AbstractArray) = reshape(kron(B, A), (length(A), length(B)))
