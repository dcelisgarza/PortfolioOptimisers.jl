function issquare(A::AbstractMatrix)
    @smart_assert(size(A, 1) == size(A, 2))
end
