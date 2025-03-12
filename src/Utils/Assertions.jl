function issquare(A::AbstractMatrix)
    @smart_assert(size(A, 1) == size(A, 2))
end
function issquarepermissive(A::AbstractMatrix)
    if !isempty(A)
        @smart_assert(size(A, 1) == size(A, 2))
    end
end
