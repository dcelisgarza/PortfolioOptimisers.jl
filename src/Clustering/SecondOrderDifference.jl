struct SecondOrderDifference{T1 <: Integer} <: NumberClustersHeuristic
    max_k::T1
end
function SecondOrderDifference(; max_k::Integer = 0)
    @smart_assert(max_k >= zero(max_k))
    return SecondOrderDifference{typeof(max_k)}(max_k)
end

export SecondOrderDifference
