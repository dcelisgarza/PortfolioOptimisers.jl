struct SecondOrderDifference{T1 <: Union{Nothing, <:Integer}} <: NumberClustersHeuristic
    max_k::T1
end
function SecondOrderDifference(; max_k::Union{Nothing, <:Integer} = nothing)
    if !isnothing(max_k)
        @smart_assert(max_k >= one(max_k))
    end
    return SecondOrderDifference{typeof(max_k)}(max_k)
end

export SecondOrderDifference
