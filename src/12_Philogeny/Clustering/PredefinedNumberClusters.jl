struct PredefinedNumberClusters{T1 <: Integer, T2 <: Union{Nothing, <:Integer}} <:
       NumberClustersHeuristic
    k::T1
    max_k::T2
end
function PredefinedNumberClusters(; k::Integer = 1,
                                  max_k::Union{Nothing, <:Integer} = nothing)
    @smart_assert(k >= one(k))
    if !isnothing(max_k)
        @smart_assert(max_k >= one(max_k))
    end
    return PredefinedNumberClusters{typeof(k), typeof(max_k)}(k, max_k)
end

export PredefinedNumberClusters
