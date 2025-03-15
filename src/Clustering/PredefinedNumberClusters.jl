struct PredefinedNumberClusters{T1 <: Integer, T2 <: Integer} <: NumberClustersHeuristic
    k::T1
    max_k::T2
end
function PredefinedNumberClusters(; k::Integer = 1, max_k::Integer = 0)
    @smart_assert(k >= one(k))
    @smart_assert(max_k >= zero(max_k))
    return PredefinedNumberClusters{typeof(k), typeof(max_k)}(k, max_k)
end

export PredefinedNumberClusters
