struct PredefinedNumberClusters{T1 <: Integer, T2 <: Integer} <: NumberClustersHeuristic
    k::T1
    max_k::T2
end
function PredefinedNumberClusters(; k::Integer = 1, max_k::Integer = 0)
    @smart_assert(k >= 1)
    @smart_assert(max_k >= 0)
    return PredefinedNumberClusters{typeof(k), typeof(max_k)}(k, max_k)
end

export PredefinedNumberClusters
