struct PredefinedNumberClusters{T1 <: Integer, T2 <: Integer} <: NumberClustersHeuristic
    k::T1
    max_k::T2
end
function PredefinedNumberClusters(; k::Integer = 1, max_k::Integer = 0)
    @smart_assert(k >= 1)
    @smart_assert(max_k >= 0)
    return PredefinedNumberClusters{typeof(k), typeof(max_k)}(k, max_k)
end
function optimal_number_clusters(nch::PredefinedNumberClusters, clustering::Hclust, args...)
    k = nch.k
    max_k = nch.max_k
    N = length(clustering.order)
    if iszero(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    max_k = min(ceil(Int, sqrt(N)), max_k)
    if k > max_k
        k = max_k
    end
    nodes = to_tree(clustering)[2]
    heights = [i.height for i ∈ nodes]
    nodes = nodes[sortperm(heights; rev = true)]
    flag = validate_k_value(clustering, nodes, k)
    if !flag
        # Above k
        flagu = false
        du = 0
        ku = k
        for i ∈ (k + 1):max_k
            flagu = validate_k_value(clustering, nodes, i)
            if flagu
                ku = i
                break
            end
        end
        if flagu
            du = ku - k
        end
        # Below k
        flagl = false
        dl = 0
        kl = k
        for i ∈ (k - 1):-1:1
            flagl = validate_k_value(clustering, nodes, i)
            if flagl
                kl = i
                break
            end
        end
        if flagl
            dl = k - kl
        end
        if du != 0 && dl == 0
            k = ku
        elseif du == 0 && dl != 0
            k = kl
        elseif du == dl
            k = max_k - ku > kl - 1 ? ku : kl
        else
            k = min(du, dl) == du ? ku : kl
        end
    end
    return k
end

export PredefinedNumberClusters
