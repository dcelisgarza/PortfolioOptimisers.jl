struct HierarchicalClustering{T1 <: Symbol} <: ClusteringAlgorithm
    linkage::T1
end
function HierarchicalClustering(; linkage::Symbol = :ward)
    return HierarchicalClustering{typeof(linkage)}(linkage)
end
struct ClusterNode{tid, tl, tr, td, tcnt}
    id::tid
    left::tl
    right::tr
    height::td
    level::tcnt
    function ClusterNode(id, left::Union{ClusterNode, Nothing} = nothing,
                         right::Union{ClusterNode, Nothing} = nothing, height::Real = 0.0,
                         level::Int = 1)
        ilevel = isnothing(left) ? level : (left.level + right.level)
        return new{typeof(id), typeof(left), typeof(right), typeof(height), typeof(level)}(id,
                                                                                           left,
                                                                                           right,
                                                                                           height,
                                                                                           ilevel)
    end
end
function is_leaf(a::ClusterNode)
    return isnothing(a.left)
end
function pre_order(a::ClusterNode, func::Function = x -> x.id)
    n = a.level
    curNode = Vector{ClusterNode}(undef, 2 * n)
    lvisited = Set()
    rvisited = Set()
    curNode[1] = a
    k = 1
    preorder = Int[]

    while k >= 1
        nd = curNode[k]
        ndid = nd.id
        if is_leaf(nd)
            push!(preorder, func(nd))
            k = k - 1
        else
            if ndid ∉ lvisited
                curNode[k + 1] = nd.left
                push!(lvisited, ndid)
                k = k + 1
            elseif ndid ∉ rvisited
                curNode[k + 1] = nd.right
                push!(rvisited, ndid)
                k = k + 1
                # If we've visited the left and right of this non-leaf
                # node already, go up in the tree.
            else
                k = k - 1
            end
        end
    end

    return preorder
end
function to_tree(a::Hclust)
    N = length(a.order)
    d = Vector{ClusterNode}(undef, 2 * N - 1)
    for i ∈ eachindex(a.order)
        d[i] = ClusterNode(i)
    end
    merges = a.merges
    heights = a.heights
    nd = nothing

    for (i, height) ∈ pairs(heights)
        fi = merges[i, 1]
        fj = merges[i, 2]

        fi = fi < 0 ? -fi : fi + N
        fj = fj < 0 ? -fj : fj + N

        nd = ClusterNode(i + N, d[fi], d[fj], height)
        d[N + i] = nd
    end
    return nd, d
end
struct HierarchicalClusteringResult{T1 <: Clustering.Hclust, T2 <: AbstractMatrix,
                                    T3 <: AbstractMatrix, T4 <: Integer} <:
       AbstractPortfolioOptimisersClusteringResult
    clustering::T1
    S::T2
    D::T3
    k::T4
end
function HierarchicalClusteringResult(; clustering::Clustering.Hclust, S::AbstractMatrix,
                                      D::AbstractMatrix, k::Integer)
    @smart_assert(!isempty(S) && !isempty(D))
    @smart_assert(k >= 1)
    return HierarchicalClusteringResult{typeof(clustering), typeof(S), typeof(D),
                                        typeof(k)}(clustering, S, D, k)
end
function _clusterise(alg::HierarchicalClustering, X::AbstractMatrix{<:Real};
                     ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                     de::PortfolioOptimisersUnionDistanceMetric = CanonicalDistance(),
                     nch::Union{<:Integer, NumberClustersHeuristic} = SecondOrderDifference(),
                     branchorder::Symbol = :optimal, dims::Int = 1)
    S = cor(ce, X; dims = dims)
    D = distance(de, S, X; dims = dims)
    clustering = hclust(D; linkage = alg.linkage, branchorder = branchorder)
    k = optimal_number_clusters(nch, clustering, D)
    return HierarchicalClusteringResult(; clustering = clustering, S = S, D = D, k = k)
end

export HierarchicalClustering, ClusterNode, HierarchicalClusteringResult
