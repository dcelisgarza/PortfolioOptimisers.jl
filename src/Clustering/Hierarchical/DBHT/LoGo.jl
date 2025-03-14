abstract type AbstractLoGo end
struct NoLoGo <: AbstractLoGo end
function LoGo!(::NoLoGo, args...; kwargs...)
    return nothing
end
struct LoGo{T1 <: PortfolioOptimisersUnionDistanceMetric,
            T2 <: SimilarityMatrixEstimator} <: AbstractLoGo
    dist::T1
    similarity::T2
end
function LoGo(; dist::PortfolioOptimisersUnionDistanceMetric = CanonicalDistance(),
              similarity::SimilarityMatrixEstimator = DBHT_MaximumDistanceSimilarity())
    return LoGo{typeof(dist), typeof(similarity)}(dist, similarity)
end
function LoGo_dist_assert(::Union{VariationInfoDistance, VariationInfoDistanceDistance},
                          sigma::AbstractMatrix, X::AbstractMatrix)
    @smart_assert(size(sigma, 1) == size(X, 2))
    return nothing
end
function LoGo_dist_assert(args...)
    return nothing
end
function LoGo!(je::LoGo, fnpdm::FixNonPositiveDefiniteMatrix, sigma::AbstractMatrix,
               X::AbstractMatrix; dims::Int = 1)
    issquare(sigma)
    LoGo_dist_assert(je.dist, sigma, X)
    s = diag(sigma)
    iscov = any(.!isone.(s))
    S = if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor(sigma, s)
    else
        sigma
    end
    D = distance(je.dist, S, X; dims = dims)
    S = dbht_similarity(je.similarity, S, D)
    separators, cliques = PMFG_T2s(S, 4)[3:4]
    sigma .= J_LoGo(sigma, separators, cliques) \ I
    fix_non_positive_definite_matrix!(fnpdm, sigma)
    return nothing
end

export NoLoGo, LoGo!, LoGo
