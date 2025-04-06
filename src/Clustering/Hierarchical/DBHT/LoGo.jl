function fit_estimator!(::Nothing, args...; kwargs...)
    return nothing
end
struct LoGo{T1 <: DistanceEstimator, T2 <: SimilarityMatrixEstimator}
    dist::T1
    sim::T2
end
function LoGo(; dist::DistanceEstimator = CanonicalDistance(),
              sim::SimilarityMatrixEstimator = DBHT_MaximumDistanceSimilarity())
    return LoGo{typeof(dist), typeof(sim)}(dist, sim)
end
function LoGo_dist_assert(::Union{VariationInfoDistance, VariationInfoDistanceDistance},
                          sigma::AbstractMatrix, X::AbstractMatrix)
    @smart_assert(size(sigma, 1) == size(X, 2))
    return nothing
end
function LoGo_dist_assert(args...)
    return nothing
end
function fit_estimator!(je::LoGo, pdm::Union{Nothing, <:PosDefEstimator},
                        sigma::AbstractMatrix, X::AbstractMatrix; dims::Int = 1)
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
    S = dbht_similarity(je.sim, S, D)
    separators, cliques = PMFG_T2s(S, 4)[3:4]
    sigma .= J_LoGo(sigma, separators, cliques) \ I
    fit_estimator!(pdm, sigma)
    return nothing
end

export fit_estimator!, LoGo
