struct Distance{T1 <: AbstractDistanceAlgorithm} <: AbstractDistanceEstimator
    alg::T1
end
function Distance(; alg::AbstractDistanceAlgorithm = SimpleDistance())
    return Distance{typeof(alg)}(alg)
end
function distance(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho,
           sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end
function distance(::Distance{<:SimpleDistance}, rho::AbstractMatrix, args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!((one(eltype(rho)) .- rho) * 0.5, zero(eltype(rho)),
                        one(eltype(rho))))
end
function distance(::Distance{<:SimpleAbsoluteDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = abs.(cor(ce, X; dims = dims, kwargs...))
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(::Distance{<:SimpleAbsoluteDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    rho = abs.(cor(ce, X; dims = dims, kwargs...))
    return rho, sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function distance(::Distance{<:SimpleAbsoluteDistance}, rho::AbstractMatrix, args...;
                  kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- abs.(rho), zero(eltype(rho)), one(eltype(rho))))
end
function distance(::Distance{<:LogDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = abs.(cor(ce, X; dims = dims, kwargs...))
    return -log.(rho)
end
function cor_and_dist(::Distance{<:LogDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = abs.(cor(ce, X; dims = dims, kwargs...))
    return rho, -log.(rho)
end
function distance(::Distance{<:LogDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return -log.(rho)
end
function cor_and_dist(::Distance{<:LogDistance},
                      ce::Union{<:LTDCovariance,
                                <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, -log.(rho)
end
function distance(::Distance{<:LogDistance}, rho::AbstractMatrix, args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(abs.(rho))
end
function distance(de::Distance{<:VariationInfoDistance}, ::Any, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise)
end
function cor_and_dist(de::Distance{<:VariationInfoDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    @smart_assert(dims in (1, 2))
    rho = cor(ce, X; dims = dims, kwargs...)
    if dims == 2
        X = transpose(X)
    end
    return rho, variation_info(X, de.alg.bins, de.alg.normalise)
end
function distance(::Distance{<:CorrelationDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(::Distance{<:CorrelationDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function distance(::Distance{<:CorrelationDistance}, rho::AbstractMatrix, args...;
                  kwargs...)
    assert_matrix_issquare(rho)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end
function distance(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(;
                             alg = VariationInfoDistance(; bins = ce.bins,
                                                         normalise = ce.normalise)), ce, X;
                    dims = dims, kwargs...)
end
function distance(::Distance{<:CanonicalDistance},
                  ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(;
                             alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                         normalise = ce.ce.normalise)), ce,
                    X; dims = dims, kwargs...)
end
function cor_and_dist(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(;
                                 alg = VariationInfoDistance(; bins = ce.bins,
                                                             normalise = ce.normalise)), ce,
                        X; dims = dims, kwargs...)
end
function cor_and_dist(::Distance{<:CanonicalDistance},
                      ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(;
                                 alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                             normalise = ce.ce.normalise)),
                        ce, X; dims = dims, kwargs...)
end
function distance(::Distance{<:CanonicalDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(; alg = LogDistance()), ce, X; dims = dims, kwargs...)
end
function cor_and_dist(::Distance{<:CanonicalDistance},
                      ce::Union{<:LTDCovariance,
                                <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; alg = LogDistance()), ce, X; dims = dims, kwargs...)
end
function distance(::Distance{<:CanonicalDistance},
                  ce::Union{<:DistanceCovariance,
                            <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(; alg = CorrelationDistance()), ce, X; dims = dims, kwargs...)
end
function cor_and_dist(::Distance{<:CanonicalDistance},
                      ce::Union{<:DistanceCovariance,
                                <:PortfolioOptimisersCovariance{<:DistanceCovariance,
                                                                <:Any}}, X::AbstractMatrix;
                      dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; alg = CorrelationDistance()), ce, X; dims = dims,
                        kwargs...)
end
function distance(::Distance{<:CanonicalDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(; alg = SimpleDistance()), ce, X; dims = dims, kwargs...)
end
function cor_and_dist(::Distance{<:CanonicalDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; alg = SimpleDistance()), ce, X; dims = dims, kwargs...)
end
function distance(::Distance{<:CanonicalDistance}, rho::AbstractMatrix, args...; kwargs...)
    return distance(Distance(; alg = SimpleDistance()), rho; kwargs...)
end

export Distance
