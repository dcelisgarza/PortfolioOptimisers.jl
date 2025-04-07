struct Distance{T1 <: AbstractDistanceAlgorithm} <: AbstractDistanceEstimator
    alg::T1
end
function Distance(; alg::AbstractDistanceAlgorithm = SimpleDistance())
    return Distance{typeof(alg)}(alg)
end
function fit_estimator(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                       X::AbstractMatrix; dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims)
    return sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end
function fit_estimator(::Distance{<:SimpleDistance}, rho::AbstractMatrix, args...;
                       kwargs...)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!((one(eltype(rho)) .- rho) * 0.5, zero(eltype(rho)),
                        one(eltype(rho))))
end
function fit_estimator(::Distance{<:SimpleAbsoluteDistance},
                       ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims))
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function fit_estimator(::Distance{<:SimpleAbsoluteDistance}, rho::AbstractMatrix, args...;
                       kwargs...)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- abs.(rho), zero(eltype(rho)), one(eltype(rho))))
end
function fit_estimator(::Distance{<:LogDistance}, ce::StatsBase.CovarianceEstimator,
                       X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims))
    return -log.(rho)
end
function fit_estimator(::Distance{<:LogDistance}, ce::LTDCovariance, X::AbstractMatrix;
                       dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims)
    return -log.(rho)
end
function fit_estimator(::Distance{<:LogDistance}, rho::AbstractMatrix, args...; kwargs...)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(abs.(rho))
end
function fit_estimator(de::Distance{<:VariationInfoDistance}, ::Any, X::AbstractMatrix;
                       dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise)
end
function fit_estimator(::Distance{<:CorrelationDistance}, ce::StatsBase.CovarianceEstimator,
                       X::AbstractMatrix; dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims)
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function fit_estimator(::Distance{<:CorrelationDistance}, rho::AbstractMatrix, args...;
                       kwargs...)
    issquare(rho)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end
function fit_estimator(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance,
                       X::AbstractMatrix; dims::Int = 1)
    return fit_estimator(Distance(;
                                  alg = VariationInfoDistance(; bins = ce.bins,
                                                              normalise = ce.normalise)),
                         ce, X; dims = dims)
end
function fit_estimator(::Distance{<:CanonicalDistance}, ce::LTDCovariance,
                       X::AbstractMatrix; dims::Int = 1)
    return fit_estimator(Distance(; alg = LogDistance()), ce, X; dims = dims)
end
function fit_estimator(::Distance{<:CanonicalDistance}, ce::DistanceCovariance,
                       X::AbstractMatrix; dims::Int = 1)
    return fit_estimator(Distance(; alg = CorrelationDistance()), ce, X; dims = dims)
end
function fit_estimator(::Distance{<:CanonicalDistance}, ce::StatsBase.CovarianceEstimator,
                       X::AbstractMatrix; dims::Int = 1)
    return fit_estimator(Distance(; alg = SimpleDistance()), ce, X; dims = dims)
end
function fit_estimator(::Distance{<:CanonicalDistance}, rho::AbstractMatrix, args...;
                       kwargs...)
    return fit_estimator(Distance(; alg = SimpleDistance()), rho)
end

export Distance
