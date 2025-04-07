struct GeneralDistance{T1 <: AbstractDistanceAlgorithm, T2 <: Integer} <:
       AbstractDistanceEstimator
    alg::T1
    power::T2
end
function GeneralDistance(; alg::AbstractDistanceAlgorithm = SimpleDistance(),
                         power::Integer = 1)
    @smart_assert(power >= one(power))
    return GeneralDistance{typeof(alg), typeof(power)}(alg, power)
end
function fit_estimator(de::GeneralDistance{<:SimpleDistance, <:Any},
                       ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = robust_cor(ce, X; dims = dims) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end
function fit_estimator(de::GeneralDistance{<:SimpleDistance, <:Any}, rho::AbstractMatrix,
                       args...; kwargs...)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    scale = isodd(de.power) ? 0.5 : 1.0
    return sqrt.(clamp!((one(eltype(rho)) .- rho .^ de.power) * scale, zero(eltype(rho)),
                        one(eltype(rho))))
end
function fit_estimator(de::GeneralDistance{<:SimpleAbsoluteDistance, <:Any},
                       ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims)) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function fit_estimator(de::GeneralDistance{<:SimpleAbsoluteDistance, <:Any},
                       rho::AbstractMatrix, args...; kwargs...)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- abs.(rho) .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end
function fit_estimator(de::GeneralDistance{<:LogDistance, <:Any},
                       ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1)
    rho = abs.(robust_cor(ce, X; dims = dims)) .^ de.power
    return -log.(rho)
end
function fit_estimator(de::GeneralDistance{<:LogDistance, <:Any}, ce::LTDCovariance,
                       X::AbstractMatrix; dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims) .^ de.power
    return -log.(rho)
end
function fit_estimator(de::GeneralDistance{<:LogDistance, <:Any}, rho::AbstractMatrix,
                       args...; kwargs...)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(abs.(rho) .^ de.power)
end
function fit_estimator(de::GeneralDistance{<:VariationInfoDistance, <:Any}, ::Any,
                       X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise) .^ de.power
end
function fit_estimator(de::GeneralDistance{<:CorrelationDistance, <:Any},
                       ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1)
    rho = robust_cor(ce, X; dims = dims) .^ de.power
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function fit_estimator(de::GeneralDistance{<:CorrelationDistance, <:Any},
                       rho::AbstractMatrix, args...; kwargs...)
    issquare(rho)
    s = diag(rho)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end
function fit_estimator(de::GeneralDistance{<:CanonicalDistance, <:Any},
                       ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1)
    return fit_estimator(GeneralDistance(;
                                         alg = VariationInfoDistance(; bins = ce.bins,
                                                                     normalise = ce.normalise),
                                         power = de.power), ce, X; dims = dims)
end
function fit_estimator(de::GeneralDistance{<:CanonicalDistance, <:Any}, ce::LTDCovariance,
                       X::AbstractMatrix; dims::Int = 1)
    return fit_estimator(GeneralDistance(; alg = LogDistance(), power = de.power), ce, X;
                         dims = dims)
end
function fit_estimator(de::GeneralDistance{<:CanonicalDistance, <:Any},
                       ce::DistanceCovariance, X::AbstractMatrix; dims::Int = 1)
    return fit_estimator(GeneralDistance(; alg = CorrelationDistance(), power = de.power),
                         ce, X; dims = dims)
end
function fit_estimator(de::GeneralDistance{<:CanonicalDistance, <:Any},
                       ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1)
    return fit_estimator(GeneralDistance(; alg = SimpleDistance(), power = de.power), ce, X;
                         dims = dims)
end
function fit_estimator(de::GeneralDistance{<:CanonicalDistance, <:Any}, rho::AbstractMatrix,
                       args...; kwargs...)
    return fit_estimator(GeneralDistance(; alg = SimpleDistance(), power = de.power), rho)
end

export GeneralDistance
