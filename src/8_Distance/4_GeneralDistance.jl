struct GeneralDistance{T1 <: Integer, T2 <: AbstractDistanceAlgorithm} <:
       AbstractDistanceEstimator
    power::T1
    alg::T2
end
function GeneralDistance(; power::Integer = 1,
                         alg::AbstractDistanceAlgorithm = SimpleDistance())
    @smart_assert(power >= one(power))
    return GeneralDistance{typeof(power), typeof(alg)}(power, alg)
end
function distance(de::GeneralDistance{<:Any, <:SimpleDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end
function distance(de::GeneralDistance{<:Any, <:SimpleDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    scale = isodd(de.power) ? 0.5 : 1.0
    return sqrt.(clamp!((one(eltype(rho)) .- rho .^ de.power) * scale, zero(eltype(rho)),
                        one(eltype(rho))))
end
function distance(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = abs.(cor(ce, X; dims = dims), kwargs...) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function distance(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- abs.(rho) .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end
function distance(de::GeneralDistance{<:Any, <:LogDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = abs.(cor(ce, X; dims = dims, kwargs...)) .^ de.power
    return -log.(rho)
end
function distance(de::GeneralDistance{<:Any, <:LogDistance}, ce::LTDCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return -log.(rho)
end
function distance(de::GeneralDistance{<:Any, <:LogDistance}, rho::AbstractMatrix, args...;
                  kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(abs.(rho) .^ de.power)
end
function distance(de::GeneralDistance{<:Any, <:VariationInfoDistance}, ::Any,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise) .^ de.power
end
function distance(de::GeneralDistance{<:Any, <:CorrelationDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function distance(de::GeneralDistance{<:Any, <:CorrelationDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    issquare(rho)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power,
                                    alg = VariationInfoDistance(; bins = ce.bins,
                                                                normalise = ce.normalise)),
                    ce, X; dims = dims, kwargs...)
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, ce::LTDCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = LogDistance()), ce, X;
                    dims = dims, kwargs...)
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, ce::DistanceCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = CorrelationDistance()), ce, X;
                    dims = dims, kwargs...)
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = SimpleDistance()), ce, X;
                    dims = dims, kwargs...)
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = SimpleDistance()), rho;
                    kwargs...)
end

export GeneralDistance
