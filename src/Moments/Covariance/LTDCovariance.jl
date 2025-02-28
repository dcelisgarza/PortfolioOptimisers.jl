
mutable struct LTDCovariance{T1 <: PortfolioOptimisersVarianceEstimator, T2 <: Real} <:
               PortfolioOptimisersCovarianceEstimator
    ve::T1
    alpha::T2
end
function LTDCovariance(; ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                       alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return LTDCovariance{typeof(ve), typeof(alpha)}(ve, alpha)
end

function lower_tail_dependence(X::AbstractMatrix, alpha::Real = 0.05)
    T, N = size(X)
    k = ceil(Int, T * alpha)
    rho = Matrix{eltype(X)}(undef, N, N)

    if k > 0
        for j ∈ axes(X, 2)
            xj = view(X, :, j)
            v = sort(xj)[k]
            maskj = xj .<= v
            for i ∈ 1:j
                xi = view(X, :, i)
                u = sort(xi)[k]
                ltd = sum(xi .<= u .&& maskj) / k
                rho[j, i] = rho[i, j] = clamp(ltd, zero(eltype(X)), one(eltype(X)))
            end
        end
    end

    return rho
end
function StatsBase.cor(ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return lower_tail_dependence(X, ce.alpha)
end
function StatsBase.cov(ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return lower_tail_dependence(X, ce.alpha) .* (std_vec ⊗ std_vec)
end

export LTDCovariance
