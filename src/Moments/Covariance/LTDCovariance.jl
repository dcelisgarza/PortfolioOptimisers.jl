
mutable struct LTDCovariance{T1 <: Real, T2 <: PortfolioOptimisersVarianceEstimator,
                             T3 <: Union{Nothing, <:AbstractWeights}} <:
               PortfolioOptimisersCovarianceEstimator
    alpha::T1
    ve::T2
    w::T3
end
function LTDCovariance(; alpha::Real = 0.05,
                       ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                       w::Union{Nothing, <:AbstractWeights} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return LTDCovariance{typeof(alpha), typeof(ve), typeof(w)}(alpha, ve, w)
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
function StatsBase.cor(ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return lower_tail_dependence(X, ce.alpha)
end
function StatsBase.cov(ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = vec(if isnothing(ce.w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.w; dims = 1)
                  end)
    return lower_tail_dependence(X, ce.alpha) .* (std_vec ⊗ std_vec)
end

export LTDCovariance
