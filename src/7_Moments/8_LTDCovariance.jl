struct LTDCovariance{T1 <: AbstractVarianceEstimator, T2 <: Real,
                     T3 <: FLoops.Transducers.Executor} <: AbstractCovarianceEstimator
    ve::T1
    alpha::T2
    threads::T3
end
function LTDCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                       alpha::Real = 0.05,
                       threads::FLoops.Transducers.Executor = ThreadedEx())
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return LTDCovariance{typeof(ve), typeof(alpha), typeof(threads)}(ve, alpha, threads)
end
function factory(ce::LTDCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return LTDCovariance(; ve = factory(ce.ve, w), alpha = ce.alpha, threads = ce.threads)
end
function lower_tail_dependence(X::AbstractMatrix, alpha::Real = 0.05,
                               threads::FLoops.Transducers.Executor = SequentialEx())
    T, N = size(X)
    k = ceil(Int, T * alpha)
    rho = Matrix{eltype(X)}(undef, N, N)
    if k > 0
        @floop threads for j in axes(X, 2)
            xj = view(X, :, j)
            v = sort(xj)[k]
            maskj = xj .<= v
            for i in 1:j
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
    return lower_tail_dependence(X, ce.alpha, ce.threads)
end
function StatsBase.cov(ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return lower_tail_dependence(X, ce.alpha, ce.threads) ⊙ (std_vec ⊗ std_vec)
end

export LTDCovariance
