struct EmpiricalPriorModel{T1 <: AbstractMatrix, T2 <: AbstractVector,
                           T3 <: AbstractMatrix} <: AbstractPriorModel_A
    X::T1
    mu::T2
    sigma::T3
end
function EmpiricalPriorModel(; X::AbstractMatrix, mu::AbstractVector, sigma::AbstractMatrix)
    @smart_assert(!isempty(X) && !isempty(mu) && !isempty(sigma))
    @smart_assert(size(X, 2) == length(mu))
    issquare(sigma)
    @smart_assert(length(mu) == size(sigma, 1))
    return EmpiricalPriorModel{typeof(X), typeof(mu), typeof(sigma)}(X, mu, sigma)
end
struct EmpiricalPriorEstimator{T1 <: StatsBase.CovarianceEstimator,
                               T2 <: ExpectedReturnsEstimator,
                               T3 <: Union{Nothing, <:Real}} <: AbstractPriorEstimator_1_0
    ce::T1
    me::T2
    horizon::T3
end
function EmpiricalPriorEstimator(;
                                 ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                 me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                 horizon::Union{Nothing, <:Real} = nothing)
    return EmpiricalPriorEstimator{typeof(ce), typeof(me), typeof(horizon)}(ce, me, horizon)
end
function prior(pe::EmpiricalPriorEstimator{<:Any, <:Any, Nothing}, X::AbstractMatrix,
               args...; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = vec(mean(pe.me, X))
    sigma = cov(pe.ce, X)
    return EmpiricalPriorModel(; X = X, mu = mu, sigma = sigma)
end
function prior(pe::EmpiricalPriorEstimator{<:Any, <:Any, <:Real}, X::AbstractMatrix,
               args...; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    X_log = log1p.(X)
    mu = vec(mean(pe.me, X_log))
    sigma = cov(pe.ce, X_log)
    mu .*= pe.horizon
    sigma .*= pe.horizon
    mu .= exp.(mu + 0.5 * diag(sigma))
    sigma .= (mu ⊗ mu) .* (exp.(sigma) .- one(eltype(sigma)))
    return EmpiricalPriorModel(; X = X, mu = mu, sigma = sigma)
end

export EmpiricalPriorEstimator, EmpiricalPriorModel
