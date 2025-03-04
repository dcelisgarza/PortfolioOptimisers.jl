struct EmpiricalPrior{T1 <: AbstractMatrix, T2 <: AbstractVector, T3 <: AbstractMatrix} <:
       AbstractPriorModel
    X::T1
    mu::T2
    sigma::T3
end
function EmpiricalPrior(; X::AbstractMatrix, mu::AbstractVector, sigma::AbstractMatrix)
    @smart_assert(size(X, 2) == length(mu) == size(sigma, 1) == size(sigma, 2))
    return EmpiricalPrior{typeof(X), typeof(mu), typeof(sigma)}(X, mu, sigma)
end
struct EmpirircalPriorEstimator{T1 <: StatsBase.CovarianceEstimator,
                                T2 <: ExpectedReturnsEstimator,
                                T3 <: Union{Nothing, <:Real}} <: PriorEstimator
    ce::T1
    me::T2
    horizon::T3
end
function EmpirircalPriorEstimator(;
                                  ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                  me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                  horizon::Union{Nothing, <:Real} = nothing)
    return EmpirircalPriorEstimator{typeof(ce), typeof(me), typeof(horizon)}(ce, me,
                                                                             horizon)
end
function prior(pe::EmpirircalPriorEstimator{<:Any, <:Any, Nothing}, X::AbstractMatrix;
               dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = vec(mean(pe.me, X))
    sigma = cov(pe.ce, X)
    return EmpiricalPrior(; X = X, mu = mu, sigma = sigma)
end
function prior(pe::EmpirircalPriorEstimator{<:Any, <:Any, <:Real}, X::AbstractMatrix;
               dims::Int = 1)
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
    return EmpiricalPrior(; X = X, mu = mu, sigma = sigma)
end

export EmpirircalPriorEstimator, EmpiricalPrior
