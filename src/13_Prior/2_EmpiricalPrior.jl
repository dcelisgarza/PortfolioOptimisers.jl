struct EmpiricalPrior{T1, T2, T3} <: AbstractLowOrderPriorEstimator_A
    ce::T1
    me::T2
    horizon::T3
    function EmpiricalPrior(ce::StatsBase.CovarianceEstimator,
                            me::AbstractExpectedReturnsEstimator,
                            horizon::Union{Nothing, <:Real})
        return new{typeof(ce), typeof(me), typeof(horizon)}(ce, me, horizon)
    end
end
function EmpiricalPrior(;
                        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        horizon::Union{Nothing, <:Real} = nothing)
    return EmpiricalPrior(ce, me, horizon)
end
function factory(pe::EmpiricalPrior, w::Union{Nothing, <:AbstractWeights} = nothing)
    return EmpiricalPrior(; me = factory(pe.me, w), ce = factory(pe.ce, w),
                          horizon = pe.horizon)
end
function prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing}, X::AbstractMatrix, args...;
               dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = vec(mean(pe.me, X; kwargs...))
    sigma = cov(pe.ce, X; kwargs...)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma)
end
function prior(pe::EmpiricalPrior{<:Any, <:Any, <:Real}, X::AbstractMatrix, args...;
               dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    X_log = log1p.(X)
    mu = vec(mean(pe.me, X_log; kwargs...))
    sigma = cov(pe.ce, X_log; kwargs...)
    mu .*= pe.horizon
    sigma .*= pe.horizon
    mu .= exp.(mu + 0.5 * diag(sigma))
    sigma .= (mu ⊗ mu) ⊙ (exp.(sigma) .- one(eltype(sigma)))
    mu .-= one(eltype(mu))
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma)
end

export EmpiricalPrior
