struct LowOrderPriorResult{T1 <: AbstractMatrix, T2 <: AbstractVector, T3 <: AbstractMatrix,
                           T4 <: Union{Nothing, <:AbstractMatrix},
                           T5 <: Union{Nothing, <:AbstractVector},
                           T6 <: Union{Nothing, <:RegressionResult},
                           T7 <: Union{Nothing, <:AbstractVector},
                           T8 <: Union{Nothing, <:AbstractMatrix},
                           T9 <: Union{Nothing, <:AbstractVector}} <: AbstractPriorResult
    X::T1
    mu::T2
    sigma::T3
    chol::T4
    w::T5
    loadings::T6
    f_mu::T7
    f_sigma::T8
    f_w::T9
end
function LowOrderPriorResult(; X::AbstractMatrix, mu::AbstractVector, sigma::AbstractMatrix,
                             chol::Union{Nothing, <:AbstractMatrix} = nothing,
                             w::Union{Nothing, <:AbstractVector} = nothing,
                             loadings::Union{Nothing, <:RegressionResult} = nothing,
                             f_mu::Union{Nothing, <:AbstractVector} = nothing,
                             f_sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                             f_w::Union{Nothing, <:AbstractVector} = nothing)
    @smart_assert(!isempty(X) && !isempty(mu) && !isempty(sigma))
    @smart_assert(size(X, 2) == length(mu))
    issquare(sigma)
    if !isnothing(w)
        @smart_assert(!isempty(w))
        @smart_assert(length(w) == size(X, 1))
    end
    loadings_flag = !isnothing(loadings)
    f_mu_flag = !isnothing(f_mu)
    f_sigma_flag = !isnothing(f_sigma)
    if loadings_flag || f_mu_flag || f_sigma_flag
        @smart_assert(loadings_flag && f_mu_flag && f_sigma_flag)
        @smart_assert(!isempty(f_mu) && !isempty(f_sigma))
        issquare(f_sigma)
        @smart_assert(size(loadings.M, 2) == length(f_mu) == size(f_sigma, 1))
        @smart_assert(size(loadings.M, 1) == length(mu))
        if !isnothing(chol)
            @smart_assert(!isempty(chol))
            @smart_assert(length(mu) == size(chol, 2))
        end
        if !isnothing(f_w)
            @smart_assert(!isempty(f_w))
            @smart_assert(length(f_w) == size(X, 1))
        end
    end
    return LowOrderPriorResult{typeof(X), typeof(mu), typeof(sigma), typeof(chol),
                               typeof(w), typeof(loadings), typeof(f_mu), typeof(f_sigma),
                               typeof(f_w)}(X, mu, sigma, chol, w, loadings, f_mu, f_sigma,
                                            f_w)
end
function prior_view(pr::LowOrderPriorResult, i::AbstractVector)
    chol = isnothing(pr.chol) ? nothing : view(pr.chol, :, i)
    return LowOrderPriorResult(; X = view(pr.X, :, i), mu = view(pr.mu, i),
                               sigma = view(pr.sigma, i, i), chol = chol, w = pr.w,
                               loadings = regression_view(pr.loadings, i), f_mu = pr.f_mu,
                               f_sigma = pr.f_sigma, f_w = pr.f_w)
end
struct EmpiricalPriorEstimator{T1 <: StatsBase.CovarianceEstimator,
                               T2 <: AbstractExpectedReturnsEstimator,
                               T3 <: Union{Nothing, <:Real}} <:
       AbstractLowOrderPriorEstimator_1_0
    ce::T1
    me::T2
    horizon::T3
end
function EmpiricalPriorEstimator(;
                                 ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                 me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                 horizon::Union{Nothing, <:Real} = nothing)
    return EmpiricalPriorEstimator{typeof(ce), typeof(me), typeof(horizon)}(ce, me, horizon)
end
function factory(pe::EmpiricalPriorEstimator,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return EmpiricalPriorEstimator(; me = factory(pe.me, w), ce = factory(pe.ce, w),
                                   horizon = pe.horizon)
end
function prior(pe::EmpiricalPriorEstimator{<:Any, <:Any, Nothing}, X::AbstractMatrix,
               args...; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = vec(mean(pe.me, X; kwargs...))
    sigma = cov(pe.ce, X; kwargs...)
    return LowOrderPriorResult(; X = X, mu = mu, sigma = sigma)
end
function prior(pe::EmpiricalPriorEstimator{<:Any, <:Any, <:Real}, X::AbstractMatrix,
               args...; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
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
    return LowOrderPriorResult(; X = X, mu = mu, sigma = sigma)
end

export EmpiricalPriorEstimator, LowOrderPriorResult
