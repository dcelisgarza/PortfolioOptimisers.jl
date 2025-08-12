struct LowOrderPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12} <:
       AbstractPriorResult
    X::T1
    mu::T2
    sigma::T3
    chol::T4
    w::T5
    ens::T6
    kld::T7
    ow::T8
    loadings::T9
    f_mu::T10
    f_sigma::T11
    f_w::T12
end
function LowOrderPrior(; X::AbstractMatrix, mu::AbstractVector, sigma::AbstractMatrix,
                       chol::Union{Nothing, <:AbstractMatrix} = nothing,
                       w::Union{Nothing, <:AbstractWeights} = nothing,
                       ens::Union{Nothing, <:Real} = nothing,
                       kld::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                       ow::Union{Nothing, <:AbstractVector} = nothing,
                       loadings::Union{Nothing, <:Regression} = nothing,
                       f_mu::Union{Nothing, <:AbstractVector} = nothing,
                       f_sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                       f_w::Union{Nothing, <:AbstractVector} = nothing)
    @assert(!isempty(X) && !isempty(mu) && !isempty(sigma))
    @assert(size(X, 2) == length(mu))
    assert_matrix_issquare(sigma)
    if !isnothing(w)
        @assert(!isempty(w))
        @assert(length(w) == size(X, 1))
    end
    if isa(kld, AbstractVector)
        @assert(!isempty(kld))
    end
    if !isnothing(ow)
        @assert(!isempty(ow))
    end
    loadings_flag = !isnothing(loadings)
    f_mu_flag = !isnothing(f_mu)
    f_sigma_flag = !isnothing(f_sigma)
    if loadings_flag || f_mu_flag || f_sigma_flag
        @assert(loadings_flag && f_mu_flag && f_sigma_flag)
        @assert(!isempty(f_mu) && !isempty(f_sigma))
        assert_matrix_issquare(f_sigma)
        @assert(size(loadings.M, 2) == length(f_mu) == size(f_sigma, 1))
        @assert(size(loadings.M, 1) == length(mu))
        if !isnothing(chol)
            @assert(!isempty(chol))
            @assert(length(mu) == size(chol, 2))
        end
        if !isnothing(f_w)
            @assert(!isempty(f_w))
            @assert(length(f_w) == size(X, 1))
        end
    end
    return LowOrderPrior(X, mu, sigma, chol, w, ens, kld, ow, loadings, f_mu, f_sigma, f_w)
end
function prior_view(pr::LowOrderPrior, i::AbstractVector)
    chol = isnothing(pr.chol) ? nothing : view(pr.chol, :, i)
    return LowOrderPrior(; X = view(pr.X, :, i), mu = view(pr.mu, i),
                         sigma = view(pr.sigma, i, i), chol = chol, w = pr.w, ens = pr.ens,
                         kld = pr.kld, ow = pr.ow,
                         loadings = regression_view(pr.loadings, i), f_mu = pr.f_mu,
                         f_sigma = pr.f_sigma, f_w = pr.f_w)
end
struct EmpiricalPrior{T1, T2, T3} <: AbstractLowOrderPriorEstimator_1_0
    ce::T1
    me::T2
    horizon::T3
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
    @assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = vec(mean(pe.me, X; kwargs...))
    sigma = cov(pe.ce, X; kwargs...)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma)
end
function prior(pe::EmpiricalPrior{<:Any, <:Any, <:Real}, X::AbstractMatrix, args...;
               dims::Int = 1, kwargs...)
    @assert(dims in (1, 2))
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

export EmpiricalPrior, LowOrderPrior
