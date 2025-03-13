struct FactorBlackLittermanPriorModel{T1 <: AbstractMatrix, T2 <: AbstractVector,
                                      T3 <: AbstractMatrix, T4 <: AbstractMatrix,
                                      T5 <: AbstractVector, T6 <: AbstractMatrix,
                                      T7 <: LoadingsMatrix, T8 <: AbstractMatrix,
                                      T9 <: AbstractVector} <: AbstractPriorModel_AFVC
    X::T1
    mu::T2
    sigma::T3
    chol::T4
    f_mu::T5
    f_sigma::T6
    loadings::T7
    f_P::T8
    f_Q::T9
end
function FactorBlackLittermanPriorModel(; X::AbstractMatrix, mu::AbstractVector,
                                        sigma::AbstractMatrix, chol::AbstractMatrix,
                                        f_mu::AbstractVector, f_sigma::AbstractMatrix,
                                        loadings::LoadingsMatrix, f_P::AbstractMatrix,
                                        f_Q::AbstractVector)
    @smart_assert(!isempty(X) &&
                  !isempty(mu) &&
                  !isempty(sigma) &&
                  !isempty(chol) &&
                  !isempty(f_mu) &&
                  !isempty(f_sigma) &&
                  !isempty(f_P) &&
                  !isempty(f_Q))
    @smart_assert(size(X, 2) ==
                  length(mu) ==
                  size(sigma, 1) ==
                  size(sigma, 2) ==
                  size(chol, 2) ==
                  size(loadings.M, 1) ==
                  length(loadings.b))
    @smart_assert(length(f_mu) ==
                  size(f_sigma, 1) ==
                  size(f_sigma, 2) ==
                  size(loadings.M, 2) ==
                  size(f_P, 2))
    @smart_assert(length(f_Q) == size(f_P, 1))
    return FactorBlackLittermanPriorModel{typeof(X), typeof(mu), typeof(sigma),
                                          typeof(chol), typeof(f_mu), typeof(f_sigma),
                                          typeof(loadings), typeof(f_P), typeof(f_Q)}(X, mu,
                                                                                      sigma,
                                                                                      chol,
                                                                                      f_mu,
                                                                                      f_sigma,
                                                                                      loadings,
                                                                                      f_P,
                                                                                      f_Q)
end
struct FactorBlackLittermanPriorEstimator{T1 <: AbstractPriorEstimatorMap_2_1,
                                          T2 <: MatrixProcessing, T3 <: MatrixProcessing,
                                          T4 <: RegressionMethod,
                                          T5 <: PortfolioOptimisersVarianceEstimator,
                                          T6 <: Union{<:LinearConstraintAtom,
                                                      <:AbstractVector{<:LinearConstraintAtom}},
                                          T7 <: DataFrame, T8 <: Real, T9 <: Bool,
                                          T10 <: Union{Nothing, <:AbstractVector},
                                          T11 <: Union{Nothing, <:AbstractVector},
                                          T12 <: Union{Nothing, <:Real},
                                          T13 <: Union{Nothing, <:Real}} <:
       AbstractPriorEstimator_2_1
    pe::T1
    f_mp::T2
    mp::T3
    re::T4
    ve::T5
    views::T6
    sets::T7
    rf::T8
    residuals::T9
    views_conf::T10
    w::T11
    l::T12
    tau::T13
end
function FactorBlackLittermanPriorEstimator(;
                                            pe::AbstractPriorEstimatorMap_2_1 = EmpiricalPriorEstimator(),
                                            f_mp::MatrixProcessing = DefaultMatrixProcessing(),
                                            mp::MatrixProcessing = DefaultMatrixProcessing(),
                                            re::RegressionMethod = ForwardRegression(),
                                            ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                                            views::Union{<:LinearConstraintAtom,
                                                         <:AbstractVector{<:LinearConstraintAtom}} = LinearConstraintAtom(),
                                            sets::DataFrame = DataFrame(), rf::Real = 0.0,
                                            residuals::Bool = true,
                                            views_conf::Union{Nothing, <:AbstractVector} = nothing,
                                            w::Union{Nothing, <:AbstractVector} = nothing,
                                            l::Union{Nothing, <:Real} = nothing,
                                            tau::Union{Nothing, <:Real} = nothing)
    if isa(views_conf, AbstractVector)
        @smart_assert(isa(views, AbstractVector))
        @smart_assert(!isempty(views))
        @smart_assert(!isempty(views_conf))
        @smart_assert(length(views) == length(views_conf))
        @smart_assert(all(zero(eltype(views_conf)) .< views_conf .< one(eltype(views_conf))))
    else
        if isa(views, AbstractVector)
            @smart_assert(!isempty(views))
        end
    end
    if !isnothing(tau)
        @smart_assert(tau > zero(tau))
    end
    return FactorBlackLittermanPriorEstimator{typeof(pe), typeof(f_mp), typeof(mp),
                                              typeof(re), typeof(ve), typeof(views),
                                              typeof(sets), typeof(rf), typeof(residuals),
                                              typeof(views_conf), typeof(w), typeof(l),
                                              typeof(tau)}(pe, f_mp, mp, re, ve, views,
                                                           sets, rf, residuals, views_conf,
                                                           w, l, tau)
end
function Base.getproperty(obj::FactorBlackLittermanPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::FactorBlackLittermanPriorEstimator, X::AbstractMatrix, F::AbstractMatrix;
               dims::Int = 1, strict::Bool = false, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    # Factor prior.
    f_prior = prior(pe.pe, F)
    f_prior_mu, f_prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    loadings = regression(pe.re, X, F)
    (; b, M) = loadings
    posterior_X = F * transpose(M) .+ transpose(b)
    f_P, f_Q = views_constraints(pe.views, pe.sets; datatype = eltype(posterior_X),
                                 strict = strict)
    @smart_assert(!isempty(f_P))
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    views_conf = pe.views_conf
    f_omega = tau * Diagonal(if isnothing(views_conf)
                                 f_P * f_prior_sigma * transpose(f_P)
                             else
                                 idx = iszero.(views_conf)
                                 views_conf[idx] .= eps(eltype(views_conf))
                                 alphas = inv.(views_conf) .- 1
                                 alphas .* f_P * f_prior_sigma * transpose(f_P)
                             end)
    f_prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @smart_assert(length(pe.w) == size(X, 2))
            pe.w
        else
            fill(inv(size(X, 2)), size(X, 2))
        end
        pe.l * (f_prior_sigma * transpose(M)) * w
    else
        f_prior_mu .- pe.rf
    end
    v1 = tau * f_prior_sigma * transpose(f_P)
    v2 = f_P * v1 + f_omega
    v3 = f_Q .- f_P * f_prior_mu
    f_posterior_mu = f_prior_mu + v1 * (v2 \ v3) .+ pe.rf
    f_posterior_sigma = f_prior_sigma + tau * f_prior_sigma - v1 * (v2 \ transpose(v1))
    mtx_process!(pe.f_mp, f_posterior_sigma, F)
    # Reconstruct the posteriors using the black litterman adjusted factor statistics.
    posterior_mu = M * f_posterior_mu .+ b
    posterior_sigma = M * f_posterior_sigma * transpose(M)
    mtx_process!(pe.mp, posterior_sigma, posterior_X)
    posterior_csigma = M * cholesky(f_posterior_sigma).L
    if pe.residuals
        err = X - posterior_X
        err_sigma = diagm(vec(var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    return FactorBlackLittermanPriorModel(; X = posterior_X, mu = posterior_mu,
                                          sigma = posterior_sigma,
                                          chol = transpose(reshape(posterior_csigma,
                                                                   length(posterior_mu), :)),
                                          f_mu = f_posterior_mu,
                                          f_sigma = f_posterior_sigma, loadings = loadings,
                                          f_P = f_P, f_Q = f_Q)
end

export FactorBlackLittermanPriorModel, FactorBlackLittermanPriorEstimator
