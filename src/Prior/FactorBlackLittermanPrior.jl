struct FactorBlackLittermanPriorEstimator{T1 <: PriorEstimator, T2 <: MatrixProcessing,
                                          T3 <: MatrixProcessing, T4 <: RegressionMethod,
                                          T5 <: PortfolioOptimisersVarianceEstimator,
                                          T6 <: Union{<:LinearConstraintAtom,
                                                      <:AbstractVector{<:LinearConstraintAtom}},
                                          T7 <: DataFrame, T8 <: Real, T9 <: Bool,
                                          T10 <: Union{Nothing, <:AbstractVector},
                                          T11 <: Union{Nothing, <:AbstractVector},
                                          T12 <: Union{Nothing, <:Real},
                                          T13 <: Union{Nothing, <:Real}}
    pe::T1
    posterior_factor_mp::T2
    posterior_mp::T3
    re::T4
    ve::T5
    factor_views::T6
    factor_sets::T7
    rf::T8
    residuals::T9
    factor_views_conf::T10
    w::T11
    l::T12
    tau::T13
end
function FactorBlackLittermanPriorEstimator(;
                                            pe::PriorEstimator                                                                    = EmpiricalPriorEstimator(;),
                                            posterior_factor_mp::MatrixProcessing                                                 = DefaultMatrixProcessing(),
                                            posterior_mp::MatrixProcessing                                                        = DefaultMatrixProcessing(),
                                            re::RegressionMethod                                                                  = ForwardRegression(),
                                            ve::PortfolioOptimisersVarianceEstimator                                              = SimpleVariance(),
                                            factor_views::Union{<:LinearConstraintAtom, <:AbstractVector{<:LinearConstraintAtom}} = LinearConstraintAtom(),
                                            factor_sets::DataFrame                                                                = DataFrame(),
                                            rf::Real                                                                              = 0.0,
                                            residuals::Bool                                                                       = true,
                                            factor_views_conf::Union{Nothing, <:AbstractVector}                                   = nothing,
                                            w::Union{Nothing, <:AbstractVector}                                                   = nothing,
                                            l::Union{Nothing, <:Real}                                                             = nothing,
                                            tau::Union{Nothing, <:Real}                                                           = nothing)
    return FactorBlackLittermanPriorEstimator{typeof(pe), typeof(posterior_factor_mp),
                                              typeof(posterior_mp), typeof(re), typeof(ve),
                                              typeof(factor_views), typeof(factor_sets),
                                              typeof(rf), typeof(residuals),
                                              typeof(factor_views_conf), typeof(w),
                                              typeof(l), typeof(tau)}(pe,
                                                                      posterior_factor_mp,
                                                                      posterior_mp, re, ve,
                                                                      factor_views,
                                                                      factor_sets, rf,
                                                                      residuals,
                                                                      factor_views_conf, w,
                                                                      l, tau)
end
function prior(pe::FactorBlackLittermanPriorEstimator, X::AbstractMatrix, F::AbstractMatrix;
               dims::Int = 1, strict::Bool = false)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    T = size(X, 1)
    # Factor statistics.
    factor_prior = prior(pe.pe, F)
    f_prior_mu, f_prior_sigma = factor_prior.mu, factor_prior.sigma
    # Black litterman on the factors.
    loadings = regression(pe.re, X, F)
    (; b, M) = loadings
    posterior_X = F * transpose(M) .+ transpose(b)
    f_P, f_Q = views_constraints(pe.factor_views, pe.factor_sets;
                                 datatype = eltype(posterior_X), strict = strict)
    @smart_assert(!isempty(f_P))
    factor_views_conf = pe.factor_views_conf
    tau = isnothing(pe.tau) ? inv(T) : pe.tau
    f_omega = tau * Diagonal(if isnothing(factor_views_conf)
                                 f_P * f_prior_sigma * transpose(f_P)
                             else
                                 idx = iszero.(factor_views_conf)
                                 factor_views_conf[idx] .= eps(eltype(factor_views_conf))
                                 alphas = inv.(factor_views_conf) .- 1
                                 alphas .* f_P * f_prior_sigma * transpose(f_P)
                             end)
    f_prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
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
    mtx_process!(pe.posterior_factor_mp, f_posterior_sigma, F)
    # Reconstruct the posteriors using the black litterman adjusted factor statistics.
    posterior_mu = M * f_posterior_mu .+ b
    posterior_sigma = M * f_posterior_sigma * transpose(M)
    posterior_csigma = M * cholesky(f_posterior_sigma).L
    if pe.residuals
        err = X - posterior_X
        err_sigma = diagm(vec(var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    mtx_process!(pe.posterior_mp, posterior_sigma, posterior_X)
    return FactorPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                       chol = transpose(reshape(posterior_csigma, length(posterior_mu), :)),
                       f_mu = f_posterior_mu, f_sigma = f_posterior_sigma,
                       loadings = loadings)
end

export FactorBlackLittermanPriorEstimator
