struct BayesianBlackLittermanPriorEstimator{T1 <: FactorModelPriorEstimator,
                                            T2 <: MatrixProcessing,
                                            T3 <: Union{<:LinearConstraintAtom,
                                                        <:AbstractVector{<:LinearConstraintAtom}},
                                            T4 <: DataFrame, T5 <: Real,
                                            T6 <: Union{Nothing, <:AbstractVector},
                                            T7 <: Union{Nothing, <:Real}} <: PriorEstimator
    pe::T1
    mp::T2
    factor_views::T3
    factor_sets::T4
    rf::T5
    factor_views_conf::T6
    tau::T7
end
function BayesianBlackLittermanPriorEstimator(;
                                              pe::FactorModelPriorEstimator                                                         = FactorModelPriorEstimator(; pe = EmpiricalPriorEstimator(; me = EquilibriumExpectedReturns())),
                                              mp::MatrixProcessing                                                                  = MatrixProcessing(),
                                              factor_views::Union{<:LinearConstraintAtom, <:AbstractVector{<:LinearConstraintAtom}} = LinearConstraintAtom(),
                                              factor_sets::DataFrame                                                                = DataFrame(),
                                              rf::Real                                                                              = 0.0,
                                              factor_views_conf::Union{Nothing, <:AbstractVector}                                   = nothing,
                                              tau::Union{Nothing, <:Real}                                                           = nothing)
    if !isnothing(factor_views_conf)
        @smart_assert(length(factor_views) == length(factor_views_conf))
        @smart_assert(all(zero(eltype(factor_views_conf)) .<
                          factor_views_conf .<
                          one(eltype(factor_views_conf))))
    end
    if !isnothing(tau)
        @smart_assert(tau > zero(tau))
    end
    return BayesianBlackLittermanPriorEstimator{typeof(pe), typeof(mp),
                                                typeof(factor_views), typeof(factor_sets),
                                                typeof(rf), typeof(factor_views_conf),
                                                typeof(tau)}(pe, mp, factor_views,
                                                             factor_sets, rf,
                                                             factor_views_conf, tau)
end
function prior(pe::BayesianBlackLittermanPriorEstimator, X::AbstractMatrix,
               F::AbstractMatrix; dims::Int = 1, strict::Bool = false)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    T, N = size(F)
    @smart_assert(nrow(pe.factor_sets) == N)
    prior_model = prior(pe.pe, X, F)
    prior_X, prior_sigma, f_mu, f_sigma, loadings = prior_model.X, prior_model.sigma,
                                                    prior_model.f_mu, prior_model.f_sigma,
                                                    prior_model.loadings
    f_P, f_Q = views_constraints(pe.factor_views, pe.factor_sets;
                                 datatype = eltype(prior_X), strict = strict)
    @smart_assert(!isempty(f_P))
    factor_views_conf = pe.factor_views_conf
    tau = isnothing(pe.tau) ? inv(T) : pe.tau
    f_omega = tau * Diagonal(if isnothing(factor_views_conf)
                                 f_P * f_sigma * transpose(f_P)
                             else
                                 idx = iszero.(factor_views_conf)
                                 factor_views_conf[idx] .= eps(eltype(factor_views_conf))
                                 alphas = inv.(factor_views_conf) .- 1
                                 alphas .* f_P * f_sigma * transpose(f_P)
                             end)
    (; b, M) = loadings
    sigma_hat = inv(f_sigma) + transpose(f_P) * (f_omega \ f_P)
    mu_hat = sigma_hat \ (f_sigma \ f_mu + transpose(f_P) * (f_omega \ f_Q))
    v1 = prior_sigma \ M
    v2 = sigma_hat + transpose(M) * v1
    v3 = inv(prior_sigma)
    posterior_sigma = inv(v3 - v1 * (v2 \ transpose(M)) * v3)
    mtx_process!(pe.mp, posterior_sigma, prior_X)
    posterior_mu = (posterior_sigma * v1 * (v2 \ sigma_hat) * mu_hat) .+ pe.rf .+ b
    return FactorPriorModel(; X = prior_X, mu = posterior_mu, sigma = posterior_sigma,
                            f_mu = f_mu, f_sigma = f_sigma, loadings = loadings,
                            chol = Matrix{eltype(posterior_sigma)}(undef, 0, 0))
end

export BayesianBlackLittermanPriorEstimator
