struct BayesianBlackLittermanPriorEstimator{T1 <: FactorPriorEstimator,
                                            T2 <: MatrixProcessing,
                                            T3 <: Union{<:LinearConstraintAtom,
                                                        <:AbstractVector{<:LinearConstraintAtom}},
                                            T4 <: DataFrame, T5 <: Real,
                                            T6 <: Union{Nothing, <:AbstractVector},
                                            T7 <: Union{Nothing, <:Real}} <: PriorEstimator
    pe::T1
    mp::T2
    views::T3
    sets::T4
    rf::T5
    views_conf::T6
    tau::T7
end
function BayesianBlackLittermanPriorEstimator(;
                                              pe::FactorPriorEstimator = FactorPriorEstimator(;
                                                                                              pe = EmpiricalPriorEstimator(;
                                                                                                                           me = EquilibriumExpectedReturns())),
                                              mp::MatrixProcessing = MatrixProcessing(),
                                              views::Union{<:LinearConstraintAtom,
                                                           <:AbstractVector{<:LinearConstraintAtom}} = LinearConstraintAtom(),
                                              sets::DataFrame = DataFrame(), rf::Real = 0.0,
                                              views_conf::Union{Nothing, <:AbstractVector} = nothing,
                                              tau::Union{Nothing, <:Real} = nothing)
    if !isnothing(views_conf)
        @smart_assert(length(views) == length(views_conf))
        @smart_assert(all(zero(eltype(views_conf)) .< views_conf .< one(eltype(views_conf))))
    end
    if !isnothing(tau)
        @smart_assert(tau > zero(tau))
    end
    return BayesianBlackLittermanPriorEstimator{typeof(pe), typeof(mp), typeof(views),
                                                typeof(sets), typeof(rf),
                                                typeof(views_conf), typeof(tau)}(pe, mp,
                                                                                 views,
                                                                                 sets, rf,
                                                                                 views_conf,
                                                                                 tau)
end
function prior(pe::BayesianBlackLittermanPriorEstimator, X::AbstractMatrix,
               F::AbstractMatrix; dims::Int = 1, strict::Bool = false)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    T, N = size(F)
    @smart_assert(nrow(pe.sets) == N)
    prior_model = prior(pe.pe, X, F)
    prior_X, prior_sigma, f_mu, f_sigma, loadings = prior_model.X, prior_model.sigma,
                                                    prior_model.f_mu, prior_model.f_sigma,
                                                    prior_model.loadings
    f_P, f_Q = views_constraints(pe.views, pe.sets; datatype = eltype(prior_X),
                                 strict = strict)
    @smart_assert(!isempty(f_P))
    views_conf = pe.views_conf
    tau = isnothing(pe.tau) ? inv(T) : pe.tau
    f_omega = tau * Diagonal(if isnothing(views_conf)
                                 f_P * f_sigma * transpose(f_P)
                             else
                                 idx = iszero.(views_conf)
                                 views_conf[idx] .= eps(eltype(views_conf))
                                 alphas = inv.(views_conf) .- 1
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
    return FactorPrior(; X = prior_X, mu = posterior_mu, sigma = posterior_sigma,
                       f_mu = f_mu, f_sigma = f_sigma, loadings = loadings,
                       chol = Matrix{eltype(posterior_sigma)}(undef, 0, 0))
end

export BayesianBlackLittermanPriorEstimator
