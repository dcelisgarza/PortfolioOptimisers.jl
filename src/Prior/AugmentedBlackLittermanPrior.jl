struct AugmentedBlackLittermanPriorEstimator{T1 <: PriorEstimator, T2 <: PriorEstimator,
                                             T3 <: MatrixProcessing, T4 <: RegressionMethod,
                                             T5 <: PortfolioOptimisersVarianceEstimator,
                                             T6 <: Union{<:LinearConstraintAtom,
                                                         <:AbstractVector{<:LinearConstraintAtom}},
                                             T7 <: Union{<:LinearConstraintAtom,
                                                         <:AbstractVector{<:LinearConstraintAtom}},
                                             T8 <: DataFrame, T9 <: DataFrame, T10 <: Real,
                                             T11 <: Bool,
                                             T12 <: Union{Nothing, <:AbstractVector},
                                             T13 <: Union{Nothing, <:AbstractVector},
                                             T14 <: Union{Nothing, <:AbstractVector},
                                             T15 <: Union{Nothing, <:Real},
                                             T16 <: Union{Nothing, <:Real}}
    a_pe::T1
    f_pe::T2
    mp::T3
    re::T4
    ve::T5
    a_views::T6
    f_views::T7
    a_sets::T8
    f_sets::T9
    rf::T10
    residuals::T11
    a_views_conf::T12
    f_views_conf::T13
    w::T14
    l::T15
    tau::T16
end
function prior(pe::AugmentedBlackLittermanPriorEstimator, X::AbstractMatrix,
               F::AbstractMatrix; dims::Int = 1, strict::Bool = false)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @smart_assert(nrow(pe.a_sets) == size(X, 2))
    @smart_assert(nrow(pe.f_sets) == size(F, 2))
    # Asset prior.
    a_prior = prior(pe.a_pe, X)
    a_prior_mu, a_prior_sigma = a_prior.mu, a_prior.sigma
    # Factor prior.
    f_prior = prior(pe.f_pe, F)
    f_prior_mu, f_prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    loadings = regression(pe.re, X, F)
    (; b, M) = loadings
    posterior_X = F * transpose(M) .+ transpose(b)
    P, Q = views_constraints(pe.a_views, pe.a_sets; datatype = eltype(posterior_X),
                             strict = strict)
    @smart_assert(!isempty(P))
    f_P, f_Q = views_constraints(pe.f_views, pe.f_sets; datatype = eltype(posterior_X),
                                 strict = strict)
    @smart_assert(!isempty(f_P))
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    a_views_conf = pe.a_views_conf
    a_omega = tau * Diagonal(if isnothing(a_views_conf)
                                 P * a_prior_sigma * transpose(P)
                             else
                                 idx = iszero.(a_views_conf)
                                 a_views_conf[idx] .= eps(eltype(a_views_conf))
                                 alphas = inv.(a_views_conf) .- 1
                                 alphas .* P * a_prior_sigma * transpose(P)
                             end)
    f_views_conf = pe.f_views_conf
    f_omega = tau * Diagonal(if isnothing(f_views_conf)
                                 f_P * f_prior_sigma * transpose(f_P)
                             else
                                 idx = iszero.(f_views_conf)
                                 f_views_conf[idx] .= eps(eltype(f_views_conf))
                                 alphas = inv.(f_views_conf) .- 1
                                 alphas .* f_P * f_prior_sigma * transpose(f_P)
                             end)
    aug_prior_sigma = hcat(vcat(a_prior_sigma, f_prior_sigma * transpose(M)),
                           vcat(M * f_prior_sigma, f_prior_sigma))
    aug_P = hcat(vcat(P, zeros(size(f_P, 1), size(P, 2))),
                 vcat(zeros(size(P, 1), size(f_P, 2)), f_P))
    aug_Q = vcat(Q, f_Q)
    aug_omega = hcat(vcat(a_omega, zeros(size(f_omega, 1), size(a_omega, 1))),
                     vcat(zeros(size(a_omega, 1), size(f_omega, 1)), f_omega))
    aug_prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @smart_assert(length(pe.w) == size(X, 2))
            pe.w
        else
            fill(inv(size(X, 2)), size(X, 2))
        end
        pe.l * (vcat(a_prior_sigma, f_prior_sigma * transpose(M))) * w
    else
        vcat(a_prior_mu, f_prior_mu) .- pe.rf
    end
    v1 = tau * aug_prior_sigma * transpose(aug_P)
    v2 = aug_P * v1 + aug_omega
    v3 = aug_Q .- aug_P * aug_prior_mu
    aug_posterior_mu = aug_prior_mu + v1 * (v2 \ v3) .+ pe.rf
    aug_posterior_sigma = aug_prior_sigma + tau * aug_prior_sigma -
                          v1 * (v2 \ transpose(v1))
    posterior_mu = aug_posterior_mu[1:size(X, 1)] .+ b
    posterior_sigma = aug_posterior_mu[1:size(X, 1), 1:size(X, 1)]
    mtx_process!(pe.mp, posterior_sigma, posterior_X)
    # if pe.residuals
    #     err = X - posterior_X
    #     err_sigma = diagm(vec(var(pe.ve, err; dims = 1)))
    #     posterior_sigma .+= err_sigma
    # end

    return nothing
end

export AugmentedBlackLittermanPriorEstimator