struct AugmentedBlackLittermanPriorModel{T1 <: AbstractMatrix, T2 <: AbstractVector,
                                         T3 <: AbstractMatrix, T4 <: AbstractVector,
                                         T5 <: AbstractMatrix, T6 <: LoadingsMatrix,
                                         T7 <: AbstractMatrix, T8 <: AbstractVector,
                                         T9 <: AbstractMatrix, T10 <: AbstractVector} <:
       AbstractBlackLittermanPriorModel
    X::T1
    mu::T2
    sigma::T3
    f_mu::T4
    f_sigma::T5
    loadings::T6
    P::T7
    Q::T8
    f_P::T9
    f_Q::T10
end
function AugmentedBlackLittermanPriorModel(; X::AbstractMatrix, mu::AbstractVector,
                                           sigma::AbstractMatrix, f_mu::AbstractVector,
                                           f_sigma::AbstractMatrix,
                                           loadings::LoadingsMatrix, P::AbstractMatrix,
                                           Q::AbstractVector, f_P::AbstractMatrix,
                                           f_Q::AbstractVector)
    @smart_assert(size(X, 2) ==
                  length(mu) ==
                  size(sigma, 1) ==
                  size(sigma, 2) ==
                  size(loadings.M, 1) ==
                  length(loadings.b) ==
                  size(P, 2))
    @smart_assert(length(f_mu) ==
                  size(f_sigma, 1) ==
                  size(f_sigma, 2) ==
                  size(loadings.M, 2) ==
                  size(f_P, 2))
    @smart_assert(length(f_Q) == size(f_P, 1))
    @smart_assert(length(Q) == size(P, 1))
    return AugmentedBlackLittermanPriorModel{typeof(X), typeof(mu), typeof(sigma),
                                             typeof(f_mu), typeof(f_sigma),
                                             typeof(loadings), typeof(P), typeof(Q),
                                             typeof(f_P), typeof(f_Q)}(X, mu, sigma, f_mu,
                                                                       f_sigma, loadings, P,
                                                                       Q, f_P, f_Q)
end
struct AugmentedBlackLittermanPriorEstimator{T1 <: AbstractPriorEstimator,
                                             T2 <: AbstractPriorEstimator,
                                             T3 <: MatrixProcessing, T4 <: RegressionMethod,
                                             T5 <: PortfolioOptimisersVarianceEstimator,
                                             T6 <: Union{<:LinearConstraintAtom,
                                                         <:AbstractVector{<:LinearConstraintAtom}},
                                             T7 <: Union{<:LinearConstraintAtom,
                                                         <:AbstractVector{<:LinearConstraintAtom}},
                                             T8 <: DataFrame, T9 <: DataFrame, T10 <: Real,
                                             T11 <: Union{Nothing, <:AbstractVector},
                                             T12 <: Union{Nothing, <:AbstractVector},
                                             T13 <: Union{Nothing, <:AbstractVector},
                                             T14 <: Union{Nothing, <:Real},
                                             T15 <: Union{Nothing, <:Real}} <:
       AbstractBlackLittermanPriorEstimator
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
    a_views_conf::T11
    f_views_conf::T12
    w::T13
    l::T14
    tau::T15
end
function AugmentedBlackLittermanPriorEstimator(;
                                               a_pe::AbstractPriorEstimator                                                     = EmpiricalPriorEstimator(),
                                               f_pe::AbstractPriorEstimator                                                     = EmpiricalPriorEstimator(),
                                               mp::MatrixProcessing                                                             = DefaultMatrixProcessing(),
                                               re::RegressionMethod                                                             = ForwardRegression(),
                                               ve::PortfolioOptimisersVarianceEstimator                                         = SimpleVariance(),
                                               a_views::Union{<:LinearConstraintAtom, <:AbstractVector{<:LinearConstraintAtom}} = LinearConstraintAtom(),
                                               f_views::Union{<:LinearConstraintAtom, <:AbstractVector{<:LinearConstraintAtom}} = LinearConstraintAtom(),
                                               a_sets::DataFrame                                                                = DataFrame(),
                                               f_sets::DataFrame                                                                = DataFrame(),
                                               rf::Real                                                                         = 0.0,
                                               a_views_conf::Union{Nothing, <:AbstractVector}                                   = nothing,
                                               f_views_conf::Union{Nothing, <:AbstractVector}                                   = nothing,
                                               w::Union{Nothing, <:AbstractVector}                                              = nothing,
                                               l::Union{Nothing, <:Real}                                                        = nothing,
                                               tau::Union{Nothing, <:Real}                                                      = nothing)
    if !isnothing(a_views_conf)
        @smart_assert(length(a_views) == length(a_views_conf))
        @smart_assert(all(zero(eltype(a_views_conf)) .<
                          a_views_conf .<
                          one(eltype(a_views_conf))))
    end
    if !isnothing(f_views_conf)
        @smart_assert(length(f_views) == length(f_views_conf))
        @smart_assert(all(zero(eltype(f_views_conf)) .<
                          f_views_conf .<
                          one(eltype(f_views_conf))))
    end
    if !isnothing(tau)
        @smart_assert(tau > zero(tau))
    end
    return AugmentedBlackLittermanPriorEstimator{typeof(a_pe), typeof(f_pe), typeof(mp),
                                                 typeof(re), typeof(ve), typeof(a_views),
                                                 typeof(f_views), typeof(a_sets),
                                                 typeof(f_sets), typeof(rf),
                                                 typeof(a_views_conf), typeof(f_views_conf),
                                                 typeof(w), typeof(l), typeof(tau)}(a_pe,
                                                                                    f_pe,
                                                                                    mp, re,
                                                                                    ve,
                                                                                    a_views,
                                                                                    f_views,
                                                                                    a_sets,
                                                                                    f_sets,
                                                                                    rf,
                                                                                    a_views_conf,
                                                                                    f_views_conf,
                                                                                    w, l,
                                                                                    tau)
end
function Base.getproperty(obj::AugmentedBlackLittermanPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.a_pe.me
    elseif sym == :ce
        obj.a_pe.ce
    else
        getfield(obj, sym)
    end
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
    aug_posterior_mu = aug_prior_mu + v1 * (v2 \ v3)
    aug_posterior_sigma = aug_prior_sigma + tau * aug_prior_sigma -
                          v1 * (v2 \ transpose(v1))
    mtx_process!(pe.mp, aug_posterior_sigma, hcat(posterior_X, F))
    posterior_mu = aug_posterior_mu[1:size(X, 2)] .+ pe.rf .+ b
    posterior_sigma = aug_posterior_sigma[1:size(X, 2), 1:size(X, 2)]
    return AugmentedBlackLittermanPriorModel(; X = posterior_X, mu = posterior_mu,
                                             sigma = posterior_sigma, f_mu = f_prior_mu,
                                             f_sigma = f_prior_sigma, loadings = loadings,
                                             P = P, Q = Q, f_P = f_P, f_Q = f_Q)
end

export AugmentedBlackLittermanPriorModel, AugmentedBlackLittermanPriorEstimator
