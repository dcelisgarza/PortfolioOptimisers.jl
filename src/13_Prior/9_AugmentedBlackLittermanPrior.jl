struct AugmentedBlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
                                    T14, T15} <: AbstractLowOrderPriorEstimator_2_2
    a_pe::T1
    f_pe::T2
    mp::T3
    re::T4
    ve::T5
    a_views::T6
    f_views::T7
    a_sets::T8
    f_sets::T9
    a_views_conf::T10
    f_views_conf::T11
    w::T12
    rf::T13
    l::T14
    tau::T15
    function AugmentedBlackLittermanPrior(a_pe::AbstractLowOrderPriorEstimatorMap_2_1,
                                          f_pe::AbstractLowOrderPriorEstimatorMap_2_1,
                                          mp::AbstractMatrixProcessingEstimator,
                                          re::AbstractRegressionEstimator,
                                          ve::AbstractVarianceEstimator,
                                          a_views::Union{<:LinearConstraintEstimator,
                                                         <:BlackLittermanViews},
                                          f_views::Union{<:LinearConstraintEstimator,
                                                         <:BlackLittermanViews},
                                          a_sets::Union{Nothing, <:AssetSets},
                                          f_sets::Union{Nothing, <:AssetSets},
                                          a_views_conf::Union{Nothing, <:Real,
                                                              <:AbstractVector},
                                          f_views_conf::Union{Nothing, <:Real,
                                                              <:AbstractVector},
                                          w::Union{Nothing, <:AbstractVector}, rf::Real,
                                          l::Union{Nothing, <:Real},
                                          tau::Union{Nothing, <:Real})
        if isa(w, AbstractVector)
            @argcheck(!isempty(w))
        end
        if isa(a_views, LinearConstraintEstimator)
            @argcheck(!isnothing(a_sets))
        end
        if isa(f_views, LinearConstraintEstimator)
            @argcheck(!isnothing(f_sets))
        end
        assert_bl_views_conf(a_views_conf, a_views)
        assert_bl_views_conf(f_views_conf, f_views)
        if !isnothing(tau)
            @argcheck(tau > zero(tau))
        end
        return new{typeof(a_pe), typeof(f_pe), typeof(mp), typeof(re), typeof(ve),
                   typeof(a_views), typeof(f_views), typeof(a_sets), typeof(f_sets),
                   typeof(a_views_conf), typeof(f_views_conf), typeof(w), typeof(rf),
                   typeof(l), typeof(tau)}(a_pe, f_pe, mp, re, ve, a_views, f_views, a_sets,
                                           f_sets, a_views_conf, f_views_conf, w, rf, l,
                                           tau)
    end
end
function AugmentedBlackLittermanPrior(;
                                      a_pe::AbstractLowOrderPriorEstimatorMap_2_1 = EmpiricalPrior(),
                                      f_pe::AbstractLowOrderPriorEstimatorMap_2_1 = EmpiricalPrior(),
                                      mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                      re::AbstractRegressionEstimator = StepwiseRegression(),
                                      ve::AbstractVarianceEstimator = SimpleVariance(),
                                      a_views::Union{<:LinearConstraintEstimator,
                                                     <:BlackLittermanViews},
                                      f_views::Union{<:LinearConstraintEstimator,
                                                     <:BlackLittermanViews},
                                      a_sets::Union{Nothing, <:AssetSets} = nothing,
                                      f_sets::Union{Nothing, <:AssetSets} = nothing,
                                      a_views_conf::Union{Nothing, <:Real,
                                                          <:AbstractVector} = nothing,
                                      f_views_conf::Union{Nothing, <:Real,
                                                          <:AbstractVector} = nothing,
                                      w::Union{Nothing, <:AbstractVector} = nothing,
                                      rf::Real = 0.0, l::Union{Nothing, <:Real} = nothing,
                                      tau::Union{Nothing, <:Real} = nothing)
    return AugmentedBlackLittermanPrior(a_pe, f_pe, mp, re, ve, a_views, f_views, a_sets,
                                        f_sets, a_views_conf, f_views_conf, w, rf, l, tau)
end
function factory(pe::AugmentedBlackLittermanPrior,
                 w::Union{Nothing, <:AbstractVector} = nothing)
    return AugmentedBlackLittermanPrior(; a_pe = factory(pe.a_pe, w),
                                        f_pe = factory(pe.f_pe, w), mp = pe.mp, re = pe.re,
                                        ve = factory(pe.ve, w), a_views = pe.a_views,
                                        f_views = pe.f_views, a_sets = pe.a_sets,
                                        f_sets = pe.f_sets, a_views_conf = pe.a_views_conf,
                                        f_views_conf = pe.f_views_conf, w = pe.w,
                                        rf = pe.rf, l = pe.l, tau = pe.tau)
end
function Base.getproperty(obj::AugmentedBlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.a_pe.me
    elseif sym == :ce
        obj.a_pe.ce
    elseif sym == :f_me
        obj.f_pe.me
    elseif sym == :f_ce
        obj.f_pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::AugmentedBlackLittermanPrior, X::AbstractMatrix, F::AbstractMatrix;
               dims::Int = 1, strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @argcheck(length(pe.a_sets.dict[pe.a_sets.key]) == size(X, 2))
    @argcheck(length(pe.f_sets.dict[pe.f_sets.key]) == size(F, 2))
    # Asset prior.
    a_prior = prior(pe.a_pe, X; strict = strict, kwargs...)
    a_prior_mu, a_prior_sigma = a_prior.mu, a_prior.sigma
    # Factor prior.
    f_prior = prior(pe.f_pe, F; strict = strict)
    f_prior_mu, f_prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    rr = regression(pe.re, X, F)
    (; b, M) = rr
    posterior_X = F * transpose(M) .+ transpose(b)
    (; P, Q) = black_litterman_views(pe.a_views, pe.a_sets; datatype = eltype(posterior_X),
                                     strict = strict)
    f_views = black_litterman_views(pe.f_views, pe.f_sets; datatype = eltype(posterior_X),
                                    strict = strict)
    f_P, f_Q = f_views.P, f_views.Q
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    a_omega = tau * calc_omega(pe.a_views_conf, P, a_prior_sigma)
    f_omega = tau * calc_omega(pe.f_views_conf, f_P, f_prior_sigma)
    aug_prior_sigma = hcat(vcat(a_prior_sigma, f_prior_sigma * transpose(M)),
                           vcat(M * f_prior_sigma, f_prior_sigma))
    aug_P = hcat(vcat(P, zeros(size(f_P, 1), size(P, 2))),
                 vcat(zeros(size(P, 1), size(f_P, 2)), f_P))
    aug_Q = vcat(Q, f_Q)
    aug_omega = hcat(vcat(a_omega, zeros(size(f_omega, 1), size(a_omega, 1))),
                     vcat(zeros(size(a_omega, 1), size(f_omega, 1)), f_omega))
    aug_prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @argcheck(length(pe.w) == size(X, 2))
            pe.w
        else
            iN = inv(size(X, 2))
            range(; start = iN, stop = iN, length = size(X, 2))
        end
        pe.l * (vcat(a_prior_sigma, f_prior_sigma * transpose(M))) * w
    else
        vcat(a_prior_mu, f_prior_mu) .- pe.rf
    end
    aug_posterior_mu, aug_posterior_sigma = vanilla_posteriors(tau, pe.rf, aug_prior_mu,
                                                               aug_prior_sigma, aug_omega,
                                                               aug_P, aug_Q)
    matrix_processing!(pe.mp, aug_posterior_sigma, hcat(posterior_X, F))
    posterior_mu = (aug_posterior_mu[1:size(X, 2)] + b) .+ pe.rf
    posterior_sigma = aug_posterior_sigma[1:size(X, 2), 1:size(X, 2)]
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         rr = rr, f_mu = f_prior_mu, f_sigma = f_prior_sigma,
                         f_w = f_prior.w)
end

export AugmentedBlackLittermanPrior
