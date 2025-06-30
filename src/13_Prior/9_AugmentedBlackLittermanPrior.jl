struct AugmentedBlackLittermanPriorEstimator{T1 <: AbstractLowOrderPriorEstimatorMap_2_1,
                                             T2 <: AbstractLowOrderPriorEstimatorMap_2_1,
                                             T3 <: AbstractMatrixProcessingEstimator,
                                             T4 <: AbstractRegressionEstimator,
                                             T5 <: AbstractVarianceEstimator,
                                             T6 <: Union{<:BlackLittermanViewsEstimator,
                                                         <:AbstractVector{<:BlackLittermanViewsEstimator}},
                                             T7 <: Union{<:BlackLittermanViewsEstimator,
                                                         <:AbstractVector{<:BlackLittermanViewsEstimator}},
                                             T8 <: DataFrame, T9 <: DataFrame,
                                             T10 <: Union{Nothing, <:AbstractVector},
                                             T11 <: Union{Nothing, <:AbstractVector},
                                             T12 <: Union{Nothing, <:AbstractVector},
                                             T13 <: Real, T14 <: Union{Nothing, <:Real},
                                             T15 <: Union{Nothing, <:Real}} <:
       AbstractLowOrderPriorEstimator_2_1
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
end
function AugmentedBlackLittermanPriorEstimator(;
                                               a_pe::AbstractLowOrderPriorEstimatorMap_2_1 = EmpiricalPriorEstimator(),
                                               f_pe::AbstractLowOrderPriorEstimatorMap_2_1 = EmpiricalPriorEstimator(),
                                               mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                               re::AbstractRegressionEstimator = StepwiseRegression(),
                                               ve::AbstractVarianceEstimator = SimpleVariance(),
                                               a_views::Union{<:BlackLittermanViewsEstimator,
                                                              <:AbstractVector{<:BlackLittermanViewsEstimator}},
                                               f_views::Union{<:BlackLittermanViewsEstimator,
                                                              <:AbstractVector{<:BlackLittermanViewsEstimator}},
                                               a_sets::DataFrame = DataFrame(),
                                               f_sets::DataFrame = DataFrame(),
                                               a_views_conf::Union{Nothing,
                                                                   <:AbstractVector} = nothing,
                                               f_views_conf::Union{Nothing,
                                                                   <:AbstractVector} = nothing,
                                               w::Union{Nothing, <:AbstractVector} = nothing,
                                               rf::Real = 0.0,
                                               l::Union{Nothing, <:Real} = nothing,
                                               tau::Union{Nothing, <:Real} = nothing)
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    if isa(a_views_conf, AbstractVector)
        @smart_assert(isa(a_views, AbstractVector))
        @smart_assert(!isempty(a_views))
        @smart_assert(!isempty(a_views_conf))
        @smart_assert(length(a_views) == length(a_views_conf))
        @smart_assert(all(x -> zero(x) < x < one(x), a_views_conf))
    else
        if isa(a_views, AbstractVector)
            @smart_assert(!isempty(a_views))
        end
    end
    if isa(f_views_conf, AbstractVector)
        @smart_assert(isa(f_views, AbstractVector))
        @smart_assert(!isempty(f_views))
        @smart_assert(!isempty(f_views_conf))
        @smart_assert(length(f_views) == length(f_views_conf))
        @smart_assert(all(x -> zero(x) < x < one(x), f_views_conf))
    else
        if isa(f_views, AbstractVector)
            @smart_assert(!isempty(f_views))
        end
    end
    if !isnothing(tau)
        @smart_assert(tau > zero(tau))
    end
    return AugmentedBlackLittermanPriorEstimator{typeof(a_pe), typeof(f_pe), typeof(mp),
                                                 typeof(re), typeof(ve), typeof(a_views),
                                                 typeof(f_views), typeof(a_sets),
                                                 typeof(f_sets), typeof(a_views_conf),
                                                 typeof(f_views_conf), typeof(w),
                                                 typeof(rf), typeof(l), typeof(tau)}(a_pe,
                                                                                     f_pe,
                                                                                     mp, re,
                                                                                     ve,
                                                                                     a_views,
                                                                                     f_views,
                                                                                     a_sets,
                                                                                     f_sets,
                                                                                     a_views_conf,
                                                                                     f_views_conf,
                                                                                     w, rf,
                                                                                     l, tau)
end
function factory(pe::AugmentedBlackLittermanPriorEstimator,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return AugmentedBlackLittermanPriorEstimator(; a_pe = factory(pe.a_pe, w),
                                                 f_pe = factory(pe.f_pe, w), mp = pe.mp,
                                                 re = pe.re, ve = factory(pe.ve, w),
                                                 a_views = pe.a_views, f_views = pe.f_views,
                                                 a_sets = pe.a_sets, f_sets = pe.f_sets,
                                                 a_views_conf = pe.a_views_conf,
                                                 f_views_conf = pe.f_views_conf, w = pe.w,
                                                 rf = pe.rf, l = pe.l, tau = pe.tau)
end
function Base.getproperty(obj::AugmentedBlackLittermanPriorEstimator, sym::Symbol)
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
function prior(pe::AugmentedBlackLittermanPriorEstimator, X::AbstractMatrix,
               F::AbstractMatrix; dims::Int = 1, strict::Bool = false, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @smart_assert(nrow(pe.a_sets) == size(X, 2))
    @smart_assert(nrow(pe.f_sets) == size(F, 2))
    if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @smart_assert(length(pe.w) == size(X, 2))
            pe.w
        else
            fill(inv(size(X, 2)), size(X, 2))
        end
    end
    # Asset prior.
    a_prior = prior(pe.a_pe, X; strict = strict, kwargs...)
    a_prior_mu, a_prior_sigma = a_prior.mu, a_prior.sigma
    # Factor prior.
    f_prior = prior(pe.f_pe, F; strict = strict)
    f_prior_mu, f_prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    loadings = regression(pe.re, X, F)
    (; b, M) = loadings
    posterior_X = F * transpose(M) .+ transpose(b)
    (; P, Q) = black_litterman_views(pe.a_views, pe.a_sets; datatype = eltype(posterior_X),
                                     strict = strict)
    f_views = black_litterman_views(pe.f_views, pe.f_sets; datatype = eltype(posterior_X),
                                    strict = strict)
    f_P, f_Q = f_views.P, f_views.Q
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    a_views_conf = pe.a_views_conf
    a_omega = tau * Diagonal(if isnothing(a_views_conf)
                                 P * a_prior_sigma * transpose(P)
                             else
                                 idx = iszero.(a_views_conf)
                                 a_views_conf[idx] .= eps(eltype(a_views_conf))
                                 alphas = inv.(a_views_conf) .- 1
                                 alphas ⊙ P * a_prior_sigma * transpose(P)
                             end)
    f_views_conf = pe.f_views_conf
    f_omega = tau * Diagonal(if isnothing(f_views_conf)
                                 f_P * f_prior_sigma * transpose(f_P)
                             else
                                 idx = iszero.(f_views_conf)
                                 f_views_conf[idx] .= eps(eltype(f_views_conf))
                                 alphas = inv.(f_views_conf) .- 1
                                 alphas ⊙ f_P * f_prior_sigma * transpose(f_P)
                             end)
    aug_prior_sigma = hcat(vcat(a_prior_sigma, f_prior_sigma * transpose(M)),
                           vcat(M * f_prior_sigma, f_prior_sigma))
    aug_P = hcat(vcat(P, zeros(size(f_P, 1), size(P, 2))),
                 vcat(zeros(size(P, 1), size(f_P, 2)), f_P))
    aug_Q = vcat(Q, f_Q)
    aug_omega = hcat(vcat(a_omega, zeros(size(f_omega, 1), size(a_omega, 1))),
                     vcat(zeros(size(a_omega, 1), size(f_omega, 1)), f_omega))
    aug_prior_mu = if !isnothing(pe.l)
        pe.l * (vcat(a_prior_sigma, f_prior_sigma * transpose(M))) * w
    else
        vcat(a_prior_mu, f_prior_mu) .- pe.rf
    end
    v1 = tau * aug_prior_sigma * transpose(aug_P)
    v2 = aug_P * v1 + aug_omega
    v3 = aug_Q - aug_P * aug_prior_mu
    aug_posterior_mu = aug_prior_mu + v1 * (v2 \ v3)
    aug_posterior_sigma = aug_prior_sigma + tau * aug_prior_sigma -
                          v1 * (v2 \ transpose(v1))
    matrix_processing!(pe.mp, aug_posterior_sigma, hcat(posterior_X, F))
    posterior_mu = (aug_posterior_mu[1:size(X, 2)] + b) .+ pe.rf
    posterior_sigma = aug_posterior_sigma[1:size(X, 2), 1:size(X, 2)]
    return LowOrderPriorResult(; X = posterior_X, mu = posterior_mu,
                               sigma = posterior_sigma, loadings = loadings,
                               f_mu = f_prior_mu, f_sigma = f_prior_sigma, f_w = f_prior.w)
end

export AugmentedBlackLittermanPriorEstimator
