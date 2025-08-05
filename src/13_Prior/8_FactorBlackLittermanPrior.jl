struct FactorBlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13} <:
       AbstractLowOrderPriorEstimator_2_1
    pe::T1
    f_mp::T2
    mp::T3
    re::T4
    ve::T5
    views::T6
    sets::T7
    views_conf::T8
    w::T9
    rf::T10
    l::T11
    tau::T12
    rsd::T13
end
function FactorBlackLittermanPrior(;
                                   pe::AbstractLowOrderPriorEstimatorMap_2_1 = EmpiricalPrior(),
                                   f_mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                   mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                   re::AbstractRegressionEstimator = StepwiseRegression(),
                                   ve::AbstractVarianceEstimator = SimpleVariance(),
                                   views::LinearConstraintEstimator,
                                   sets::Union{<:AssetSets,
                                               #! Start: to delete
                                               <:DataFrame
                                               #! End: to delete
                                               } = DataFrame(),
                                   views_conf::Union{Nothing, <:AbstractVector} = nothing,
                                   w::Union{Nothing, <:AbstractWeights} = nothing,
                                   rf::Real = 0.0, l::Union{Nothing, <:Real} = nothing,
                                   tau::Union{Nothing, <:Real} = nothing, rsd::Bool = true)
    if isa(views_conf, AbstractVector)
        @smart_assert(isa(views.val, AbstractVector))
        @smart_assert(!isempty(views_conf))
        @smart_assert(length(views.val) == length(views_conf))
        @smart_assert(all(x -> zero(x) < x < one(x), views_conf))
    end
    if !isnothing(tau)
        @smart_assert(tau > zero(tau))
    end
    return FactorBlackLittermanPrior(pe, f_mp, mp, re, ve, views, sets, views_conf, w, rf,
                                     l, tau, rsd)
end
function factory(pe::FactorBlackLittermanPrior,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return FactorBlackLittermanPrior(; pe = factory(pe.pe, w), f_mp = pe.f_mp, mp = pe.mp,
                                     re = pe.re, ve = factory(pe.ve, w), views = pe.views,
                                     sets = pe.sets, views_conf = pe.views_conf, w = pe.w,
                                     rf = pe.rf, l = pe.l, tau = pe.tau, rsd = pe.rsd)
end
function Base.getproperty(obj::FactorBlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::FactorBlackLittermanPrior, X::AbstractMatrix, F::AbstractMatrix;
               dims::Int = 1, strict::Bool = false, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @smart_assert(length(pe.sets.dict[pe.sets.key]) == size(F, 2))
    # Factor prior.
    f_prior = prior(pe.pe, F; strict = strict)
    prior_mu, prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    loadings = regression(pe.re, X, F)
    (; b, M) = loadings
    posterior_X = F * transpose(M) .+ transpose(b)
    (; P, Q) = black_litterman_views(pe.views, pe.sets; datatype = eltype(posterior_X),
                                     strict = strict)
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    omega = tau * calc_omega(pe.views_conf, P, prior_sigma)
    prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @smart_assert(length(pe.w) == size(X, 2))
            pe.w
        else
            range(; start = inv(size(X, 2)), stop = inv(size(X, 2)), length = size(X, 2))
        end
        pe.l * (prior_sigma * transpose(M)) * w
    else
        prior_mu .- pe.rf
    end
    f_posterior_mu, f_posterior_sigma = vanilla_posteriors(tau, pe.rf, prior_mu,
                                                           prior_sigma, omega, P, Q)
    matrix_processing!(pe.f_mp, f_posterior_sigma, F)
    # Reconstruct the posteriors using the black litterman adjusted factor statistics.
    posterior_mu = M * f_posterior_mu + b
    posterior_sigma = M * f_posterior_sigma * transpose(M)
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    posterior_csigma = M * cholesky(f_posterior_sigma).L
    if pe.rsd
        err = X - posterior_X
        err_sigma = diagm(vec(var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         chol = transpose(reshape(posterior_csigma, length(posterior_mu),
                                                  :)), w = f_prior.w, loadings = loadings,
                         f_mu = f_posterior_mu, f_sigma = f_posterior_sigma,
                         f_w = f_prior.w)
end

export FactorBlackLittermanPrior
