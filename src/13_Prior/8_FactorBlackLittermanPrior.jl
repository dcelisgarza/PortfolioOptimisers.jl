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
                                   views::Union{<:LinearConstraintEstimator,
                                                <:BlackLittermanViews},
                                   sets::Union{Nothing, <:AssetSets} = nothing,
                                   views_conf::Union{Nothing, <:Real, <:AbstractVector} = nothing,
                                   w::Union{Nothing, <:AbstractWeights} = nothing,
                                   rf::Real = 0.0, l::Union{Nothing, <:Real} = nothing,
                                   tau::Union{Nothing, <:Real} = nothing, rsd::Bool = true)
    if isa(views, LinearConstraintEstimator)
        @argcheck(!isnothing(sets))
    end
    assert_bl_views_conf(views_conf, views)
    if !isnothing(tau)
        @argcheck(tau > zero(tau))
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
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @argcheck(length(pe.sets.dict[pe.sets.key]) == size(F, 2))
    # Factor prior.
    f_prior = prior(pe.pe, F; strict = strict)
    prior_mu, prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    rr = regression(pe.re, X, F)
    (; b, M) = rr
    posterior_X = F * transpose(M) .+ transpose(b)
    (; P, Q) = black_litterman_views(pe.views, pe.sets; datatype = eltype(posterior_X),
                                     strict = strict)
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    omega = tau * calc_omega(pe.views_conf, P, prior_sigma)
    prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @argcheck(length(pe.w) == size(X, 2))
            pe.w
        else
            iN = inv(size(X, 2))
            range(; start = iN, stop = iN, length = size(X, 2))
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
                                                  :)), w = f_prior.w, rr = rr,
                         f_mu = f_posterior_mu, f_sigma = f_posterior_sigma,
                         f_w = f_prior.w)
end

export FactorBlackLittermanPrior
