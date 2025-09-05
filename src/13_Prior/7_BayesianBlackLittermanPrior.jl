struct BayesianBlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7} <:
       AbstractLowOrderPriorEstimator_2_2
    pe::T1
    mp::T2
    views::T3
    sets::T4
    views_conf::T5
    rf::T6
    tau::T7
end
function BayesianBlackLittermanPrior(;
                                     pe::AbstractLowOrderPriorEstimatorMap_2_2 = FactorPrior(;
                                                                                             pe = EmpiricalPrior(;
                                                                                                                 me = EquilibriumExpectedReturns())),
                                     mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                     views::Union{<:LinearConstraintEstimator,
                                                  <:BlackLittermanViews},
                                     sets::Union{Nothing, <:AssetSets} = nothing,
                                     views_conf::Union{Nothing, <:AbstractVector} = nothing,
                                     rf::Real = 0.0, tau::Union{Nothing, <:Real} = nothing)
    if isa(views, LinearConstraintEstimator)
        @argcheck(!isnothing(sets))
    end
    assert_bl_views_conf(views_conf, views)
    if !isnothing(tau)
        @argcheck(tau > zero(tau))
    end
    return BayesianBlackLittermanPrior(pe, mp, views, sets, views_conf, rf, tau)
end
function factory(pe::BayesianBlackLittermanPrior,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return BayesianBlackLittermanPrior(; pe = factory(pe.pe, w), mp = pe.mp,
                                       views = pe.views, sets = pe.sets,
                                       views_conf = pe.views_conf, rf = pe.rf, tau = pe.tau)
end
function Base.getproperty(obj::BayesianBlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::BayesianBlackLittermanPrior, X::AbstractMatrix, F::AbstractMatrix;
               dims::Int = 1, strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @argcheck(length(pe.sets.dict[pe.sets.key]) == size(F, 2))
    prior_result = prior(pe.pe, X, F; strict = strict, kwargs...)
    posterior_X, prior_sigma, f_mu, f_sigma, rr = prior_result.X, prior_result.sigma,
                                                  prior_result.f_mu, prior_result.f_sigma,
                                                  prior_result.rr
    (; P, Q) = black_litterman_views(pe.views, pe.sets; datatype = eltype(posterior_X),
                                     strict = strict)
    tau = isnothing(pe.tau) ? inv(size(F, 1)) : pe.tau
    f_omega = tau * calc_omega(pe.views_conf, P, f_sigma)
    (; b, M) = rr
    sigma_hat = f_sigma \ I + transpose(P) * (f_omega \ P)
    mu_hat = sigma_hat \ (f_sigma \ f_mu + transpose(P) * (f_omega \ Q))
    v1 = prior_sigma \ M
    v2 = sigma_hat + transpose(M) * v1
    v3 = prior_sigma \ I
    posterior_sigma = (v3 - v1 * (v2 \ transpose(M)) * v3) \ I
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    posterior_mu = (posterior_sigma * v1 * (v2 \ sigma_hat) * mu_hat + b) .+ pe.rf
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         rr = rr, f_mu = f_mu, f_sigma = f_sigma)
end

export BayesianBlackLittermanPrior
