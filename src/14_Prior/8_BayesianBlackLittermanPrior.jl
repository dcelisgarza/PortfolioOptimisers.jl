struct BayesianBlackLittermanPriorEstimator{T1 <: AbstractLowOrderPriorEstimatorMap_2_2,
                                            T2 <: AbstractMatrixProcessingEstimator,
                                            T3 <: Union{<:BlackLittermanViewsEstimator,
                                                        <:AbstractVector{<:BlackLittermanViewsEstimator}},
                                            T4 <: DataFrame, T5 <: Real,
                                            T6 <: Union{Nothing, <:AbstractVector},
                                            T7 <: Union{Nothing, <:Real}} <:
       AbstractLowOrderPriorEstimator_2_2
    pe::T1
    mp::T2
    views::T3
    sets::T4
    rf::T5
    views_conf::T6
    tau::T7
end
function BayesianBlackLittermanPriorEstimator(;
                                              pe::AbstractLowOrderPriorEstimatorMap_2_2 = FactorPriorEstimator(;
                                                                                                               pe = EmpiricalPriorEstimator(;
                                                                                                                                            me = EquilibriumExpectedReturns())),
                                              mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                              views::Union{<:BlackLittermanViewsEstimator,
                                                           <:AbstractVector{<:BlackLittermanViewsEstimator}},
                                              sets::DataFrame = DataFrame(), rf::Real = 0.0,
                                              views_conf::Union{Nothing, <:AbstractVector} = nothing,
                                              tau::Union{Nothing, <:Real} = nothing)
    if isa(views_conf, AbstractVector)
        @smart_assert(isa(views, AbstractVector))
        @smart_assert(!isempty(views))
        @smart_assert(!isempty(views_conf))
        @smart_assert(length(views) == length(views_conf))
        @smart_assert(all(zero(eltype(views_conf)) .< views_conf .< one(eltype(views_conf))))
    else
        if isa(views, AbstractVector)
            @smart_assert(!isempty(views))
        end
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
function factory(pe::BayesianBlackLittermanPriorEstimator,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return BayesianBlackLittermanPriorEstimator(; pe = factory(pe.pe, w), mp = pe.mp,
                                                views = pe.views, sets = pe.sets,
                                                rf = pe.rf, views_conf = pe.views_conf,
                                                tau = pe.tau)
end
function Base.getproperty(obj::BayesianBlackLittermanPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::BayesianBlackLittermanPriorEstimator, X::AbstractMatrix,
               F::AbstractMatrix; dims::Int = 1, strict::Bool = false, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @smart_assert(nrow(pe.sets) == size(F, 2))
    prior_model = prior(pe.pe, X, F; strict = strict, kwargs...)
    posterior_X, prior_sigma, f_mu, f_sigma, loadings = prior_model.X, prior_model.sigma,
                                                        prior_model.f_mu,
                                                        prior_model.f_sigma,
                                                        prior_model.loadings
    f_views = black_litterman_views(pe.views, pe.sets; datatype = eltype(posterior_X),
                                    strict = strict)
    f_P, f_Q = f_views.P, f_views.Q
    tau = isnothing(pe.tau) ? inv(size(F, 1)) : pe.tau
    views_conf = pe.views_conf
    f_omega = tau * Diagonal(if isnothing(views_conf)
                                 f_P * f_sigma * transpose(f_P)
                             else
                                 idx = iszero.(views_conf)
                                 views_conf[idx] .= eps(eltype(views_conf))
                                 alphas = inv.(views_conf) .- 1
                                 alphas ⊙ f_P * f_sigma * transpose(f_P)
                             end)
    (; b, M) = loadings
    sigma_hat = inv(f_sigma) + transpose(f_P) * (f_omega \ f_P)
    mu_hat = sigma_hat \ (f_sigma \ f_mu + transpose(f_P) * (f_omega \ f_Q))
    v1 = prior_sigma \ M
    v2 = sigma_hat + transpose(M) * v1
    v3 = inv(prior_sigma)
    posterior_sigma = inv(v3 - v1 * (v2 \ transpose(M)) * v3)
    matrix_processing!(pe.mp, posterior_sigma, posterior_X)
    posterior_mu = (posterior_sigma * v1 * (v2 \ sigma_hat) * mu_hat + b) .+ pe.rf
    return EmpiricalPartialFactorPriorResult(;
                                             pr = EmpiricalPriorResult(; X = posterior_X,
                                                                       mu = posterior_mu,
                                                                       sigma = posterior_sigma),
                                             fpr = PartialFactorPriorResult(; mu = f_mu,
                                                                            sigma = f_sigma,
                                                                            loadings = loadings))
end

export BayesianBlackLittermanPriorEstimator
