struct FactorBlackLittermanPriorEstimator{T1 <: AbstractLowOrderPriorEstimatorMap_2_1,
                                          T2 <: AbstractMatrixProcessingEstimator,
                                          T3 <: AbstractMatrixProcessingEstimator,
                                          T4 <: AbstractRegressionEstimator,
                                          T5 <: AbstractVarianceEstimator,
                                          T6 <: Union{<:AbstractString, Expr,
                                                      <:AbstractVector{<:AbstractString},
                                                      <:AbstractVector{Expr},
                                                      <:AbstractVector{<:Union{<:AbstractString,
                                                                               Expr}},
                                            #! Start: to delete
                                                      <:BlackLittermanViewsEstimator,
                                                      <:AbstractVector{<:BlackLittermanViewsEstimator}
                                            #! End: to delete
                                            }, T7 <: DataFrame,
                                          T8 <: Union{Nothing, <:AbstractVector},
                                          T9 <: Union{Nothing, <:AbstractVector},
                                          T10 <: Real, T11 <: Union{Nothing, <:Real},
                                          T12 <: Union{Nothing, <:Real}, T13 <: Bool} <:
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
function FactorBlackLittermanPriorEstimator(;
                                            pe::AbstractLowOrderPriorEstimatorMap_2_1 = EmpiricalPriorEstimator(),
                                            f_mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                            mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                            re::AbstractRegressionEstimator = StepwiseRegression(),
                                            ve::AbstractVarianceEstimator = SimpleVariance(),
                                            views::Union{<:AbstractString, Expr,
                                                         <:AbstractVector{<:AbstractString},
                                                         <:AbstractVector{Expr},
                                                         <:AbstractVector{<:Union{<:AbstractString,
                                                                                  Expr}},
                                                         #! Start: to delete
                                                         <:BlackLittermanViewsEstimator,
                                                         <:AbstractVector{<:BlackLittermanViewsEstimator}
                                                         #! End: to delete
                                                         },
                                            sets::Union{<:AssetSets,
                                                        #! Start: to delete
                                                        <:DataFrame
                                                        #! End: to delete
                                                        } = DataFrame(),
                                            views_conf::Union{Nothing, <:AbstractVector} = nothing,
                                            w::Union{Nothing, <:AbstractVector} = nothing,
                                            rf::Real = 0.0,
                                            l::Union{Nothing, <:Real} = nothing,
                                            tau::Union{Nothing, <:Real} = nothing,
                                            rsd::Bool = true)
    if isa(views_conf, AbstractVector)
        @smart_assert(isa(views, AbstractVector))
        @smart_assert(!isempty(views))
        @smart_assert(!isempty(views_conf))
        @smart_assert(length(views) == length(views_conf))
        @smart_assert(all(x -> zero(x) < x < one(x), views_conf))
    else
        if isa(views, AbstractVector)
            @smart_assert(!isempty(views))
        end
    end
    if !isnothing(tau)
        @smart_assert(tau > zero(tau))
    end
    return FactorBlackLittermanPriorEstimator{typeof(pe), typeof(f_mp), typeof(mp),
                                              typeof(re), typeof(ve), typeof(views),
                                              typeof(sets), typeof(views_conf), typeof(w),
                                              typeof(rf), typeof(l), typeof(tau),
                                              typeof(rsd)}(pe, f_mp, mp, re, ve, views,
                                                           sets, views_conf, w, rf, l, tau,
                                                           rsd)
end
function factory(pe::FactorBlackLittermanPriorEstimator,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return FactorBlackLittermanPriorEstimator(; pe = factory(pe.pe, w), f_mp = pe.f_mp,
                                              mp = pe.mp, re = pe.re,
                                              ve = factory(pe.ve, w), views = pe.views,
                                              sets = pe.sets, views_conf = pe.views_conf,
                                              w = pe.w, rf = pe.rf, l = pe.l, tau = pe.tau,
                                              rsd = pe.rsd)
end
function Base.getproperty(obj::FactorBlackLittermanPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::FactorBlackLittermanPriorEstimator, X::AbstractMatrix, F::AbstractMatrix;
               dims::Int = 1, strict::Bool = false, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    # Factor prior.
    f_prior = prior(pe.pe, F; strict = strict)
    f_prior_mu, f_prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    loadings = regression(pe.re, X, F)
    (; b, M) = loadings
    posterior_X = F * transpose(M) .+ transpose(b)
    f_views = black_litterman_views(pe.views, pe.sets; datatype = eltype(posterior_X),
                                    strict = strict)
    f_P, f_Q = f_views.P, f_views.Q
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    views_conf = pe.views_conf
    f_omega = tau * Diagonal(if isnothing(views_conf)
                                 f_P * f_prior_sigma * transpose(f_P)
                             else
                                 idx = iszero.(views_conf)
                                 views_conf[idx] .= eps(eltype(views_conf))
                                 alphas = inv.(views_conf) .- 1
                                 alphas ⊙ f_P * f_prior_sigma * transpose(f_P)
                             end)
    f_prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @smart_assert(length(pe.w) == size(X, 2))
            pe.w
        else
            fill(inv(size(X, 2)), size(X, 2))
        end
        pe.l * (f_prior_sigma * transpose(M)) * w
    else
        f_prior_mu .- pe.rf
    end
    v1 = tau * f_prior_sigma * transpose(f_P)
    v2 = f_P * v1 + f_omega
    v3 = f_Q - f_P * f_prior_mu
    f_posterior_mu = (f_prior_mu + v1 * (v2 \ v3)) .+ pe.rf
    f_posterior_sigma = f_prior_sigma + tau * f_prior_sigma - v1 * (v2 \ transpose(v1))
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
    return LowOrderPriorResult(; X = posterior_X, mu = posterior_mu,
                               sigma = posterior_sigma,
                               chol = transpose(reshape(posterior_csigma,
                                                        length(posterior_mu), :)),
                               w = f_prior.w, loadings = loadings, f_mu = f_posterior_mu,
                               f_sigma = f_posterior_sigma, f_w = f_prior.w)
end

export FactorBlackLittermanPriorEstimator
