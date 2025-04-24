struct BlackLittermanPriorEstimator{T1 <: AbstractLowOrderPriorEstimatorMap_1o2_1o2,
                                    T2 <: AbstractMatrixProcessingEstimator,
                                    T3 <: Union{<:BlackLittermanViewsEstimator,
                                                <:AbstractVector{<:BlackLittermanViewsEstimator}},
                                    T4 <: DataFrame, T5 <: Real,
                                    T6 <: Union{Nothing, <:AbstractVector},
                                    T7 <: Union{Nothing, <:Real}} <:
       AbstractLowOrderPriorEstimator_1o2_1o2
    pe::T1
    mp::T2
    views::T3
    sets::T4
    rf::T5
    views_conf::T6
    tau::T7
end
function Base.getproperty(obj::BlackLittermanPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function BlackLittermanPriorEstimator(;
                                      pe::AbstractLowOrderPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(;
                                                                                                              me = EquilibriumExpectedReturns()),
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
    return BlackLittermanPriorEstimator{typeof(pe), typeof(mp), typeof(views), typeof(sets),
                                        typeof(rf), typeof(views_conf), typeof(tau)}(pe, mp,
                                                                                     views,
                                                                                     sets,
                                                                                     rf,
                                                                                     views_conf,
                                                                                     tau)
end
function factory(pe::BlackLittermanPriorEstimator,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return BlackLittermanPriorEstimator(; pe = factory(pe.pe, w), mp = pe.mp,
                                        views = pe.views, sets = pe.sets, rf = pe.rf,
                                        views_conf = pe.views_conf, tau = pe.tau)
end
function prior(pe::BlackLittermanPriorEstimator, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    @smart_assert(nrow(pe.sets) == size(X, 2))

    prior_model = prior(pe.pe, X, F; strict = strict, kwargs...)
    posterior_X, prior_mu, prior_sigma = prior_model.X, prior_model.mu, prior_model.sigma

    (; P, Q) = views = black_litterman_views(pe.views, pe.sets;
                                             datatype = eltype(posterior_X),
                                             strict = strict)
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    views_conf = pe.views_conf
    omega = tau * Diagonal(if isnothing(views_conf)
                               P * prior_sigma * transpose(P)
                           else
                               idx = iszero.(views_conf)
                               views_conf[idx] .= eps(eltype(views_conf))
                               alphas = inv.(views_conf) .- 1
                               alphas ⊙ P * prior_sigma * transpose(P)
                           end)

    v1 = tau * prior_sigma * transpose(P)
    v2 = P * v1 + omega
    v3 = Q - P * prior_mu
    posterior_mu = (prior_mu + v1 * (v2 \ v3)) .+ pe.rf
    posterior_sigma = prior_sigma + tau * prior_sigma - v1 * (v2 \ transpose(v1))
    matrix_processing!(pe.mp, posterior_sigma, posterior_X)
    return EmpiricalPriorResult(; X = posterior_X, mu = posterior_mu,
                                sigma = posterior_sigma)
end

export BlackLittermanPriorEstimator
