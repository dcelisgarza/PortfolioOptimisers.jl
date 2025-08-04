struct BlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7} <:
       AbstractLowOrderPriorEstimator_1o2_1o2
    pe::T1
    mp::T2
    views::T3
    sets::T4
    views_conf::T5
    rf::T6
    tau::T7
end
function Base.getproperty(obj::BlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function BlackLittermanPrior(;
                             pe::AbstractLowOrderPriorEstimatorMap_1o2_1o2 = EmpiricalPrior(;
                                                                                            me = EquilibriumExpectedReturns()),
                             mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                             views::LinearConstraintEstimator,
                             sets::Union{<:AssetSets,
                                         #! Start: to delete
                                         <:DataFrame
                                         #! End: to delete
                                         } = DataFrame(),
                             views_conf::Union{Nothing, <:AbstractVector} = nothing,
                             rf::Real = 0.0, tau::Union{Nothing, <:Real} = nothing)
    if isa(views_conf, AbstractVector)
        @smart_assert(isa(views.val, AbstractVector))
        @smart_assert(!isempty(views_conf))
        @smart_assert(length(views.val) == length(views_conf))
        @smart_assert(all(x -> zero(x) < x < one(x), views_conf))
    end
    if !isnothing(tau)
        @smart_assert(tau > zero(tau))
    end
    return BlackLittermanPrior(pe, mp, views, sets, views_conf, rf, tau)
end
function factory(pe::BlackLittermanPrior, w::Union{Nothing, <:AbstractWeights} = nothing)
    return BlackLittermanPrior(; pe = factory(pe.pe, w), mp = pe.mp, views = pe.views,
                               sets = pe.sets, views_conf = pe.views_conf, rf = pe.rf,
                               tau = pe.tau)
end
function prior(pe::BlackLittermanPrior, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    @smart_assert(length(pe.sets.dict[pe.sets.key]) == size(X, 2))
    prior_model = prior(pe.pe, X, F; strict = strict, kwargs...)
    posterior_X, prior_mu, prior_sigma = prior_model.X, prior_model.mu, prior_model.sigma
    (; P, Q) = black_litterman_views(pe.views, pe.sets; datatype = eltype(posterior_X),
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
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma)
end

export BlackLittermanPrior
