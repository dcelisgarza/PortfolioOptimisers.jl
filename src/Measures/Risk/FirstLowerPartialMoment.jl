struct FirstLowerPartialModel{T1 <: RiskMeasureSettings,
                              T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                              T3 <: Union{Nothing, <:AbstractWeights},
                              T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetRiskMeasure
    settings::T1
    target::T2
    w::T3
    mu::T4
end
function FirstLowerPartialModel(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0,
                                w::Union{Nothing, <:AbstractWeights} = nothing,
                                mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    return FirstLowerPartialModel{typeof(settings), typeof(target), typeof(w), typeof(mu)}(settings,
                                                                                           target,
                                                                                           w,
                                                                                           mu)
end
function (r::FirstLowerPartialModel)(X::AbstractMatrix, w::AbstractVector,
                                     fees::Fees = Fees())
    x = calc_net_returns(X, w, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    val = val[val .<= zero(eltype(val))]
    return -sum(val) / length(x)
end
function cluster_risk_measure_factory(r::FirstLowerPartialModel; prior::AbstractPriorModel,
                                      cluster::AbstractVector, kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return FirstLowerPartialModel(; settings = r.settings, target = target, w = r.w,
                                  mu = mu)
end
function risk_measure_factory(r::FirstLowerPartialModel; prior::AbstractPriorModel,
                              kwargs...)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return FirstLowerPartialModel(; settings = r.settings, target = r.target, w = r.w,
                                  mu = mu)
end

export FirstLowerPartialModel
