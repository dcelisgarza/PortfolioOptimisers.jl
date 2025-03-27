struct SemiVariance{T1 <: RiskMeasureSettings, T2 <: VarianceFormulation,
                    T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                    T4 <: Union{Nothing, <:AbstractWeights},
                    T5 <: Union{Nothing, <:AbstractVector{<:Real}}} <: TargetRiskMeasure
    settings::T1
    formulation::T2
    target::T3
    w::T4
    mu::T5
end
function SemiVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                      formulation::VarianceFormulation = SOC(),
                      target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                      w::Union{Nothing, <:AbstractWeights} = nothing,
                      mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    return SemiVariance(settings, formulation, target, w, mu)
end
function (r::SemiVariance)(w::AbstractVector, X::AbstractMatrix,
                           fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_target_ret_mu(x, w, r)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return dot(val, val) / (length(x) - 1)
end
function risk_measure_factory(r::SemiVariance, prior::AbstractPriorModel; kwargs...)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = r.target, w = r.w, mu = mu)
end
function risk_measure_factory(r::SemiVariance, prior::EntropyPoolingModel; kwargs...)
    w = risk_measure_nothing_vec_factory(r.w, prior.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = r.target, w = w, mu = mu)
end
function risk_measure_factory(r::SemiVariance,
                              prior::HighOrderPriorModel{<:EntropyPoolingModel, <:Any,
                                                         <:Any, <:Any, <:Any}; kwargs...)
    w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = r.target, w = w, mu = mu)
end
function cluster_risk_measure_factory(r::SemiVariance, prior::AbstractPriorModel,
                                      cluster::AbstractVector; kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = target, w = r.w, mu = mu)
end
function cluster_risk_measure_factory(r::SemiVariance, prior::EntropyPoolingModel,
                                      cluster::AbstractVector; kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    w = risk_measure_nothing_vec_factory(r.w, prior.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = target, w = w, mu = mu)
end
function cluster_risk_measure_factory(r::SemiVariance,
                                      prior::HighOrderPriorModel{<:EntropyPoolingModel,
                                                                 <:Any, <:Any, <:Any,
                                                                 <:Any},
                                      cluster::AbstractVector; kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = target, w = w, mu = mu)
end

export SemiVariance
