struct MeanAbsoluteDeviation{T1 <: RiskMeasureSettings,
                             T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                             T3 <: Union{Nothing, <:AbstractWeights},
                             T4 <: Union{Nothing, <:AbstractVector{<:Real}},
                             T5 <: Union{Nothing, <:AbstractWeights}} <: TargetRiskMeasure
    settings::T1
    target::T2
    w::T3
    mu::T4
    we::T5
end
function MeanAbsoluteDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                               target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                               w::Union{Nothing, <:AbstractWeights} = nothing,
                               mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                               we::Union{Nothing, <:AbstractWeights} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return MeanAbsoluteDeviation{typeof(settings), typeof(target), typeof(w), typeof(mu),
                                 typeof(we)}(settings, target, w, mu, we)
end
function (r::MeanAbsoluteDeviation)(w::AbstractVector, X::AbstractMatrix,
                                    fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_target_ret_mu(x, w, r)
    we = r.we
    return isnothing(we) ? mean(abs.(x .- mu)) : mean(abs.(x .- mu), we)
end
function risk_measure_factory(r::MeanAbsoluteDeviation, prior::AbstractPriorModel, args...)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return MeanAbsoluteDeviation(; settings = r.settings, target = r.target, w = r.w,
                                 mu = mu, we = r.we)
end
function risk_measure_factory(r::MeanAbsoluteDeviation, prior::EntropyPoolingModel, args...)
    w = risk_measure_nothing_vec_factory(r.w, prior.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    we = risk_measure_nothing_vec_factory(r.we, prior.w)
    return MeanAbsoluteDeviation(; settings = r.settings, target = r.target, w = w, mu = mu,
                                 we = we)
end
function risk_measure_factory(r::MeanAbsoluteDeviation,
                              (prior::HighOrderPriorModel{<:EntropyPoolingModel, <:Any,
                                                          <:Any, <:Any, <:Any}), args...)
    w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    we = risk_measure_nothing_vec_factory(r.we, prior.pm.w)
    return MeanAbsoluteDeviation(; settings = r.settings, target = r.target, w = w, mu = mu,
                                 we = we)
end
function cluster_risk_measure_factory(r::MeanAbsoluteDeviation, prior::AbstractPriorModel,
                                      cluster::AbstractVector, args...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return MeanAbsoluteDeviation(; settings = r.settings, target = target, w = r.w, mu = mu,
                                 we = r.we)
end
function cluster_risk_measure_factory(r::MeanAbsoluteDeviation, prior::EntropyPoolingModel,
                                      cluster::AbstractVector, args...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    w = risk_measure_nothing_vec_factory(r.w, prior.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    we = risk_measure_nothing_vec_factory(r.we, prior.w)
    return MeanAbsoluteDeviation(; settings = r.settings, target = target, w = w, mu = mu,
                                 we = we)
end
function cluster_risk_measure_factory(r::MeanAbsoluteDeviation,
                                      prior::HighOrderPriorModel{<:EntropyPoolingModel,
                                                                 <:Any, <:Any, <:Any,
                                                                 <:Any},
                                      cluster::AbstractVector, args...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    we = risk_measure_nothing_vec_factory(r.we, prior.pm.w)
    return MeanAbsoluteDeviation(; settings = r.settings, target = target, w = w, mu = mu,
                                 we = we)
end

export MeanAbsoluteDeviation
