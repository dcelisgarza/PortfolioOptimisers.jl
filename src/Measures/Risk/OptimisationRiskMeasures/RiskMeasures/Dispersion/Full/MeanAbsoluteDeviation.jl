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
function (r::MeanAbsoluteDeviation)(X::AbstractMatrix, w::AbstractVector,
                                    fees::Fees = Fees())
    x = calc_net_returns(X, w, fees)
    mu = calc_target_ret_mu(x, w, r)
    we = r.we
    return isnothing(we) ? mean(abs.(x .- mu)) : mean(abs.(x .- mu), we)
end
function risk_measure_factory(r::MeanAbsoluteDeviation; prior::AbstractPriorModel,
                              kwargs...)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return MeanAbsoluteDeviation(; settings = r.settings, target = r.target, w = r.w,
                                 mu = mu, we = r.we)
end
function cluster_risk_measure_factory(r::MeanAbsoluteDeviation; prior::AbstractPriorModel,
                                      cluster::AbstractVector, kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return MeanAbsoluteDeviation(; settings = r.settings, target = target, w = r.w, mu = mu,
                                 we = r.we)
end

export MeanAbsoluteDeviation
