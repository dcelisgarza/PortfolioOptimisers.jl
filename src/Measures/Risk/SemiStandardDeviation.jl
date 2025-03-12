struct SemiStandardDeviation{T1 <: RiskMeasureSettings,
                             T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                             T3 <: Union{Nothing, <:AbstractWeights},
                             T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetRiskMeasure
    settings::T1
    target::T2
    w::T3
    mu::T4
end
function SemiStandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                               target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                               w::Union{Nothing, <:AbstractWeights} = nothing,
                               mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    return SemiStandardDeviation{typeof(settings), typeof(target), typeof(w), typeof(mu)}(settings,
                                                                                          target,
                                                                                          w,
                                                                                          mu)
end
function (r::SemiStandardDeviation)(X::AbstractMatrix, w::AbstractVector,
                                    fees::Fees = Fees())
    x = calc_net_returns(X, w, fees)
    mu = calc_target_ret_mu(x, w, r)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return sqrt(dot(val, val) / (length(x) - 1))
end
function risk_measure_factory(r::SemiStandardDeviation, prior::AbstractPriorModel,
                              cluster::AbstractVector)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return SemiStandardDeviation(; settings = r.settings, target = target, w = r.w, mu = mu)
end
function risk_measure_factory(r::SemiStandardDeviation, prior::AbstractPriorModel)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return SemiStandardDeviation(; settings = r.settings, target = r.target, w = r.w,
                                 mu = mu)
end

export SemiStandardDeviation
