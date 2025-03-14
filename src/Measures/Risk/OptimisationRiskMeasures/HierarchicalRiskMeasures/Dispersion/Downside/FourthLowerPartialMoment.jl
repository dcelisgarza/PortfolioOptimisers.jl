struct FourthLowerPartialMoment{T1 <: RiskMeasureSettings,
                                T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                                T3 <: Union{Nothing, <:AbstractWeights},
                                T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetHierarchicalRiskMeasure
    settings::T1
    target::T2
    w::T3
    mu::T4
end
function FourthLowerPartialMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                                  w::Union{Nothing, <:AbstractWeights} = nothing,
                                  mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return FourthLowerPartialMoment{typeof(settings), typeof(target), typeof(w),
                                    typeof(mu)}(settings, target, w, mu)
end
function (r::FourthLowerPartialMoment)(X::AbstractMatrix, w::AbstractVector,
                                       fees::Fees = Fees())
    x = calc_net_returns(X, w, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return sum(val[val .<= zero(eltype(val))] .^ 4) / length(x)
end
function risk_measure_factory(r::FourthLowerPartialMoment; prior::AbstractPriorModel,
                              kwargs...)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return FourthLowerPartialMoment(; settings = r.settings, target = r.target, w = r.w,
                                    mu = mu)
end
function cluster_risk_measure_factory(r::FourthLowerPartialMoment;
                                      prior::AbstractPriorModel, cluster::AbstractVector,
                                      kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return FourthLowerPartialMoment(; settings = r.settings, target = target, w = r.w,
                                    mu = mu)
end

export FourthLowerPartialMoment
