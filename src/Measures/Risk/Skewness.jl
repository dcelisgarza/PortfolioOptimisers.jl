struct Skewness{T1 <: RiskMeasureSettings, T2 <: PortfolioOptimisersVarianceEstimator,
                T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                T4 <: Union{Nothing, <:AbstractWeights},
                T5 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetNoOptimisationRiskMeasure
    settings::T1
    ve::T2
    target::T3
    w::T4
    mu::T5
end
function Skewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                  target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                  w::Union{Nothing, <:AbstractWeights} = nothing,
                  mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return Skewness{typeof(settings), typeof(ve), typeof(target), typeof(w), typeof(mu)}(settings,
                                                                                         ve,
                                                                                         target,
                                                                                         w,
                                                                                         mu)
end
function (r::Skewness)(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    x = calc_net_returns(X, w, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    sigma = std(r.ve, x)
    return sum(val .^ 3) / length(x) / sigma^3
end
function risk_measure_factory(r::Skewness; prior::AbstractPriorModel, kwargs...)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return Skewness(; settings = r.settings, target = r.target, w = r.w, mu = mu)
end
function cluster_risk_measure_factory(r::Skewness; prior::AbstractPriorModel,
                                      cluster::AbstractVector, kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return Skewness(; settings = r.settings, target = target, w = r.w, mu = mu)
end

export Skewness
