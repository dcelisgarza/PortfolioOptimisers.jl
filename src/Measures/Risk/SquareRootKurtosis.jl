struct SquareRootKurtosis{T1 <: RiskMeasureSettings,
                          T2 <: Union{Nothing, <:AbstractWeights},
                          T3 <: Union{Nothing, <:AbstractVector{<:Real}},
                          T4 <: Union{Nothing, <:AbstractMatrix}} <: MuRiskMeasure
    settings::T1
    w::T2
    mu::T3
    kt::T4
end
function SquareRootKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                            w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                            kt::Union{Nothing, <:AbstractMatrix, Nothing} = nothing)
    if !isnothing(kt) && !isempty(kt)
        @smart_assert(size(kt, 1) == size(kt, 2))
    end
    return SquareRootKurtosis{typeof(settings), typeof(w), typeof(mu), typeof(kt)}(settings,
                                                                                   w, mu,
                                                                                   kt)
end
function (r::SquareRootKurtosis)(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    x = calc_net_returns(X, w, fees)
    mu = calc_ret_mu(x, w, r)
    val = x .- mu
    return sqrt(sum(val .^ 4) / length(x))
end
function _cluster_risk_measure_factory(r::SquareRootKurtosis, prior::HighOrderPriorModel,
                                       cluster::AbstractVector)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    idx = fourth_moment_cluster_factory(size(prior.X, 2), cluster)
    kt = risk_measure_nothing_matrix_factory(r.kt, prior.kt, idx)
    return SquareRootKurtosis(; settings = r.settings, w = r.w, mu = mu, kt = kt)
end
function _cluster_risk_measure_factory(r::SquareRootKurtosis,
                                       prior::LowOrderAbstractPriorModel,
                                       cluster::AbstractVector)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    idx = fourth_moment_cluster_factory(size(prior.X, 2), cluster)
    kt = risk_measure_nothing_matrix_factory(r.kt, nothing, idx)
    return SquareRootKurtosis(; settings = r.settings, w = r.w, mu = mu, kt = kt)
end
function cluster_risk_measure_factory(r::SquareRootKurtosis; prior::AbstractPriorModel,
                                      cluster::AbstractVector, kwargs...)
    return _cluster_risk_measure_factory(r, prior, cluster)
end
function _risk_measure_factory(r::SquareRootKurtosis, prior::HighOrderPriorModel)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    kt = risk_measure_nothing_matrix_factory(r.kt, prior.kt)
    return SquareRootKurtosis(; settings = r.settings, w = r.w, mu = mu, kt = kt)
end
function _risk_measure_factory(r::SquareRootKurtosis, prior::LowOrderAbstractPriorModel)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    kt = risk_measure_nothing_matrix_factory(r.kt, nothing)
    return SquareRootKurtosis(; settings = r.settings, w = r.w, mu = mu, kt = kt)
end
function risk_measure_factory(r::SquareRootKurtosis; prior::AbstractPriorModel, kwargs...)
    return _risk_measure_factory(r, prior)
end

export SquareRootKurtosis
