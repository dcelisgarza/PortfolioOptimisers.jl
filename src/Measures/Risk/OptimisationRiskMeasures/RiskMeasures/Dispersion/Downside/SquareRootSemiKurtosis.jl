struct SquareRootSemiKurtosis{T1 <: RiskMeasureSettings,
                              T2 <: Union{Nothing, <:AbstractWeights},
                              T3 <: Union{Nothing, <:AbstractVector{<:Real}},
                              T4 <: Union{Nothing, <:AbstractMatrix}} <: MuRiskMeasure
    settings::T1
    w::T2
    mu::T3
    kt::T4
end
function SquareRootSemiKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                w::Union{Nothing, <:AbstractWeights} = nothing,
                                mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                                kt::Union{Nothing, <:AbstractMatrix, Nothing} = nothing)
    mu_flag = isa(mu, AbstractVector)
    kt_flag = isa(kt, AbstractMatrix)
    if mu_flag
        @smart_assert(!isempty(mu))
    end
    if kt_flag
        @smart_assert(!isempty(kt))
        issquare(kt)
    end
    if mu_flag && kt_flag
        @smart_assert(length(mu)^2 == size(kt, 2))
    end
    return SquareRootSemiKurtosis{typeof(settings), typeof(w), typeof(mu), typeof(kt)}(settings,
                                                                                       w,
                                                                                       mu,
                                                                                       kt)
end
function (r::SquareRootSemiKurtosis)(X::AbstractMatrix, w::AbstractVector,
                                     fees::Fees = Fees())
    x = calc_net_returns(X, w, fees)
    mu = calc_ret_mu(x, w, r)
    val = x .- mu
    return sqrt(sum(val[val .<= zero(eltype(val))] .^ 4) / length(x))
end
function _risk_measure_factory(r::SquareRootSemiKurtosis, prior::HighOrderPriorModel)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    skt = risk_measure_nothing_matrix_factory(r.kt, prior.skt)
    return SquareRootSemiKurtosis(; settings = r.settings, w = r.w, mu = mu, kt = skt)
end
function _risk_measure_factory(r::SquareRootSemiKurtosis, prior::LowOrderAbstractPriorModel)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    skt = risk_measure_nothing_matrix_factory(r.kt, nothing)
    return SquareRootSemiKurtosis(; settings = r.settings, w = r.w, mu = mu, kt = skt)
end
function risk_measure_factory(r::SquareRootSemiKurtosis; prior::AbstractPriorModel,
                              kwargs...)
    return _risk_measure_factory(r, prior)
end
function _cluster_risk_measure_factory(r::SquareRootSemiKurtosis,
                                       prior::HighOrderPriorModel, cluster::AbstractVector)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
    skt = risk_measure_nothing_matrix_factory(r.kt, prior.skt, idx)
    return SquareRootSemiKurtosis(; settings = r.settings, w = r.w, mu = mu, kt = skt)
end
function _cluster_risk_measure_factory(r::SquareRootSemiKurtosis,
                                       prior::LowOrderAbstractPriorModel,
                                       cluster::AbstractVector)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
    skt = risk_measure_nothing_matrix_factory(r.kt, nothing, idx)
    return SquareRootSemiKurtosis(; settings = r.settings, w = r.w, mu = mu, kt = skt)
end
function cluster_risk_measure_factory(r::SquareRootSemiKurtosis; prior::AbstractPriorModel,
                                      cluster::AbstractVector, kwargs...)
    return _cluster_risk_measure_factory(r, prior, cluster)
end

export SquareRootSemiKurtosis
