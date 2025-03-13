struct NegativeSemiSkewness{T1 <: RiskMeasureSettings, T2 <: MatrixProcessing,
                            T3 <: Union{Nothing, <:AbstractMatrix},
                            T4 <: Union{Nothing, <:AbstractMatrix}} <: SkewRiskMeasure
    settings::T1
    mp::T2
    sk::T3
    V::T4
end
function NegativeSemiSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                              mp::MatrixProcessing = NonPositiveDefiniteMatrixProcessing(),
                              sk::Union{Nothing, <:AbstractMatrix} = nothing,
                              V::Union{Nothing, <:AbstractMatrix} = nothing)
    sk_flag = isa(sk, AbstractMatrix)
    V_flag = isa(V, AbstractMatrix)
    if any((!sk_flag, !V_flag))
        @smart_assert(all((!sk_flag, !V_flag)),
                      "If either sk or V, is nothing, both must be nothing.")
    else
        @smart_assert(!isempty(sk))
        @smart_assert(!isempty(V))
        @smart_assert(size(sk, 1)^2 == size(sk, 2))
        issquare(V)
    end
    return NegativeSemiSkewness{typeof(settings), typeof(mp), typeof(sk), typeof(V)}(settings,
                                                                                     mp, sk,
                                                                                     V)
end
function (r::NegativeSemiSkewness)(w::AbstractVector)
    return sqrt(dot(w, r.V, w))
end
function _risk_measure_factory(r::NegativeSemiSkewness, prior::HighOrderPriorModel)
    sk = risk_measure_nothing_matrix_factory(r.sk, prior.ssk)
    V = risk_measure_nothing_matrix_factory(r.V, prior.SV)
    return NegativeSemiSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V)
end
function _risk_measure_factory(r::NegativeSemiSkewness, ::LowOrderAbstractPriorModel)
    sk = risk_measure_nothing_matrix_factory(r.sk, nothing)
    V = risk_measure_nothing_matrix_factory(r.V, nothing)
    return NegativeSemiSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V)
end
function risk_measure_factory(r::NegativeSemiSkewness; prior::AbstractPriorModel, kwargs...)
    return _risk_measure_factory(r, prior)
end
function cluster_negative_skewness(::NegativeSemiSkewness{<:Any, <:Any, Nothing, <:Any},
                                   ::Nothing, prior::AbstractPriorModel,
                                   cluster::AbstractVector)
    throw(ArgumentError("Neither the risk measure, nor the prior have the required data."))
end
function cluster_negative_skewness(skew_rm::NegativeSemiSkewness{<:Any, <:Any,
                                                                 <:AbstractMatrix, <:Any},
                                   prior::AbstractPriorModel, cluster::AbstractVector)
    idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
    sk = view(skew_rm.sk, cluster, idx)
    V = __coskewness(sk, prior.X, skew_rm.mp)
    if all(iszero.(diag(V)))
        V += eps(eltype(sk)) * I
    end
    return sk, V
end
function cluster_negative_skewness(::NegativeSemiSkewness{<:Any, <:Any, Nothing, <:Any},
                                   prior::HighOrderPriorModel{<:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any,
                                                              <:AbstractMatrix,
                                                              <:AbstractMatrix,
                                                              <:MatrixProcessing},
                                   cluster::AbstractVector)
    idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
    sk = view(prior.ssk, cluster, idx)
    V = __coskewness(sk, prior.X, prior.sskmp)
    if all(iszero.(diag(V)))
        V += eps(eltype(sk)) * I
    end
    return sk, V
end
function cluster_risk_measure_factory(r::NegativeSemiSkewness; prior::AbstractPriorModel,
                                      cluster::AbstractVector, kwargs...)
    sk, V = cluster_negative_skewness(r, prior, cluster)
    return NegativeSemiSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V)
end

export NegativeSemiSkewness
