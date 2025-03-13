struct NegativeQuadraticSemiSkewness{T1 <: RiskMeasureSettings, T2 <: MatrixProcessing,
                                     T3 <: Union{Nothing, <:AbstractMatrix},
                                     T4 <: Union{Nothing, <:AbstractMatrix}} <:
       SkewRiskMeasure
    settings::T1
    mp::T2
    sk::T3
    V::T4
end
function NegativeQuadraticSemiSkewness(;
                                       settings::RiskMeasureSettings = RiskMeasureSettings(),
                                       mp::MatrixProcessing = NonPositiveDefiniteMatrixProcessing(),
                                       sk::Union{Nothing, <:AbstractMatrix} = nothing,
                                       V::Union{Nothing, <:AbstractMatrix} = nothing)
    csk_invalid = isnothing(sk) || isempty(sk)
    v_invalid = isnothing(V) || isempty(V)
    if any((csk_invalid, v_invalid))
        @smart_assert(all((csk_invalid, v_invalid)),
                      "If either sk or V, is nothing or empty, both must be nothing or empty.")
    else
        @smart_assert(size(sk, 1)^2 == size(sk, 2))
        issquare(V)
    end
    if !isnothing(sk) && !isempty(sk)
        @smart_assert(size(sk, 1)^2 == size(sk, 2))
    end
    if !isnothing(V) && !isempty(V)
        @smart_assert(size(V, 1) == size(V, 2))
    end
    return NegativeQuadraticSemiSkewness{typeof(settings), typeof(mp), typeof(sk),
                                         typeof(V)}(settings, mp, sk, V)
end
function (r::NegativeQuadraticSemiSkewness)(w::AbstractVector)
    return dot(w, r.V, w)
end
function _risk_measure_factory(r::NegativeQuadraticSemiSkewness, prior::HighOrderPriorModel)
    sk = risk_measure_nothing_matrix_factory(r.sk, prior.ssk)
    V = risk_measure_nothing_matrix_factory(r.V, prior.SV)
    return NegativeQuadraticSemiSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V)
end
function _risk_measure_factory(r::NegativeQuadraticSemiSkewness,
                               ::LowOrderAbstractPriorModel)
    sk = risk_measure_nothing_matrix_factory(r.sk, nothing)
    V = risk_measure_nothing_matrix_factory(r.V, nothing)
    return NegativeQuadraticSemiSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V)
end
function risk_measure_factory(r::NegativeQuadraticSemiSkewness; prior::AbstractPriorModel,
                              kwargs...)
    return _risk_measure_factory(r, prior)
end
function cluster_negative_skewness(::NegativeQuadraticSemiSkewness{<:Any, <:Any, Nothing,
                                                                   <:Any}, ::Nothing,
                                   prior::AbstractPriorModel, cluster::AbstractVector)
    throw(ArgumentError("Neither the risk measure, nor the prior have the required data."))
end
function cluster_negative_skewness(skew_rm::NegativeQuadraticSemiSkewness{<:Any, <:Any,
                                                                          <:AbstractMatrix,
                                                                          <:Any},
                                   prior::AbstractPriorModel, cluster::AbstractVector)
    if !isempty(skew_rm.sk)
        idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
        sk = view(skew_rm.sk, cluster, idx)
        V = __coskewness(sk, prior.X, skew_rm.mp)
        if all(iszero.(diag(V)))
            V += eps(eltype(sk)) * I
        end
    else
        throw(ArgumentError("Neither the risk measure, nor the prior have the required data."))
    end
    return sk, V
end
function cluster_negative_skewness(skew_rm::NegativeQuadraticSemiSkewness{<:Any, <:Any,
                                                                          <:AbstractMatrix,
                                                                          <:Any},
                                   prior::HighOrderPriorModel{<:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any,
                                                              <:AbstractMatrix,
                                                              <:AbstractMatrix,
                                                              <:MatrixProcessing},
                                   cluster::AbstractVector)
    if !isempty(skew_rm.sk)
        idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
        sk = view(skew_rm.sk, cluster, idx)
        V = __coskewness(sk, prior.X, skew_rm.mp)
        if all(iszero.(diag(V)))
            V += eps(eltype(sk)) * I
        end
    elseif !isempty(prior.ssk)
        idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
        sk = view(prior.ssk, cluster, idx)
        V = __coskewness(sk, prior.X, prior.sskmp)
        if all(iszero.(diag(V)))
            V += eps(eltype(sk)) * I
        end
    else
        throw(ArgumentError("Neither the risk measure, nor the prior have the required data."))
    end
    return sk, V
end
function cluster_negative_skewness(::NegativeQuadraticSemiSkewness{<:Any, <:Any, Nothing,
                                                                   <:Any},
                                   prior::HighOrderPriorModel{<:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any,
                                                              <:AbstractMatrix,
                                                              <:AbstractMatrix,
                                                              <:MatrixProcessing},
                                   cluster::AbstractVector)
    if !isempty(prior.ssk)
        idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
        sk = view(prior.ssk, cluster, idx)
        V = __coskewness(sk, prior.X, prior.sskmp)
        if all(iszero.(diag(V)))
            V += eps(eltype(sk)) * I
        end
    else
        throw(ArgumentError("Neither the risk measure, nor the prior have the required data."))
    end
    return sk, V
end
function cluster_risk_measure_factory(r::NegativeQuadraticSemiSkewness;
                                      prior::AbstractPriorModel, cluster::AbstractVector,
                                      kwargs...)
    sk, V = cluster_negative_skewness(r, prior, cluster)
    return NegativeQuadraticSemiSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V)
end

export NegativeQuadraticSemiSkewness
