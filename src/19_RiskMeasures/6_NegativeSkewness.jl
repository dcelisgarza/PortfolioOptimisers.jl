const NegativeSkewnessAlgorithm = Union{<:SqrtRiskExpr, <:QuadRiskExpr}
struct NegativeSkewness{T1 <: RiskMeasureSettings, T2 <: AbstractMatrixProcessingEstimator,
                        T3 <: Union{Nothing, <:AbstractMatrix},
                        T4 <: Union{Nothing, <:AbstractMatrix},
                        T5 <: NegativeSkewnessAlgorithm} <: AbstractNegativeSkewRiskMeasure
    settings::T1
    mp::T2
    sk::T3
    V::T4
    alg::T5
end
function NegativeSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing(),
                          sk::Union{Nothing, <:AbstractMatrix} = nothing,
                          V::Union{Nothing, <:AbstractMatrix} = nothing,
                          alg::NegativeSkewnessAlgorithm = SqrtRiskExpr())
    sk_flag = isnothing(sk)
    V_flag = isnothing(V)
    if sk_flag || V_flag
        @smart_assert(sk_flag && V_flag,
                      "If either sk or V, is nothing, both must be nothing.")
    else
        @smart_assert(!isempty(sk))
        @smart_assert(!isempty(V))
        @smart_assert(size(sk, 1)^2 == size(sk, 2))
        issquare(V)
    end
    return NegativeSkewness{typeof(settings), typeof(mp), typeof(sk), typeof(V),
                            typeof(alg)}(settings, mp, sk, V, alg)
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:SqrtRiskExpr})(w::AbstractVector)
    return sqrt(dot(w, r.V, w))
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:QuadRiskExpr})(w::AbstractVector)
    return dot(w, r.V, w)
end
function factory(r::NegativeSkewness, prior::HighOrderPriorResult, args...; kwargs...)
    sk = nothing_scalar_array_factory(r.sk, prior.sk)
    V = nothing_scalar_array_factory(r.V, prior.V)
    return NegativeSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V, alg = r.alg)
end
function factory(r::NegativeSkewness, ::LowOrderPriorResult, args...; kwargs...)
    return r
end
function risk_measure_view(r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:Any}, ::Any,
                           args...)
    return r
end
function risk_measure_view(r::NegativeSkewness{<:Any, <:Any, <:AbstractMatrix,
                                               <:AbstractMatrix, <:Any}, i::AbstractVector,
                           X::AbstractMatrix)
    sk = r.sk
    idx = fourth_moment_index_factory(size(sk, 1), i)
    sk = view(r.sk, i, idx)
    V = __coskewness(sk, view(X, :, i), r.mp)
    return NegativeSkewness(; settings = r.settings, alg = r.alg, mp = r.mp, sk = sk, V = V)
end

export SqrtRiskExpr, QuadRiskExpr, NegativeSkewness
