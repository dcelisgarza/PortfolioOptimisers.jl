struct NegativeSkewness{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    mp::T2
    sk::T3
    V::T4
    alg::T5
    function NegativeSkewness(settings::RiskMeasureSettings,
                              mp::AbstractMatrixProcessingEstimator,
                              sk::Union{Nothing, <:AbstractMatrix},
                              V::Union{Nothing, <:AbstractMatrix}, alg::QuadSqrtRiskExpr)
        sk_flag = isnothing(sk)
        V_flag = isnothing(V)
        if sk_flag || V_flag
            @argcheck(sk_flag && V_flag,
                      "If either sk or V, is nothing, both must be nothing.")
        else
            @argcheck(!isempty(sk))
            @argcheck(!isempty(V))
            @argcheck(size(sk, 1)^2 == size(sk, 2))
            assert_matrix_issquare(V)
        end
        return new{typeof(settings), typeof(mp), typeof(sk), typeof(V), typeof(alg)}(settings,
                                                                                     mp, sk,
                                                                                     V, alg)
    end
end
function NegativeSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing(),
                          sk::Union{Nothing, <:AbstractMatrix} = nothing,
                          V::Union{Nothing, <:AbstractMatrix} = nothing,
                          alg::QuadSqrtRiskExpr = SqrtRiskExpr())
    return NegativeSkewness(settings, mp, sk, V, alg)
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:SqrtRiskExpr})(w::AbstractVector)
    return sqrt(dot(w, r.V, w))
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:QuadRiskExpr})(w::AbstractVector)
    return dot(w, r.V, w)
end
function factory(r::NegativeSkewness, prior::HighOrderPrior, args...; kwargs...)
    sk = nothing_scalar_array_factory(r.sk, prior.sk)
    V = nothing_scalar_array_factory(r.V, prior.V)
    return NegativeSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V, alg = r.alg)
end
function factory(r::NegativeSkewness, ::LowOrderPrior, args...; kwargs...)
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

export NegativeSkewness
