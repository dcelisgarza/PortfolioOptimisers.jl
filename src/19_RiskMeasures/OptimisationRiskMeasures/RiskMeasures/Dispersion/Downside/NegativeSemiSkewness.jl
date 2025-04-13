struct NegativeSemiSkewness{T1 <: RiskMeasureSettings,
                            T2 <: AbstractMatrixProcessingEstimator,
                            T3 <: Union{Nothing, <:AbstractMatrix},
                            T4 <: Union{Nothing, <:AbstractMatrix}} <: SkewRiskMeasure
    settings::T1
    mp::T2
    sk::T3
    V::T4
end
function NegativeSemiSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                              mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing(),
                              sk::Union{Nothing, <:AbstractMatrix} = nothing,
                              V::Union{Nothing, <:AbstractMatrix} = nothing)
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
    return NegativeSemiSkewness{typeof(settings), typeof(mp), typeof(sk), typeof(V)}(settings,
                                                                                     mp, sk,
                                                                                     V)
end
function (r::NegativeSemiSkewness)(w::AbstractVector)
    return sqrt(dot(w, r.V, w))
end

export NegativeSemiSkewness
