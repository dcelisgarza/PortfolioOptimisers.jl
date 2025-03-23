struct NegativeSkewness{T1 <: RiskMeasureSettings, T2 <: MatrixProcessing,
                        T3 <: Union{Nothing, <:AbstractMatrix},
                        T4 <: Union{Nothing, <:AbstractMatrix}} <: SkewRiskMeasure
    settings::T1
    mp::T2
    sk::T3
    V::T4
end
function NegativeSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
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
    return NegativeSkewness{typeof(settings), typeof(mp), typeof(sk), typeof(V)}(settings,
                                                                                 mp, sk, V)
end
function (r::NegativeSkewness)(w::AbstractVector)
    return sqrt(dot(w, r.V, w))
end

export NegativeSkewness
