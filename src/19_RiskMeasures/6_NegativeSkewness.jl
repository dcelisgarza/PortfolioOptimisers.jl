abstract type AbstractNegativeSkewnessAlgorithm <: AbstractAlgorithm end
struct LinearNegativeSkewness{T1 <: AbstractMomentAlgorithm} <:
       AbstractNegativeSkewnessAlgorithm
    alg::T1
end
function LinearNegativeSkewness(; alg::AbstractMomentAlgorithm = Full())
    return LinearNegativeSkewness{typeof(alg)}(alg)
end
struct QuadraticNegativeSkewness{T1 <: AbstractMomentAlgorithm} <:
       AbstractNegativeSkewnessAlgorithm
    alg::T1
end
function QuadraticNegativeSkewness(; alg::AbstractMomentAlgorithm = Full())
    return QuadraticNegativeSkewness{typeof(alg)}(alg)
end
struct NegativeSkewness{T1 <: RiskMeasureSettings, T2 <: AbstractNegativeSkewnessAlgorithm,
                        T3 <: AbstractMatrixProcessingEstimator,
                        T4 <: Union{Nothing, <:AbstractMatrix},
                        T5 <: Union{Nothing, <:AbstractMatrix}} <:
       AbstractNegativeSkewRiskMeasure
    settings::T1
    alg::T2
    mp::T3
    sk::T4
    V::T5
end
function NegativeSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          alg::AbstractNegativeSkewnessAlgorithm = LinearNegativeSkewness(),
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
    return NegativeSkewness{typeof(settings), typeof(alg), typeof(mp), typeof(sk),
                            typeof(V)}(settings, alg, mp, sk, V)
end
function (r::NegativeSkewness{<:Any, <:LinearNegativeSkewness, <:Any, <:Any, <:Any})(w::AbstractVector)
    return sqrt(dot(w, r.V, w))
end
function (r::NegativeSkewness{<:Any, <:QuadraticNegativeSkewness, <:Any, <:Any, <:Any})(w::AbstractVector)
    return dot(w, r.V, w)
end

export LinearNegativeSkewness, QuadraticNegativeSkewness, NegativeSkewness
