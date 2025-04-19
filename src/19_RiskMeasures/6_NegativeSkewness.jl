abstract type AbstractNegativeSkewnessAlgorithm <: AbstractAlgorithm end
struct LinearNegativeSkewness <: AbstractNegativeSkewnessAlgorithm end
struct QuadraticNegativeSkewness <: AbstractNegativeSkewnessAlgorithm end
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
function risk_measure_factory(r::NegativeSkewness, prior::HighOrderPriorResult, args...;
                              kwargs...)
    sk = risk_measure_nothing_real_array_factory(r.sk, prior.sk)
    V = risk_measure_nothing_real_array_factory(r.V, prior.V)
    return NegativeSkewness(; settings = r.settings, alg = r.alg, mp = r.mp, sk = sk, V = V)
end
function risk_measure_factory(r::NegativeSkewness, ::AbstractLowOrderPriorResult, args...;
                              kwargs...)
    sk = risk_measure_nothing_real_array_factory(r.sk, nothing)
    V = risk_measure_nothing_real_array_factory(r.V, nothing)
    return NegativeSkewness(; settings = r.settings, alg = r.alg, mp = r.mp, sk = sk, V = V)
end
function risk_measure_view(::NegativeSkewness{<:Any, <:Any, <:Any, Nothing, <:Any},
                           prior::Union{Nothing, <:AbstractLowOrderPriorResult,
                                        <:HighOrderPriorResult{<:Any, <:Any, Nothing, <:Any,
                                                               <:Any}}, i::AbstractVector,
                           args...; kwargs...)
    throw(ArgumentError("Neither the risk measure, nor the prior have the required data."))
end
function risk_measure_view(r::NegativeSkewness{<:Any, <:Any, <:Any, <:AbstractMatrix,
                                               <:Any}, prior::AbstractPriorResult,
                           i::AbstractVector, args...; kwargs...)
    idx = fourth_moment_index_factory(size(prior.X, 2), i)
    sk = view(r.sk, i, idx)
    V = __coskewness(sk, view(prior.X, :, i), r.mp)
    if all(iszero, diag(V))
        V[diagind(V)] = I(size(V, 1))
    end
    return NegativeSkewness(; settings = r.settings, alg = r.alg, mp = r.mp, sk = sk, V = V)
end
function risk_measure_view(r::NegativeSkewness{<:Any, <:Any, <:Any, Nothing, <:Any},
                           prior::HighOrderPriorResult{<:Any, <:Any, <:AbstractMatrix,
                                                       <:AbstractMatrix,
                                                       <:AbstractMatrixProcessingEstimator},
                           i::AbstractVector, args...; kwargs...)
    idx = fourth_moment_index_factory(size(prior.X, 2), i)
    sk = view(prior.sk, i, idx)
    V = __coskewness(sk, view(prior.X, :, i), prior.skmp)
    if all(iszero, diag(V))
        V[diagind(V)] = I(size(V, 1))
    end
    return NegativeSkewness(; settings = r.settings, alg = r.alg, mp = r.mp, sk = sk, V = V)
end

export LinearNegativeSkewness, QuadraticNegativeSkewness, NegativeSkewness
