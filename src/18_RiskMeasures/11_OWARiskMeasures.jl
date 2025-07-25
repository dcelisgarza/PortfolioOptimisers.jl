abstract type OrderedWeightsArrayFormulation <: AbstractAlgorithm end
struct ExactOrderedWeightsArray <: OrderedWeightsArrayFormulation end
struct ApproxOrderedWeightsArray{T1 <: AbstractVector{<:Real}} <:
       OrderedWeightsArrayFormulation
    p::T1
end
function ApproxOrderedWeightsArray(; p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50])
    @smart_assert(!isempty(p))
    @smart_assert(all(x -> x > zero(x), p))
    return ApproxOrderedWeightsArray{typeof(p)}(p)
end
struct OrderedWeightsArray{T1 <: RiskMeasureSettings,
                           T2 <: Union{Nothing, <:AbstractVector},
                           T3 <: OrderedWeightsArrayFormulation} <:
       OrderedWeightsArrayRiskMeasure
    settings::T1
    w::T2
    alg::T3
end
function OrderedWeightsArray(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             w::Union{Nothing, <:AbstractVector} = nothing,
                             alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return OrderedWeightsArray{typeof(settings), typeof(w), typeof(alg)}(settings, w, alg)
end
function (r::OrderedWeightsArray)(x::AbstractVector)
    w = isnothing(r.w) ? owa_gmd(length(x)) : r.w
    return dot(w, sort!(x))
end
struct OrderedWeightsArrayRange{T1 <: RiskMeasureSettings,
                                T2 <: Union{Nothing, <:AbstractVector},
                                T3 <: Union{Nothing, <:AbstractVector},
                                T4 <: OrderedWeightsArrayFormulation} <:
       OrderedWeightsArrayRiskMeasure
    settings::T1
    w1::T2
    w2::T3
    alg::T4
end
function OrderedWeightsArrayRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  w1::Union{Nothing, <:AbstractVector} = nothing,
                                  w2::Union{Nothing, <:AbstractVector} = nothing,
                                  alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
                                  rev::Bool = false)
    w1_flag = !isnothing(w1)
    w2_flag = !isnothing(w2)
    if w1_flag
        @smart_assert(!isempty(w1))
    end
    if w2_flag
        @smart_assert(!isempty(w2))
        if !rev
            w2 = reverse(w2)
        end
    end
    if w1_flag && w2_flag
        @smart_assert(length(w1) == length(w2))
    end
    return OrderedWeightsArrayRange{typeof(settings), typeof(w1), typeof(w2), typeof(alg)}(settings,
                                                                                           w1,
                                                                                           w2,
                                                                                           alg)
end
function (r::OrderedWeightsArrayRange)(x::AbstractVector)
    w1 = isnothing(r.w1) ? owa_tg(length(x)) : r.w1
    w2 = isnothing(r.w2) ? reverse(w1) : r.w2
    w = w1 - w2
    return dot(w, sort!(x))
end

export ExactOrderedWeightsArray, ApproxOrderedWeightsArray, OrderedWeightsArray,
       OrderedWeightsArrayRange
