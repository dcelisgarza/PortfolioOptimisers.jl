abstract type OrderedWeightsArrayFormulation <: AbstractAlgorithm end
struct ExactOrderedWeightsArray <: OrderedWeightsArrayFormulation end
struct ApproxOrderedWeightsArray{T1} <: OrderedWeightsArrayFormulation
    p::T1
    function ApproxOrderedWeightsArray(p::AbstractVector{<:Real})
        @argcheck(!isempty(p))
        @argcheck(all(x -> x > zero(x), p))
        return new{typeof(p)}(p)
    end
end
function ApproxOrderedWeightsArray(; p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50])
    return ApproxOrderedWeightsArray(p)
end
struct OrderedWeightsArray{T1, T2, T3} <: OrderedWeightsArrayRiskMeasure
    settings::T1
    w::T2
    alg::T3
    function OrderedWeightsArray(settings::RiskMeasureSettings,
                                 w::Union{Nothing, <:AbstractVector},
                                 alg::OrderedWeightsArrayFormulation)
        if isa(w, AbstractVector)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(alg)}(settings, w, alg)
    end
end
function OrderedWeightsArray(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             w::Union{Nothing, <:AbstractVector} = nothing,
                             alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(settings, w, alg)
end
function (r::OrderedWeightsArray)(x::AbstractVector)
    w = isnothing(r.w) ? owa_gmd(length(x)) : r.w
    return dot(w, sort!(x))
end
struct OrderedWeightsArrayRange{T1, T2, T3, T4} <: OrderedWeightsArrayRiskMeasure
    settings::T1
    w1::T2
    w2::T3
    alg::T4
    function OrderedWeightsArrayRange(settings::RiskMeasureSettings,
                                      w1::Union{Nothing, <:AbstractVector},
                                      w2::Union{Nothing, <:AbstractVector},
                                      alg::OrderedWeightsArrayFormulation, rev::Bool)
        w1_flag = !isnothing(w1)
        w2_flag = !isnothing(w2)
        if w1_flag
            @argcheck(!isempty(w1))
        end
        if w2_flag
            @argcheck(!isempty(w2))
            if !rev
                w2 = reverse(w2)
            end
        end
        if w1_flag && w2_flag
            @argcheck(length(w1) == length(w2))
        end
        return new{typeof(settings), typeof(w1), typeof(w2), typeof(alg)}(settings, w1, w2,
                                                                          alg)
    end
end
function OrderedWeightsArrayRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  w1::Union{Nothing, <:AbstractVector} = nothing,
                                  w2::Union{Nothing, <:AbstractVector} = nothing,
                                  alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
                                  rev::Bool = false)
    return OrderedWeightsArrayRange(settings, w1, w2, alg, rev)
end
function (r::OrderedWeightsArrayRange)(x::AbstractVector)
    w1 = isnothing(r.w1) ? owa_tg(length(x)) : r.w1
    w2 = isnothing(r.w2) ? reverse(w1) : r.w2
    w = w1 - w2
    return dot(w, sort!(x))
end

export ExactOrderedWeightsArray, ApproxOrderedWeightsArray, OrderedWeightsArray,
       OrderedWeightsArrayRange
