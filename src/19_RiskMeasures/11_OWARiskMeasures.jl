abstract type OrderedWeightsArrayFormulation <: AbstractAlgorithm end
struct ExactOrderedWeightsArray <: OrderedWeightsArrayFormulation end
struct ApproxOrderedWeightsArray{T1 <: AbstractVector{<:Real}} <:
       OrderedWeightsArrayFormulation
    p::T1
end
function ApproxOrderedWeightsArray(; p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50])
    @smart_assert(!isempty(p))
    @smart_assert(all(p .> zero(eltype(p))))
    return ApproxOrderedWeightsArray{typeof(p)}(p)
end
struct OrderedWeightsArray{T1 <: RiskMeasureSettings, T2 <: OrderedWeightsArrayFormulation,
                           T3 <: Union{Nothing, <:AbstractVector}} <:
       OrderedWeightsArrayRiskMeasure
    settings::T1
    formulation::T2
    w::T3
end
function OrderedWeightsArray(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             formulation::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
                             w::Union{Nothing, <:AbstractVector} = nothing)
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return OrderedWeightsArray{typeof(settings), typeof(formulation), typeof(w)}(settings,
                                                                                 formulation,
                                                                                 w)
end
function (r::OrderedWeightsArray)(x::AbstractVector)
    w = isnothing(r.w) ? owa_gmd(length(x)) : r.w
    return dot(w, sort!(x))
end

export ExactOrderedWeightsArray, ApproxOrderedWeightsArray, OrderedWeightsArray
