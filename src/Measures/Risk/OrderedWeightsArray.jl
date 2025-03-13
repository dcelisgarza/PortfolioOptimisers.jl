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
    return OrderedWeightsArray{typeof(settings), typeof(formulation), typeof(w)}(settings,
                                                                                 formulation,
                                                                                 w)
end
function (r::OrderedWeightsArray)(x::AbstractVector)
    w = isnothing(r.w) ? owa_gmd(length(x)) : r.w
    return dot(w, sort!(x))
end

export OrderedWeightsArray
