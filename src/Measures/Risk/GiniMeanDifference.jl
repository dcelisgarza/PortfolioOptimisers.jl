struct GiniMeanDifference{T1 <: RiskMeasureSettings,
                          T2 <: OrderedWeightsArrayFormulation} <:
       OrderedWeightsArrayRiskMeasure
    settings::T1
    formulation:::T2
end
function GiniMeanDifference(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                            formulation::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return GiniMeanDifference{typeof(settings), typeof(formulation)}(settings, formulation)
end
function (::GiniMeanDifference)(x::AbstractVector)
    w = owa_gmd(length(x))
    return dot(w, sort!(x))
end

export GiniMeanDifference
