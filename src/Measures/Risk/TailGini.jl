struct TailGini{T1 <: RiskMeasureSettings, T2 <: OrderedWeightsArrayFormulation, T3 <: Real,
                T4 <: Real, T5 <: Integer} <: OrderedWeightsArrayRiskMeasure
    settings::T1
    formulation::T2
    alpha_i::T3
    alpha::T4
    a_sim::T5
end
function TailGini(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  formulation::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
                  alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    return TailGini{typeof(settings), typeof(formulation), typeof(alpha_i), typeof(alpha),
                    typeof(a_sim)}(settings, formulation, alpha_i, alpha, a_sim)
end
function (r::TailGini)(x::AbstractVector)
    w = owa_tg(length(x); alpha_i = r.alpha_i, alpha = r.alpha, a_sim = r.a_sim)
    return dot(w, sort!(x))
end

export TailGini
