struct TailGiniRange{T1 <: RiskMeasureSettings, T2 <: OrderedWeightsArrayFormulation,
                     T3 <: Real, T4 <: Real, T5 <: Integer, T6 <: Real, T7 <: Real,
                     T8 <: Integer} <: OrderedWeightsArrayRiskMeasure
    settings::T1
    formulation::T2
    alpha_i::T3
    alpha::T4
    a_sim::T5
    beta_i::T6
    beta::T7
    b_sim::T8
end
function TailGiniRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                       formulation::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
                       alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100,
                       beta_i = 0.0001, beta::Real = 0.05, b_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    @smart_assert(zero(beta) < beta_i < beta < one(beta))
    @smart_assert(b_sim > zero(b_sim))
    return TailGiniRange{typeof(settings), typeof(formulation), typeof(alpha_i),
                         typeof(alpha), typeof(a_sim), typeof(beta_i), typeof(beta),
                         typeof(b_sim)}(settings, formulation, alpha_i, alpha, a_sim,
                                        beta_i, beta, b_sim)
end
function (r::TailGiniRange)(x::AbstractVector)
    w = owa_tgrg(length(x); alpha_i = r.alpha_i, alpha = r.alpha, a_sim = r.a_sim,
                 beta_i = r.beta_i, beta = r.beta, b_sim = r.b_sim)
    return dot(w, sort!(x))
end

export TailGiniRange
