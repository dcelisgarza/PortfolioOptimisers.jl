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
struct GiniMeanDifference{T1 <: RiskMeasureSettings,
                          T2 <: OrderedWeightsArrayFormulation} <:
       OrderedWeightsArrayRiskMeasure
    settings::T1
    formulation::T2
end
function GiniMeanDifference(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                            formulation::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return GiniMeanDifference{typeof(settings), typeof(formulation)}(settings, formulation)
end
function (::GiniMeanDifference)(x::AbstractVector)
    w = owa_gmd(length(x))
    return dot(w, sort!(x))
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

export ExactOrderedWeightsArray, ApproxOrderedWeightsArray, OrderedWeightsArray,
       GiniMeanDifference, TailGini, TailGiniRange
