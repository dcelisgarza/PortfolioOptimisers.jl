abstract type BrownianDistanceVarianceFormulation <: AbstractAlgorithm end
struct NormOneConeBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
struct IneqBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
struct BrownianDistanceVariance{T1 <: RiskMeasureSettings,
                                T2 <: Union{<:RSOCRiskExpr, <:QuadRiskExpr},
                                T3 <: BrownianDistanceVarianceFormulation} <: RiskMeasure
    settings::T1
    formulation::T2
    cformulation::T3
end
function BrownianDistanceVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  formulation::Union{<:RSOCRiskExpr, <:QuadRiskExpr} = RSOCRiskExpr(),
                                  cformulation::BrownianDistanceVarianceFormulation = IneqBrownianDistanceVariance())
    return BrownianDistanceVariance{typeof(settings), typeof(formulation),
                                    typeof(cformulation)}(settings, formulation,
                                                          cformulation)
end
function (::BrownianDistanceVariance)(x::AbstractVector)
    T = length(x)
    iT2 = inv(T^2)
    D = Matrix{eltype(x)}(undef, T, T)
    D .= x
    D .-= transpose(x)
    D .= abs.(D)
    val = iT2 * (dot(D, D) + iT2 * sum(D)^2)
    return val
end

export NormOneConeBrownianDistanceVariance, IneqBrownianDistanceVariance,
       BrownianDistanceVariance
