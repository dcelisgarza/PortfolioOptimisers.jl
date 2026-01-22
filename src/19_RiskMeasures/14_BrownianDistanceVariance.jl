abstract type BrownianDistanceVarianceFormulation <: AbstractAlgorithm end
struct NormOneConeBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
struct IneqBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
const BDVarRkFormulations = Union{<:RSOCRiskExpr, <:QuadRiskExpr}
struct BrownianDistanceVariance{T1, T2, T3} <: RiskMeasure
    settings::T1
    alg1::T2
    alg2::T3
    function BrownianDistanceVariance(settings::RiskMeasureSettings,
                                      alg1::BDVarRkFormulations,
                                      alg2::BrownianDistanceVarianceFormulation)
        return new{typeof(settings), typeof(alg1), typeof(alg2)}(settings, alg1, alg2)
    end
end
function BrownianDistanceVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  alg1::BDVarRkFormulations = QuadRiskExpr(),
                                  alg2::BrownianDistanceVarianceFormulation = NormOneConeBrownianDistanceVariance())
    return BrownianDistanceVariance(settings, alg1, alg2)
end
function (::BrownianDistanceVariance)(x::VecNum)
    T = length(x)
    iT2 = inv(T^2)
    D = Matrix{eltype(x)}(undef, T, T)
    D .= x
    D .-= transpose(x)
    D .= abs.(D)
    val = iT2 * (LinearAlgebra.dot(D, D) + iT2 * sum(D)^2)
    return val
end

export NormOneConeBrownianDistanceVariance, IneqBrownianDistanceVariance,
       BrownianDistanceVariance
