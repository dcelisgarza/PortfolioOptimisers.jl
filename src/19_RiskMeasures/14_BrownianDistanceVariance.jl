abstract type BrownianDistanceVarianceFormulation <: AbstractAlgorithm end
struct NormOneConeBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
struct IneqBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
const BDVarRkFormulations = Union{<:RSOCRiskExpr, <:QuadRiskExpr}
struct BrownianDistanceVariance{T1, T2, T3} <: RiskMeasure
    settings::T1
    alg::T2
    algc::T3
    function BrownianDistanceVariance(settings::RiskMeasureSettings,
                                      alg::BDVarRkFormulations,
                                      algc::BrownianDistanceVarianceFormulation)
        return new{typeof(settings), typeof(alg), typeof(algc)}(settings, alg, algc)
    end
end
function BrownianDistanceVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  alg::BDVarRkFormulations = QuadRiskExpr(),
                                  algc::BrownianDistanceVarianceFormulation = NormOneConeBrownianDistanceVariance())
    return BrownianDistanceVariance(settings, alg, algc)
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
