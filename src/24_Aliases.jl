# Literature acronym aliases — each is a `const` pointing at the canonical long-form type.
# These are exported so callers who know the acronym can discover and use it directly via
# autocomplete without having to look up the full spelling.  Each alias is unambiguous in
# CONTEXT.md; ambiguous abbreviations (e.g. MAD ↔ Mean vs Median)
# are intentionally omitted.

# ── Clustering / hierarchical optimisers ──────────────────────────────────────
"""
Alias for [`HierarchicalRiskParity`](@ref).
"""
const HRP = HierarchicalRiskParity
"""
Alias for [`HierarchicalEqualRiskContribution`](@ref).
"""
const HERC = HierarchicalEqualRiskContribution
"""
Alias for [`SchurComplementHierarchicalRiskParity`](@ref).
"""
const SCHRP = SchurComplementHierarchicalRiskParity

# ── Meta-optimisers ───────────────────────────────────────────────────────────
"""
Alias for [`NestedClustered`](@ref).
"""
const NCO = NestedClustered
"""
Alias for [`Stacking`](@ref).
"""
const STO = Stacking
"""
Alias for [`SubsetResampling`](@ref).
"""
const SSR = SubsetResampling

# ── JuMP-based optimisers ─────────────────────────────────────────────────────
"""
Alias for [`MeanRisk`](@ref).
"""
const MR = MeanRisk
"""
Alias for [`RiskBudgeting`](@ref).
"""
const RB = RiskBudgeting
"""
Alias for [`RelaxedRiskBudgeting`](@ref).
"""
const RRB = RelaxedRiskBudgeting
"""
Alias for [`FactorRiskContribution`](@ref).
"""
const FRC = FactorRiskContribution
"""
Alias for [`NearOptimalCentering`](@ref).
"""
const NOC = NearOptimalCentering

# ── Risk measures — Moment risk measure family ──────────────────────────────────────
"""
    FLM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing) -> LowOrderMoment

Alias for the first Lower Moment (FLM) risk measure.
"""
function FLM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             w::Option{<:ObsWeights} = nothing,
             mu::Option{<:Num_VecNum_VecScalar} = nothing)::LowOrderMoment
    return LowOrderMoment(; settings = settings, w = w, mu = mu, alg = FirstLowerMoment())
end
"""
    MAD(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing) -> LowOrderMoment

Alias for the Mean Absolute Deviation (MAD) risk measure.
"""
function MAD(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             w::Option{<:ObsWeights} = nothing,
             mu::Option{<:Num_VecNum_VecScalar} = nothing)::LowOrderMoment
    return LowOrderMoment(; settings = settings, w = w, mu = mu,
                          alg = MeanAbsoluteDeviation())
end
"""
    SCM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing,
          ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
          alg::SecondMomentFormulation = SquaredSOCRiskExpr()) -> LowOrderMoment

Alias for the Second Central Moment (SCM) risk measure. This can represent the scenario based variance or standard deviation.
"""
function SCM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             w::Option{<:ObsWeights} = nothing,
             mu::Option{<:Num_VecNum_VecScalar} = nothing,
             ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
             alg::SecondMomentFormulation = SquaredSOCRiskExpr())::LowOrderMoment
    return LowOrderMoment(; settings = settings, w = w, mu = mu,
                          alg = SecondMoment(; ve = ve, alg1 = FullMoment(), alg2 = alg))
end
"""
    SLM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing,
          ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
          alg::SecondMomentFormulation = SquaredSOCRiskExpr()) -> LowOrderMoment

Alias for the Second Lower Moment (SLM) risk measure. This can represent the scenario based semi-variance or semi-standard deviation.
"""
function SLM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             w::Option{<:ObsWeights} = nothing,
             mu::Option{<:Num_VecNum_VecScalar} = nothing,
             ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
             alg::SecondMomentFormulation = SquaredSOCRiskExpr())::LowOrderMoment
    return LowOrderMoment(; settings = settings, w = w, mu = mu,
                          alg = SecondMoment(; ve = ve, alg1 = SemiMoment(), alg2 = alg))
end
"""
    ECM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing, p::Integer = 2,
          ddof::Integer = 0) -> LowOrderMoment

Alias for the square root of the central even moment of order `2p`.
"""
function ECM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             w::Option{<:ObsWeights} = nothing,
             mu::Option{<:Num_VecNum_VecScalar} = nothing, p::Integer = 2,
             ddof::Integer = 0)::LowOrderMoment
    return LowOrderMoment(; settings = settings, w = w, mu = mu,
                          alg = EvenMoment(; p = p, ddof = ddof, alg = FullMoment()))
end
"""
    ELM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing, p::Integer = 2,
          ddof::Integer = 0) -> LowOrderMoment

Alias for the square root of the lower even moment of order `2p`.
"""
function ELM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             w::Option{<:ObsWeights} = nothing,
             mu::Option{<:Num_VecNum_VecScalar} = nothing, p::Integer = 2,
             ddof::Integer = 0)::LowOrderMoment
    return LowOrderMoment(; settings = settings, w = w, mu = mu,
                          alg = EvenMoment(; p = p, ddof = ddof, alg = SemiMoment()))
end
"""
    TLM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing) -> HighOrderMoment

Alias for the Third Lower Moment (TLM) risk measure.
"""
function TLM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             w::Option{<:ObsWeights} = nothing,
             mu::Option{<:Num_VecNum_VecScalar} = nothing)::HighOrderMoment
    return HighOrderMoment(; settings = settings, w = w, mu = mu, alg = ThirdLowerMoment())
end
"""
    SSK(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing,
          ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing)) -> HighOrderMoment

Alias for the Standardised Third Lower Moment (SSK) risk measure. This represents the scenario based semi-skewness of the return distribution.
"""
function SSK(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             w::Option{<:ObsWeights} = nothing,
             mu::Option{<:Num_VecNum_VecScalar} = nothing,
             ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing))::HighOrderMoment
    return HighOrderMoment(; settings = settings, w = w, mu = mu,
                           alg = StandardisedHighOrderMoment(; ve = ve,
                                                             alg = ThirdLowerMoment()))
end
"""
    FTCM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing) -> HighOrderMoment

Alias for the Fourth Central Moment (FTCM) risk measure.
"""
function FTCM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
              w::Option{<:ObsWeights} = nothing,
              mu::Option{<:Num_VecNum_VecScalar} = nothing)::HighOrderMoment
    return HighOrderMoment(; settings = settings, w = w, mu = mu,
                           alg = FourthMoment(; alg = FullMoment()))
end
"""
    FTLM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing) -> HighOrderMoment

Alias for the Fourth Lower Moment (FTLM) risk measure.
"""
function FTLM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
              w::Option{<:ObsWeights} = nothing,
              mu::Option{<:Num_VecNum_VecScalar} = nothing)::HighOrderMoment
    return HighOrderMoment(; settings = settings, w = w, mu = mu,
                           alg = FourthMoment(; alg = SemiMoment()))
end
"""
    KT(; settings::RiskMeasureSettings = RiskMeasureSettings(),
         w::Option{<:ObsWeights} = nothing,
         mu::Option{<:Num_VecNum_VecScalar} = nothing,
         ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing)) -> HighOrderMoment

Alias for the Standardised Fourth Central Moment (KT) risk measure. This represents the scenario based kurtosis of the return distribution.
"""
function KT(; settings::RiskMeasureSettings = RiskMeasureSettings(),
            w::Option{<:ObsWeights} = nothing, mu::Option{<:Num_VecNum_VecScalar} = nothing,
            ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing))::HighOrderMoment
    return HighOrderMoment(; settings = settings, w = w, mu = mu,
                           alg = StandardisedHighOrderMoment(; ve = ve,
                                                             alg = FourthMoment(;
                                                                                alg = FullMoment())))
end
"""
    SKT(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing,
          ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing)) -> HighOrderMoment

Alias for the Standardised Fourth Lower Moment (SKT) risk measure. This represents the scenario based semi-kurtosis of the return distribution.
"""
function SKT(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             w::Option{<:ObsWeights} = nothing,
             mu::Option{<:Num_VecNum_VecScalar} = nothing,
             ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing))::HighOrderMoment
    return HighOrderMoment(; settings = settings, w = w, mu = mu,
                           alg = StandardisedHighOrderMoment(; ve = ve,
                                                             alg = FourthMoment(;
                                                                                alg = SemiMoment())))
end

# ── Risk measures — Value-at-Risk family ──────────────────────────────────────
"""
Alias for [`ValueatRisk`](@ref).
"""
const VaR = ValueatRisk
"""
Alias for [`ConditionalValueatRisk`](@ref) (CVaR / Expected Shortfall).
"""
const CVaR = ConditionalValueatRisk
"""
Alias for [`EntropicValueatRisk`](@ref).
"""
const EVaR = EntropicValueatRisk
"""
Alias for [`RelativisticValueatRisk`](@ref).
"""
const RVaR = RelativisticValueatRisk
"""
Alias for [`PowerNormValueatRisk`](@ref).
"""
const PNVaR = PowerNormValueatRisk

# ── Risk measures — Drawdown-at-Risk family ───────────────────────────────────
"""
Alias for [`DrawdownatRisk`](@ref).
"""
const DaR = DrawdownatRisk
"""
Alias for [`ConditionalDrawdownatRisk`](@ref).
"""
const CDaR = ConditionalDrawdownatRisk
"""
Alias for [`EntropicDrawdownatRisk`](@ref).
"""
const EDaR = EntropicDrawdownatRisk
"""
Alias for [`RelativisticDrawdownatRisk`](@ref).
"""
const RDaR = RelativisticDrawdownatRisk
"""
Alias for [`PowerNormDrawdownatRisk`](@ref).
"""
const PNDaR = PowerNormDrawdownatRisk
"""
Alias for [`RelativeDrawdownatRisk`](@ref).
"""
const R_DaR = RelativeDrawdownatRisk
"""
Alias for [`RelativeConditionalDrawdownatRisk`](@ref).
"""
const R_CDaR = RelativeConditionalDrawdownatRisk
"""
Alias for [`RelativeEntropicDrawdownatRisk`](@ref).
"""
const R_EDaR = RelativeEntropicDrawdownatRisk
"""
Alias for [`RelativeRelativisticDrawdownatRisk`](@ref).
"""
const R_RDaR = RelativeRelativisticDrawdownatRisk
"""
Alias for [`RelativePowerNormDrawdownatRisk`](@ref).
"""
const PNDaR_R = RelativePowerNormDrawdownatRisk

export HRP, HERC, SCHRP, NCO, STO, SSR, MR, RB, RRB, FRC, NOC, VaR, CVaR, EVaR, RVaR, PNVaR,
       DaR, CDaR, EDaR, RDaR, PNDaR, R_DaR, R_CDaR, R_EDaR, R_RDaR, PNDaR_R, FLM, MAD, SCM,
       SLM, ECM, ELM, TLM, SSK, FTCM, FTLM, KT, SKT
