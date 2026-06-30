# Literature acronym aliases — each is a `const` pointing at the canonical long-form type.
# These are exported so callers who know the acronym can discover and use it directly via
# autocomplete without having to look up the full spelling.  Each alias is unambiguous in
# CONTEXT.md; ambiguous abbreviations (e.g. MAD ↔ Mean vs Median)
# are intentionally omitted.

# ── Clustering / hierarchical optimisers ──────────────────────────────────────
"""
    HRP

Alias for [`HierarchicalRiskParity`](@ref).
"""
const HRP = HierarchicalRiskParity
"""
    HERC

Alias for [`HierarchicalEqualRiskContribution`](@ref).
"""
const HERC = HierarchicalEqualRiskContribution
"""
    SCHRP

Alias for [`SchurComplementHierarchicalRiskParity`](@ref).
"""
const SCHRP = SchurComplementHierarchicalRiskParity

# ── Meta-optimisers ───────────────────────────────────────────────────────────
"""
    NCO

Alias for [`NestedClustered`](@ref).
"""
const NCO = NestedClustered
"""
    STO

Alias for [`Stacking`](@ref).
"""
const STO = Stacking
"""
    SSR

Alias for [`SubsetResampling`](@ref).
"""
const SSR = SubsetResampling

# ── JuMP-based optimisers ─────────────────────────────────────────────────────
"""
    MR

Alias for [`MeanRisk`](@ref).
"""
const MR = MeanRisk
"""
    RB

Alias for [`RiskBudgeting`](@ref).
"""
const RB = RiskBudgeting
"""
    RRB

Alias for [`RelaxedRiskBudgeting`](@ref).
"""
const RRB = RelaxedRiskBudgeting
"""
    FRC

Alias for [`FactorRiskContribution`](@ref).
"""
const FRC = FactorRiskContribution
"""
Alias for [`NearOptimalCentering`](@ref).
"""
const NOC = NearOptimalCentering

# ── Risk measures — Covariance-based ──────────────────────────────────────
"""
    SD

Alias for [`StandardDeviation`](@ref).
"""
const SD = StandardDeviation

"""
    UcVariance

Alias for [`UncertaintySetVariance`](@ref).
"""
const UcVariance = UncertaintySetVariance

# ── Risk measures — Moment risk measure family ──────────────────────────────────────
"""
    FLM(; settings::RiskMeasureSettings = RiskMeasureSettings(),
          w::Option{<:ObsWeights} = nothing,
          mu::Option{<:Num_VecNum_VecScalar} = nothing) -> LowOrderMoment

Alias for the first Lower Moment (FLM) risk measure [`LowOrderMoment`](@ref) + [`FirstLowerMoment`](@ref).
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

Alias for the Mean Absolute Deviation (MAD) risk measure [`LowOrderMoment`](@ref) + [`MeanAbsoluteDeviation`](@ref).
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

Alias for the Second Central Moment (SCM) risk measure [`LowOrderMoment`](@ref) + [`SecondMoment`](@ref) + [`FullMoment`](@ref).
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

Alias for the Second Lower Moment (SLM) risk measure [`LowOrderMoment`](@ref) + [`SecondMoment`](@ref) + [`SemiMoment`](@ref). This can represent the scenario based semi-variance or semi-standard deviation.
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

Alias for the square root of the central even moment of order `2p` [`LowOrderMoment`](@ref) + [`EvenMoment`](@ref) + [`FullMoment`](@ref).
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

Alias for the square root of the lower even moment of order `2p` [`LowOrderMoment`](@ref) + [`EvenMoment`](@ref) + [`SemiMoment`](@ref).
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

Alias for the Third Lower Moment (TLM) risk measure [`HighOrderMoment`](@ref) + [`ThirdLowerMoment`](@ref).
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

Alias for the Standardised Third Lower Moment (SSK) risk measure [`HighOrderMoment`](@ref) + [`StandardisedHighOrderMoment`](@ref) + [`ThirdLowerMoment`](@ref). This represents the scenario based semi-skewness of the return distribution.
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

Alias for the Fourth Central Moment (FTCM) risk measure [`HighOrderMoment`](@ref) + [`FourthMoment`](@ref) + [`FullMoment`](@ref).
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

Alias for the Fourth Lower Moment (FTLM) risk measure [`HighOrderMoment`](@ref) + [`FourthMoment`](@ref) + [`SemiMoment`](@ref).
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

Alias for the Standardised Fourth Central Moment (KT) risk measure [`HighOrderMoment`](@ref) + [`StandardisedHighOrderMoment`](@ref) + [`FourthMoment`](@ref) + [`FullMoment`](@ref). This represents the scenario based kurtosis of the return distribution.
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

Alias for the Standardised Fourth Lower Moment (SKT) risk measure [`HighOrderMoment`](@ref) + [`StandardisedHighOrderMoment`](@ref) + [`FourthMoment`](@ref) + [`SemiMoment`](@ref). This represents the scenario based semi-kurtosis of the return distribution.
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
    VaR

Alias for [`ValueatRisk`](@ref).
"""
const VaR = ValueatRisk
"""
    CVaR

Alias for [`ConditionalValueatRisk`](@ref).
"""
const CVaR = ConditionalValueatRisk
"""
    DRCVaR

Alias for [`DistributionallyRobustConditionalValueatRisk`](@ref).
"""
const DRCVaR = DistributionallyRobustConditionalValueatRisk
"""
    EVaR

Alias for [`EntropicValueatRisk`](@ref).
"""
const EVaR = EntropicValueatRisk
"""
    RVaR

Alias for [`RelativisticValueatRisk`](@ref).
"""
const RVaR = RelativisticValueatRisk
"""
    PNVaR

Alias for [`PowerNormValueatRisk`](@ref).
"""
const PNVaR = PowerNormValueatRisk

# ── Risk measures — Value-at-Risk Range family ──────────────────────────────────────
"""
    VaR_RG

Alias for [`ValueatRiskRange`](@ref).
"""
const VaR_RG = ValueatRiskRange
"""
    CVaR_RG

Alias for [`ConditionalValueatRiskRange`](@ref).
"""
const CVaR_RG = ConditionalValueatRiskRange
"""
    DRCVaR_RG

Alias for [`DistributionallyRobustConditionalValueatRiskRange`](@ref).
"""
const DRCVaR_RG = DistributionallyRobustConditionalValueatRiskRange
"""
    EVaR_RG

Alias for [`EntropicValueatRiskRange`](@ref).
"""
const EVaR_RG = EntropicValueatRiskRange
"""
    RVaR_RG

Alias for [`RelativisticValueatRiskRange`](@ref).
"""
const RVaR_RG = RelativisticValueatRiskRange
"""
    PNVaR_RG

Alias for [`PowerNormValueatRiskRange`](@ref).
"""
const PNVaR_RG = PowerNormValueatRiskRange

# ── Risk measures — Drawdown-at-Risk family ──────────────────────────────────────
"""
    DaR

Alias for [`DrawdownatRisk`](@ref).
"""
const DaR = DrawdownatRisk
"""
    CDaR

Alias for [`ConditionalDrawdownatRisk`](@ref).
"""
const CDaR = ConditionalDrawdownatRisk
"""
    DRCDaR

Alias for [`DistributionallyRobustConditionalDrawdownatRisk`](@ref).
"""
const DRCDaR = DistributionallyRobustConditionalDrawdownatRisk
"""
    EDaR

Alias for [`EntropicDrawdownatRisk`](@ref).
"""
const EDaR = EntropicDrawdownatRisk
"""
    RDaR

Alias for [`RelativisticDrawdownatRisk`](@ref).
"""
const RDaR = RelativisticDrawdownatRisk
"""
    PNDaR

Alias for [`PowerNormDrawdownatRisk`](@ref).
"""
const PNDaR = PowerNormDrawdownatRisk

"""
    R_DaR

Alias for [`RelativeDrawdownatRisk`](@ref).
"""
const R_DaR = RelativeDrawdownatRisk
"""
    R_CDaR

Alias for [`RelativeConditionalDrawdownatRisk`](@ref).
"""
const R_CDaR = RelativeConditionalDrawdownatRisk
"""
    R_EDaR

Alias for [`RelativeEntropicDrawdownatRisk`](@ref).
"""
const R_EDaR = RelativeEntropicDrawdownatRisk
"""
    R_RDaR

Alias for [`RelativeRelativisticDrawdownatRisk`](@ref).
"""
const R_RDaR = RelativeRelativisticDrawdownatRisk
"""
    R_PNDaR

Alias for [`RelativePowerNormDrawdownatRisk`](@ref).
"""
const R_PNDaR = RelativePowerNormDrawdownatRisk

# ── Risk measures — Ordered Weights Array family ──────────────────────────────────────
"""
    OWA

Alias for [`OrderedWeightsArray`](@ref).
"""
const OWA = OrderedWeightsArray
"""
    OWA_RG

Alias for [`OrderedWeightsArrayRange`](@ref).
"""
const OWA_RG = OrderedWeightsArrayRange
