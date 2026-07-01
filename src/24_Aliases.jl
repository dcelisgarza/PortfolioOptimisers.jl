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
    NOC

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
    WR

Alias for [`WorstRealisation`](@ref).
"""
const WR = WorstRealisation
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
    RG

Alias for [`Range`](@ref).
"""
const RG = Range
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
"""
    GVaR_RG

Alias for [`GenericValueatRiskRange`](@ref).
"""
const GVaR_RG = GenericValueatRiskRange

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
    OWA_GMD(; settings::RiskMeasureSettings = RiskMeasureSettings(),
              alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()) -> OrderedWeightsArray

Alias for the Gini Mean Difference (GMD) [`OrderedWeightsArray`](@ref) risk measure using [`owa_gmd`](@ref) weights.
"""
function OWA_GMD(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                 alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(; settings = settings, w = owa_gmd, alg = alg)
end
"""
    OWA_CVaR(; settings::RiskMeasureSettings = RiskMeasureSettings(),
               alpha::Number = 0.05,
               alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()) -> OrderedWeightsArray

Alias for the Conditional Value at Risk (CVaR) [`OrderedWeightsArray`](@ref) risk measure using [`OrderedWeightsArrayConditionalValueatRisk`](@ref) weights at significance level `alpha`.
"""
function OWA_CVaR(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  alpha::Number = 0.05,
                  alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(; settings = settings,
                               w = OrderedWeightsArrayConditionalValueatRisk(;
                                                                             alpha = alpha),
                               alg = alg)
end
"""
    OWA_TG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             alpha_i::Number = 1e-4,
             alpha::Number = 0.05,
             a_sim::Integer = 100,
             alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()) -> OrderedWeightsArray

Alias for the Tail Gini (TG) [`OrderedWeightsArray`](@ref) risk measure using [`OrderedWeightsArrayTailGini`](@ref) weights.
"""
function OWA_TG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                alpha_i::Number = 1e-4, alpha::Number = 0.05, a_sim::Integer = 100,
                alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(; settings = settings,
                               w = OrderedWeightsArrayTailGini(; alpha_i = alpha_i,
                                                               alpha = alpha,
                                                               a_sim = a_sim), alg = alg)
end
"""
    OWA_WR(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()) -> OrderedWeightsArray

Alias for the Worst Realisation (WR) [`OrderedWeightsArray`](@ref) risk measure using [`owa_wr`](@ref) weights.
"""
function OWA_WR(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(; settings = settings, w = owa_wr, alg = alg)
end
"""
    OWA_RG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()) -> OrderedWeightsArray

Alias for the Range (RG) [`OrderedWeightsArray`](@ref) risk measure using [`owa_rg`](@ref) weights.
"""
function OWA_RG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(; settings = settings, w = owa_rg, alg = alg)
end
"""
    OWA_CVaR_RG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  alpha::Number = 0.05,
                  beta::Number = alpha,
                  alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()) -> OrderedWeightsArray

Alias for the Conditional Value at Risk Range (CVaR RG) [`OrderedWeightsArray`](@ref) risk measure using [`OrderedWeightsArrayConditionalValueatRiskRange`](@ref) weights.
"""
function OWA_CVaR_RG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                     alpha::Number = 0.05, beta::Number = alpha,
                     alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(; settings = settings,
                               w = OrderedWeightsArrayConditionalValueatRiskRange(;
                                                                                  alpha = alpha,
                                                                                  beta = beta),
                               alg = alg)
end
"""
    OWA_TG_RG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                alpha_i::Number = 1e-4,
                alpha::Number = 0.05,
                a_sim::Integer = 100,
                beta_i::Number = alpha_i,
                beta::Number = alpha,
                b_sim::Integer = a_sim,
                alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()) -> OrderedWeightsArray

Alias for the Tail Gini Range (TG RG) [`OrderedWeightsArray`](@ref) risk measure using [`OrderedWeightsArrayTailGiniRange`](@ref) weights.
"""
function OWA_TG_RG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                   alpha_i::Number = 1e-4, alpha::Number = 0.05, a_sim::Integer = 100,
                   beta_i::Number = alpha_i, beta::Number = alpha, b_sim::Integer = a_sim,
                   alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(; settings = settings,
                               w = OrderedWeightsArrayTailGiniRange(; alpha_i = alpha_i,
                                                                    alpha = alpha,
                                                                    a_sim = a_sim,
                                                                    beta_i = beta_i,
                                                                    beta = beta,
                                                                    b_sim = b_sim),
                               alg = alg)
end
"""
    OWA_LMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion(),
                  k::Integer = 2,
                  alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray()) -> OrderedWeightsArray

Alias for the L-Moment [`OrderedWeightsArray`](@ref) risk measure using [`LinearMoment`](@ref) weights of order `k`.
"""
function OWA_LMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                     method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion(),
                     k::Integer = 2,
                     alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray())
    return OrderedWeightsArray(; settings = settings,
                               w = LinearMoment(; method = method, k = k), alg = alg)
end

# ── Risk measures — Drawdown family ──────────────────────────────────────
"""
    ADD

Alias for [`AverageDrawdown`](@ref).
"""
const ADD = AverageDrawdown
"""
    R_ADD

Alias for [`RelativeAverageDrawdown`](@ref).
"""
const R_ADD = RelativeAverageDrawdown
"""
    UCI

Alias for [`UlcerIndex`](@ref).
"""
const UCI = UlcerIndex
"""
    R_UCI

Alias for [`RelativeUlcerIndex`](@ref).
"""
const R_UCI = RelativeUlcerIndex
"""
    MDD

Alias for [`MaximumDrawdown`](@ref).
"""
const MDD = MaximumDrawdown
"""
    R_MDD

Alias for [`RelativeMaximumDrawdown`](@ref).
"""
const R_MDD = RelativeMaximumDrawdown

# ── Risk measures — Nonlinear relationships ──────────────────────────────────────
"""
    BDVariance

Alias for [`BrownianDistanceVariance`](@ref).
"""
const BDVariance = BrownianDistanceVariance

# ── Risk measures — Tracking and turnover ──────────────────────────────────────
"""
    TrRM

Alias for [`TrackingRiskMeasure`](@ref).
"""
const TrRM = TrackingRiskMeasure
"""
    RkTrRM

Alias for [`RiskTrackingRiskMeasure`](@ref).
"""
const RkTrRM = RiskTrackingRiskMeasure
"""
    TnRM

Alias for [`TurnoverRiskMeasure`](@ref).
"""
const TnRM = TurnoverRiskMeasure

# ── Risk measures — Higher-order moments ──────────────────────────────────────
"""
    VSK

Alias for [`VarianceSkewKurtosis`](@ref).
"""
const VSK = VarianceSkewKurtosis

# ── Risk measures — Performance risk measures ──────────────────────────────────────
"""
    NonOptRkRatio

Alias for [`NonOptimisationRiskRatio`](@ref).
"""
const NonOptRkRatio = NonOptimisationRiskRatio
