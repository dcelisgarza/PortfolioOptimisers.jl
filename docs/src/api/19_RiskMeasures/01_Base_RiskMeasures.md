# Base Risk Measures

All risk measures are defined as their whole names, however this can be unwieldy, so we also provide convenience aliases defined in [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/25_Aliases).

All concrete risk measures can be used as functors (callable structs) to compute their associated risk quantity, according to its [`risk_input_kind`](@ref), or via [`expected_risk`](@ref).

## Quick-pick guide

Three legal-usage classes:

- Optimisation: JuMP formulation; usable as objective/constraint.
- Hierarchical: clustering proxy, no JuMP formulation.
- Non-Optimisation: analysis only.

Table Key:

- `QP`: Quadratic programming (generates a quadratic expression).
- `NOC`: Norm one none.
- `SOC`: Second order cone
- `RSOC`: Rotated second order cone.
- `NIC`: Norm infinity cone.
- `EC`: Exponential cone.
- `PC`: 3D Power cone.
- `Sym`: Symmetric matrix space.
- `PSD`: Positive semi-definite cone.
- `MIP`: Mixed-integer variables.
- `*`: Carries the requirements of its inner risk measures.
- `-`: Not applicable.

| Type          | Name (Alias)                                                                                                                                  | Compatibility       | Requirements   | Rec. Solver                 |
|:------------- |:--------------------------------------------------------------------------------------------------------------------------------------------- |:------------------- |:-------------- |:--------------------------- |
| Dispersion    | [`Variance`](@ref)                                                                                                                            | JuMP + Hierarchical | QP, SOC        | Clarabel                    |
| Dispersion    | SDP graph / Risk contribution [`Variance`](@ref)                                                                                              | JuMP + Hierarchical | SDP            | Clarabel, SCS               |
| Dispersion    | [`StandardDeviation`](@ref) ([`SD`](@ref))                                                                                                    | JuMP + Hierarchical | SOC            | Clarabel                    |
| Dispersion    | Box [`UncertaintySetVariance`](@ref) ([`UcVariance`](@ref))                                                                                   | JuMP + Hierarchical | Sym            | Clarabel                    |
| Dispersion    | Ellipse [`UncertaintySetVariance`](@ref) ([`UcVariance`](@ref))                                                                               | JuMP + Hierarchical | Sym, PSD, SOC  | Clarabel, SCS               |
| Dispersion    | [`LowOrderMoment`](@ref) ([`FLM`](@ref))                                                                                                      | JuMP + Hierarchical | -              | Clarabel                    |
| Dispersion    | [`LowOrderMoment`](@ref) ([`MAD`](@ref))                                                                                                      | JuMP + Hierarchical | -              | Clarabel                    |
| Dispersion    | Direct / Squared SOC [`LowOrderMoment`](@ref) ([`SCM`](@ref), [`SLM`](@ref))                                                                  | JuMP + Hierarchical | QP, SOC        | Clarabel                    |
| Dispersion    | Sum of Squares [`LowOrderMoment`](@ref) ([`SCM`](@ref), [`SLM`](@ref))                                                                        | JuMP + Hierarchical | SOC, RSOC      | Clarabel                    |
| Dispersion    | SOC [`LowOrderMoment`](@ref) ([`SCM`](@ref), [`SLM`](@ref))                                                                                   | JuMP + Hierarchical | SOC            | Clarabel                    |
| Dispersion    | [`LowOrderMoment`](@ref) ([`ECM`](@ref), [`ELM`](@ref))                                                                                       | JuMP + Hierarchical | PC             | Clarabel                    |
| Dispersion    | [`HighOrderMoment`](@ref) ([`TLM`](@ref), [`SSK`](@ref), [`FTCM`](@ref), [`FTLM`](@ref), [`KT`](@ref), [`SKT`](@ref))                         | Hierarchical        | -              | -                           |
| Dispersion    | Direct / Squared SOC Exact [`Kurtosis`](@ref)                                                                                                 | JuMP + Hierarchical | QP, SOC, PSD   | Clarabel, SCS               |
| Dispersion    | Sum of Squares Exact [`Kurtosis`](@ref)                                                                                                       | JuMP + Hierarchical | SOC, RSOC, PSD | Clarabel, SCS               |
| Dispersion    | SOC Exact [`Kurtosis`](@ref)                                                                                                                  | JuMP + Hierarchical | SOC, PSD       | Clarabel, SCS               |
| Dispersion    | Direct / Squared SOC Approx [`Kurtosis`](@ref)                                                                                                | JuMP + Hierarchical | QP, SOC        | Clarabel                    |
| Dispersion    | Sum of Squares Approx [`Kurtosis`](@ref)                                                                                                      | JuMP + Hierarchical | SOC, RSOC      | Clarabel                    |
| Dispersion    | SOC Exact Approx [`Kurtosis`](@ref)                                                                                                           | JuMP + Hierarchical | SOC            | Clarabel                    |
| Dispersion    | [`NegativeSkewness`](@ref)                                                                                                                    | JuMP + Hierarchical | QP, SOC        | Clarabel                    |
| Dispersion    | Square Root [`NegativeSkewness`](@ref)                                                                                                        | JuMP + Hierarchical | SOC            | Clarabel                    |
| Tail loss     | Exact [`ValueatRisk`](@ref) ([`VaR`](@ref))                                                                                                   | JuMP + Hierarchical | MIP            | Pajarito (Clarabel + HiGHS) |
| Tail loss     | Approx [`ValueatRisk`](@ref) ([`VaR`](@ref))                                                                                                  | JuMP + Hierarchical | SOC            | Clarabel                    |
| Tail drawdown | [`DrawdownatRisk`](@ref) ([`DaR`](@ref))                                                                                                      | JuMP + Hierarchical | MIP            | Pajarito (Clarabel + HiGHS) |
| Tail drawdown | [`RelativeDrawdownatRisk`](@ref) ([`R_DaR`](@ref))                                                                                            | Hierarchical        | -              | Clarabel                    |
| Dispersion    | Exact [`ValueatRiskRange`](@ref) ([`VaR_RG`](@ref))                                                                                           | JuMP + Hierarchical | MIP            | Pajarito (Clarabel + HiGHS) |
| Dispersion    | Approx [`ValueatRiskRange`](@ref) ([`VaR_RG`](@ref))                                                                                          | JuMP + Hierarchical | SOC            | Clarabel                    |
| Tail loss     | [`ConditionalValueatRisk`](@ref) ([`CVaR`](@ref))                                                                                             | JuMP + Hierarchical | -              | Clarabel                    |
| Tail loss     | [`DistributionallyRobustConditionalValueatRisk`](@ref) ([`DRCVaR`](@ref))                                                                     | JuMP + Hierarchical | NIC            | Clarabel                    |
| Tail drawdown | [`ConditionalDrawdownatRisk`](@ref) ([`CDaR`](@ref))                                                                                          | JuMP + Hierarchical | -              | Clarabel                    |
| Tail drawdown | [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) ([`DRCDaR`](@ref))                                                                  | JuMP + Hierarchical | NIC            | Clarabel                    |
| Tail drawdown | [`RelativeConditionalDrawdownatRisk`](@ref) ([`R_CDaR`](@ref))                                                                                | Hierarchical        | -              | Clarabel                    |
| Dispersion    | [`ConditionalValueatRiskRange`](@ref) ([`CVaR_RG`](@ref))                                                                                     | JuMP + Hierarchical | -              | Clarabel                    |
| Dispersion    | [`DistributionallyRobustConditionalValueatRiskRange`](@ref) ([`DRCVaR_RG`](@ref))                                                             | JuMP + Hierarchical | NIC            | Clarabel                    |
| Tail loss     | [`EntropicValueatRisk`](@ref) ([`EVaR`](@ref))                                                                                                | JuMP + Hierarchical | EC             | Clarabel                    |
| Tail drawdown | [`EntropicDrawdownatRisk`](@ref) ([`EDaR`](@ref))                                                                                             | JuMP + Hierarchical | EC             | Clarabel                    |
| Tail drawdown | [`RelativeEntropicDrawdownatRisk`](@ref) ([`R_EDaR`](@ref))                                                                                   | Hierarchical        | EC             | Clarabel                    |
| Dispersion    | [`EntropicValueatRiskRange`](@ref) ([`EVaR_RG`](@ref))                                                                                        | JuMP + Hierarchical | EC             | Clarabel                    |
| Tail loss     | [`RelativisticValueatRisk`](@ref) ([`RVaR`](@ref))                                                                                            | JuMP + Hierarchical | PC             | Clarabel                    |
| Tail drawdown | [`RelativisticDrawdownatRisk`](@ref) ([`RDaR`](@ref))                                                                                         | JuMP + Hierarchical | PC             | Clarabel                    |
| Tail drawdown | [`RelativeRelativisticDrawdownatRisk`](@ref) ([`R_RDaR`](@ref))                                                                               | Hierarchical        | PC             | Clarabel                    |
| Dispersion    | [`RelativisticValueatRiskRange`](@ref) ([`RVaR_RG`](@ref))                                                                                    | JuMP + Hierarchical | PC             | Clarabel                    |
| Tail loss     | [`PowerNormValueatRisk`](@ref) ([`PNVaR`](@ref))                                                                                              | JuMP + Hierarchical | PC             | Clarabel                    |
| Tail drawdown | [`PowerNormDrawdownatRisk`](@ref) ([`PNDaR`](@ref))                                                                                           | JuMP + Hierarchical | PC             | Clarabel                    |
| Tail drawdown | [`RelativePowerNormDrawdownatRisk`](@ref) ([`R_PNDaR`](@ref))                                                                                 | Hierarchical        | PC             | Clarabel                    |
| Dispersion    | [`PowerNormValueatRiskRange`](@ref) ([`PNVaR_RG`](@ref))                                                                                      | JuMP + Hierarchical | PC             | Clarabel                    |
| Dispersion    | Exact [`OrderedWeightsArray`](@ref) ([`OWA_GMD`](@ref), [`OWA_RG`](@ref), [`OWA_CVaR_RG`](@ref), [`OWA_TG_RG`](@ref), [`OWA_LMoment`](@ref))  | JuMP + Hierarchical | -              | Clarabel                    |
| Dispersion    | Approx [`OrderedWeightsArray`](@ref) ([`OWA_GMD`](@ref), [`OWA_RG`](@ref), [`OWA_CVaR_RG`](@ref), [`OWA_TG_RG`](@ref), [`OWA_LMoment`](@ref)) | JuMP + Hierarchical | PC             | Clarabel                    |
| Tail loss     | Exact [`OrderedWeightsArray`](@ref) ([`OWA_CVaR`](@ref), [`OWA_TG`](@ref), [`OWA_WR`](@ref))                                                  | JuMP + Hierarchical | -              | Clarabel                    |
| Tail loss     | Approx [`OrderedWeightsArray`](@ref) ([`OWA_CVaR`](@ref), [`OWA_TG`](@ref), [`OWA_WR`](@ref))                                                 | JuMP + Hierarchical | PC             | Clarabel                    |
| Dispersion    | Exact [`OrderedWeightsArrayRange`](@ref)                                                                                                      | JuMP + Hierarchical | -              | Clarabel                    |
| Dispersion    | Approx [`OrderedWeightsArrayRange`](@ref)                                                                                                     | JuMP + Hierarchical | PC             | Clarabel                    |
| Drawdown      | [`AverageDrawdown`](@ref) ([`ADD`](@ref))                                                                                                     | JuMP + Hierarchical | -              | Clarabel                    |
| Drawdown      | [`RelativeAverageDrawdown`](@ref) ([`R_ADD`](@ref))                                                                                           | Hierarchical        | -              | -                           |
| Drawdown      | [`UlcerIndex`](@ref) ([`UCI`](@ref))                                                                                                          | JuMP + Hierarchical | SOC            | Clarabel                    |
| Drawdown      | [`RelativeUlcerIndex`](@ref) ([`R_UCI`](@ref))                                                                                                | Hierarchical        | -              | -                           |
| Tail drawdown | [`MaximumDrawdown`](@ref) ([`MDD`](@ref))                                                                                                     | JuMP + Hierarchical | -              | Clarabel                    |
| Tail drawdown | [`RelativeMaximumDrawdown`](@ref) ([`R_MDD`](@ref))                                                                                           | Hierarchical        | -              | -                           |
| Dispersion    | Direct [`BrownianDistanceVariance`](@ref) ([`BDVariance`](@ref))                                                                              | JuMP + Hierarchical | QP             | Clarabel                    |
| Dispersion    | Sum of Squares [`BrownianDistanceVariance`](@ref) ([`BDVariance`](@ref))                                                                      | JuMP + Hierarchical | QP, RSOC       | Clarabel                    |
| Tail loss     | [`WorstRealisation`](@ref) ([`WR`](@ref))                                                                                                     | JuMP + Hierarchical | -              | Clarabel                    |
| Tail loss     | [`Range`](@ref) ([`RG`](@ref))                                                                                                                | JuMP + Hierarchical | -              | Clarabel                    |
| Turnover      | [`TurnoverRiskMeasure`](@ref) ([`TnRM`](@ref))                                                                                                | JuMP + Hierarchical | NOC            | Clarabel                    |
| Tracking      | L1 Norm [`TrackingRiskMeasure`](@ref) ([`TrRM`](@ref))                                                                                        | JuMP + Hierarchical | NOC            | Clarabel                    |
| Tracking      | L2 Norm [`TrackingRiskMeasure`](@ref) ([`TrRM`](@ref))                                                                                        | JuMP + Hierarchical | SOC            | Clarabel                    |
| Tracking      | Squared L2 Norm [`TrackingRiskMeasure`](@ref) ([`TrRM`](@ref))                                                                                | JuMP + Hierarchical | QP, SOC        | Clarabel                    |
| Tracking      | Lp Norm [`TrackingRiskMeasure`](@ref) ([`TrRM`](@ref))                                                                                        | JuMP + Hierarchical | PC             | Clarabel                    |
| Tracking      | Infinity Norm [`TrackingRiskMeasure`](@ref) ([`TrRM`](@ref))                                                                                  | JuMP + Hierarchical | NIC            | Clarabel                    |
| Risk tracking | Independent variable [`RiskTrackingRiskMeasure`](@ref) ([`RkTrRM`](@ref))                                                                     | JuMP + Hierarchical | *              | Clarabel                    |
| Risk tracking | Dependent variable [`RiskTrackingRiskMeasure`](@ref) ([`RkTrRM`](@ref))                                                                       | JuMP + Hierarchical | NOC, *         | Clarabel                    |
| Dispersion    | [`VarianceSkewKurtosis`](@ref) ([`VSK`](@ref))                                                                                                | JuMP + Hierarchical | Sym, PSD       | SCS                         |
| Dispersion    | [`GenericValueatRiskRange`](@ref) ([`GVaR_RG`](@ref))                                                                                         | JuMP + Hierarchical | *              | Clarabel                    |
| Ratio         | [`RiskRatio`](@ref)                                                                                                                           | Hierarchical        | *              | *                           |
| Ratio         | [`NonOptimisationRiskRatio`](@ref) ([`NonOptRkRatio`](@ref))                                                                                  | -                   | *              | *                           |
| Flat          | [`EqualRisk`](@ref)                                                                                                                           | -                   | -              | -                           |
| Dispersion    | [`MedianAbsoluteDeviation`](@ref)                                                                                                             | Hierarchical        | -              | -                           |
| Performance   | [`MeanReturn`](@ref)                                                                                                                          | -                   | -              | -                           |
| Performance   | [`MeanReturnRiskRatio`](@ref)                                                                                                                 | -                   | -              | -                           |
| Performance   | [`ExpectedReturn`](@ref)                                                                                                                      | -                   | -              | -                           |
| Performance   | [`ExpectedReturnRiskRatio`](@ref)                                                                                                             | -                   | -              | -                           |
| Dispersion    | [`ThirdCentralMoment`](@ref)                                                                                                                  | -                   | -              | -                           |
| Dispersion    | [`Skewness`](@ref)                                                                                                                            | -                   | -              | -                           |

```@docs
RiskMeasure
HierarchicalRiskMeasure
Frontier
RiskMeasureSettings
HierarchicalRiskMeasureSettings
SumScalariser
MaxScalariser
MinScalariser
LogSumExpScalariser
scalarise
scalarise_combine
scalarise_map
scalarise_logsumexp
AbstractBaseRiskMeasure
NonOptimisationRiskMeasure
OptimisationRiskMeasure
AbstractRiskMeasureSettings
JuMPRiskMeasureSettings
FrontierBoundEstimator
LinearBound
SquareRootBound
SquaredBound
Scalariser
NonHierarchicalScalariser
HierarchicalScalariser
nothing_scalar_array_selector
risk_measure_nothing_scalar_array_view
solver_selector
VecBaseRM
VecOptRM
OptRM_VecOptRM
VecRM
RM_VecRM
RkRtBounds
Front_NumVec
bigger_is_better
needs_previous_weights(::AbstractBaseRiskMeasure)
RiskInputKind
NetReturnsInput
WeightsReturnsFeesInput
WeightsInput
risk_input_kind
supports_precomputed_returns(r::AbstractBaseRiskMeasure)
supports_precomputed_returns(::NetReturnsInput, ::Any)
supports_precomputed_returns(::WeightsInput, ::Any)
supports_precomputed_returns(::WeightsReturnsFeesInput, r::AbstractBaseRiskMeasure)
weight_independent_target(::Nothing)
weight_independent_target(::Number)
weight_independent_target(::Any)
factory(rs::AbstractBaseRiskMeasure, args...; kwargs...)
factory(rs::VecBaseRM, args...; kwargs...)
port_opt_view(rs::AbstractBaseRiskMeasure, ::Any, ::Any, args...)
_Frontier
```
