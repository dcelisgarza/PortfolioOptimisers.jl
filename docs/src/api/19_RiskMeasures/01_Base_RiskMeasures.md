# Base Risk Measures

All risk measures are defined as their whole names, however this can be unwieldy, so we also provide convenience aliases defined in [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/24_Aliases).

All concrete risk measures can be used as functors (callable structs) to compute their associated risk quantity, according to its [`risk_input_kind`](@ref), or via [`expected_risk`](@ref).

## Quick-pick guide

Three legal-usage classes:

- Optimisation: JuMP formulation; usable as objective/constraint.
- Hierarchical: clustering proxy, no JuMP formulation.
- Non-Optimisation: analysis only.

| Type          | Name (Alias)                                                                                                          | Compatibility       | JuMP Expr & Cones  | Solver                      |
|:------------- |:--------------------------------------------------------------------------------------------------------------------- |:------------------- |:------------------ |:--------------------------- |
| Dispersion    | [`Variance`](@ref)                                                                                                    | JuMP + Hierarchical | QuadExpr, SOC      | Clarabel                    |
| Dispersion    | [`StandardDeviation`](@ref) ([`SD`](@ref))                                                                            | JuMP + Hierarchical | SOC                | Clarabel                    |
| Dispersion    | Box [`UncertaintySetVariance`](@ref) ([`UcVariance`](@ref))                                                           | JuMP + Hierarchical | Sym                | Clarabel                    |
| Dispersion    | Ellipse [`UncertaintySetVariance`](@ref) ([`UcVariance`](@ref))                                                       | JuMP + Hierarchical | Sym, PSD, SOC      | Clarabel, SCS               |
| Dispersion    | [`LowOrderMoment`](@ref) ([`FLM`](@ref))                                                                              | JuMP + Hierarchical | -                  | Clarabel                    |
| Dispersion    | [`LowOrderMoment`](@ref) ([`MAD`](@ref))                                                                              | JuMP + Hierarchical | -                  | Clarabel                    |
| Dispersion    | Direct / Squared SOC [`LowOrderMoment`](@ref) ([`SCM`](@ref), [`SLM`](@ref))                                          | JuMP + Hierarchical | QuadExpr, SOC      | Clarabel                    |
| Dispersion    | Sum of Squares [`LowOrderMoment`](@ref) ([`SCM`](@ref), [`SLM`](@ref))                                                | JuMP + Hierarchical | SOC, RSOC          | Clarabel                    |
| Dispersion    | SOC [`LowOrderMoment`](@ref) ([`SCM`](@ref), [`SLM`](@ref))                                                           | JuMP + Hierarchical | SOC                | Clarabel                    |
| Dispersion    | [`LowOrderMoment`](@ref) ([`ECM`](@ref), [`ELM`](@ref))                                                               | JuMP + Hierarchical | PC                 | Clarabel                    |
| Dispersion    | [`HighOrderMoment`](@ref) ([`TLM`](@ref), [`SSK`](@ref), [`FTCM`](@ref), [`FTLM`](@ref), [`KT`](@ref), [`SKT`](@ref)) | Hierarchical        | -                  | -                           |
| Dispersion    | Direct / Squared SOC Exact [`Kurtosis`](@ref)                                                                         | JuMP + Hierarchical | QuadExpr, SOC, PSD | Clarabel, SCS               |
| Dispersion    | Sum of Squares Exact [`Kurtosis`](@ref)                                                                               | JuMP + Hierarchical | SOC, RSOC, PSD     | Clarabel, SCS               |
| Dispersion    | SOC Exact [`Kurtosis`](@ref)                                                                                          | JuMP + Hierarchical | SOC, PSD           | Clarabel, SCS               |
| Dispersion    | Direct / Squared SOC Approx [`Kurtosis`](@ref)                                                                        | JuMP + Hierarchical | QuadExpr, SOC      | Clarabel                    |
| Dispersion    | Sum of Squares Approx [`Kurtosis`](@ref)                                                                              | JuMP + Hierarchical | SOC, RSOC          | Clarabel                    |
| Dispersion    | SOC Exact Approx [`Kurtosis`](@ref)                                                                                   | JuMP + Hierarchical | SOC                | Clarabel                    |
| Dispersion    | [`NegativeSkewness`](@ref)                                                                                            | JuMP + Hierarchical | QuadExpr, SOC      | Clarabel                    |
| Dispersion    | Square Root [`NegativeSkewness`](@ref)                                                                                | JuMP + Hierarchical | SOC                | Clarabel                    |
| Tail loss     | Exact [`ValueatRisk`](@ref) ([`VaR`](@ref))                                                                           | JuMP + Hierarchical | MIP                | Pajarito (Clarabel + HiGHS) |
| Tail loss     | Approx [`ValueatRisk`](@ref) ([`VaR`](@ref))                                                                          | JuMP + Hierarchical | SOC                | Clarabel                    |
| Tail drawdown | [`DrawdownatRisk`](@ref) ([`DaR`](@ref))                                                                              | JuMP + Hierarchical | MIP                | Pajarito (Clarabel + HiGHS) |
| Tail drawdown | [`RelativeDrawdownatRisk`](@ref) ([`R_DaR`](@ref))                                                                    | Hierarchical        | -                  | Clarabel                    |
| Dispersion    | Exact [`ValueatRiskRange`](@ref) ([`VaR_RG`](@ref))                                                                   | JuMP + Hierarchical | MIP                | Pajarito (Clarabel + HiGHS) |
| Dispersion    | Approx [`ValueatRiskRange`](@ref) ([`VaR_RG`](@ref))                                                                  | JuMP + Hierarchical | SOC                | Clarabel                    |
| Tail loss     | [`ConditionalValueatRisk`](@ref) ([`CVaR`](@ref))                                                                     | JuMP + Hierarchical | -                  | Clarabel                    |
| Tail loss     | [`DistributionallyRobustConditionalValueatRisk`](@ref) ([`DRCVaR`](@ref))                                             | JuMP + Hierarchical | NIC                | Clarabel                    |
| Tail drawdown | [`ConditionalDrawdownatRisk`](@ref) ([`CDaR`](@ref))                                                                  | JuMP + Hierarchical | -                  | Clarabel                    |
| Tail drawdown | [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) ([`DRCDaR`](@ref))                                          | JuMP + Hierarchical | NIC                | Clarabel                    |
| Tail drawdown | [`RelativeConditionalDrawdownatRisk`](@ref) ([`R_CDaR`](@ref))                                                        | Hierarchical        | -                  | Clarabel                    |
| Dispersion    | [`ConditionalValueatRiskRange`](@ref) ([`CVaR_RG`](@ref))                                                             | JuMP + Hierarchical | -                  | Clarabel                    |
| Dispersion    | [`DistributionallyRobustConditionalValueatRiskRange`](@ref) ([`DRCVaR_RG`](@ref))                                     | JuMP + Hierarchical | NIC                | Clarabel                    |
| Tail loss     | [`EntropicValueatRisk`](@ref) ([`EVaR`](@ref))                                                                        | JuMP + Hierarchical | EC                 | Clarabel                    |
| Tail drawdown | [`EntropicDrawdownatRisk`](@ref) ([`EDaR`](@ref))                                                                     | JuMP + Hierarchical | EC                 | Clarabel                    |
| Tail drawdown | [`RelativeEntropicDrawdownatRisk`](@ref) ([`R_EDaR`](@ref))                                                           | Hierarchical        | -                  | Clarabel                    |
| Dispersion    | [`EntropicValueatRiskRange`](@ref) ([`EVaR_RG`](@ref))                                                                | JuMP + Hierarchical | EC                 | Clarabel                    |
| Tail loss     | [`RelativisticValueatRisk`](@ref) ([`RVaR`](@ref))                                                                    | JuMP + Hierarchical | PC                 | Clarabel                    |
| Tail drawdown | [`RelativisticDrawdownatRisk`](@ref) ([`RDaR`](@ref))                                                                 | JuMP + Hierarchical | PC                 | Clarabel                    |
| Tail drawdown | [`RelativeRelativisticDrawdownatRisk`](@ref) ([`R_RDaR`](@ref))                                                       | Hierarchical        | -                  | Clarabel                    |
| Dispersion    | [`RelativisticValueatRiskRange`](@ref) ([`RVaR_RG`](@ref))                                                            | JuMP + Hierarchical | PC                 | Clarabel                    |
| Tail loss     | [`PowerNormValueatRisk`](@ref) ([`PNVaR`](@ref))                                                                      | JuMP + Hierarchical | PC                 | Clarabel                    |
| Tail drawdown | [`PowerNormDrawdownatRisk`](@ref) ([`PNDaR`](@ref))                                                                   | JuMP + Hierarchical | PC                 | Clarabel                    |
| Tail drawdown | [`RelativePowerNormDrawdownatRisk`](@ref) ([`R_PNDaR`](@ref))                                                         | Hierarchical        | -                  | Clarabel                    |
| Dispersion    | [`PowerNormValueatRiskRange`](@ref) ([`PNVaR_RG`](@ref))                                                              | JuMP + Hierarchical | PC                 | Clarabel                    |

<!-- continue with owa,, make aliases for the different weights -->

**Tail / quantile measures (XatRisk family)** — pick by what you are measuring:

| Goal                                  | Returns-based                                      | Drawdown-based                                         |
|:------------------------------------- |:-------------------------------------------------- |:------------------------------------------------------ |
| Conditional tail loss (CVaR / CDaR)   | [`ConditionalValueatRisk`](@ref) ([`CVaR`](@ref))  | [`ConditionalDrawdownatRisk`](@ref) ([`CDaR`](@ref))   |
| Entropic tail loss (EVaR / EDaR)      | [`EntropicValueatRisk`](@ref) ([`EVaR`](@ref))     | [`EntropicDrawdownatRisk`](@ref) ([`EDaR`](@ref))      |
| Relativistic tail loss (RVaR / RDaR)  | [`RelativisticValueatRisk`](@ref) ([`RVaR`](@ref)) | [`RelativisticDrawdownatRisk`](@ref) ([`RDaR`](@ref))  |
| Power-norm tail loss (PNVaR / PNDaR)  | [`PowerNormValueatRisk`](@ref) ([`PNVaR`](@ref))   | [`PowerNormDrawdownatRisk`](@ref) ([`PNDaR`](@ref))    |
| Pointwise tail loss (VaR / DaR)       | [`ValueatRisk`](@ref) ([`VaR`](@ref))              | [`DrawdownatRisk`](@ref) ([`DaR`](@ref))               |

- Append the suffix `Range` to the full name (or `_RG` to the alias) to any of the returns based risk measures above for the gap between the upper and lower tail (e.g. [`ConditionalValueatRiskRange`](@ref), [`CVaR_RG`](@ref)).
- Append the prefix `Relative` to the full name (or `R_` to the alias) to any of the drawdowns based risk measures above for the compounded version of drawdowns.

**Dispersion / moment measures:** [`Variance`](@ref), [`StandardDeviation`](@ref), [`MeanAbsoluteDeviation`](@ref), [`MedianAbsoluteDeviation`](@ref), [`Kurtosis`](@ref), [`NegativeSkewness`](@ref), [`OWA`](@ref)-based ([`OrderedWeightsArray`](@ref)).

**Drawdown statistics (non-quantile):** [`AverageDrawdown`](@ref), [`UlcerIndex`](@ref), [`MaximumDrawdown`](@ref).

**Clustering proxies (Hierarchical only):** [`EqualRisk`](@ref), [`RiskRatio`](@ref), and all `Relative…` drawdown forms.

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
