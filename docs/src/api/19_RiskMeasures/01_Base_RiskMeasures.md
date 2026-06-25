# Base Risk Measures

All risk measures are defined as their whole names, however this can be unwieldy, so we also provide convenience aliases defined in [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/24_Aliases).

All concrete risk measures can be used as functors (callable structs) to compute their associated risk quantity, according to its [`risk_input_kind`](@ref), or via [`expected_risk`](@ref).

## Quick-pick guide

Three legal-usage classes:

- Optimisation: JuMP formulation; usable as objective/constraint.
- Hierarchical: clustering proxy, no JuMP formulation.
- Non-Optimisation: analysis only.

| Type       | Name (Alias)                                                                 | Compatibility       | JuMP Expr     | Solver        |
|:---------- |:---------------------------------------------------------------------------- |:------------------- |:--------------|:------------- |
| Dispersion | [`Variance`](@ref)                                                           | JuMP + Hierarchical | QuadExpr, SOC | Clarabel      |
| Dispersion | [`StandardDeviation`](@ref) ([`SD`](@ref))                                   | JuMP + Hierarchical | SOC           | Clarabel      |
| Dispersion | Box [`UncertaintySetVariance`](@ref) ([`UcVariance`](@ref))                  | JuMP + Hierarchical | Sym           | Clarabel      |
| Dispersion | Ellipse [`UncertaintySetVariance`](@ref) ([`UcVariance`](@ref))              | JuMP + Hierarchical | Sym, PSD, SOC | Clarabel, SCS |
| Dispersion | [`LowOrderMoment`](@ref) ([`FLM`](@ref))                                     | JuMP + Hierarchical | -             | Clarabel      |
| Dispersion | [`LowOrderMoment`](@ref) ([`MAD`](@ref))                                     | JuMP + Hierarchical | -             | Clarabel      |
| Dispersion | Direct / Squared SOC [`LowOrderMoment`](@ref) ([`SCM`](@ref), [`SLM`](@ref)) | JuMP + Hierarchical | QuadExpr, SOC | Clarabel      |
| Dispersion | SOC [`LowOrderMoment`](@ref) ([`SCM`](@ref), [`SLM`](@ref))                  | JuMP + Hierarchical | SOC           | Clarabel      |
| Dispersion | Sum of Squares [`LowOrderMoment`](@ref) ([`SCM`](@ref), [`SLM`](@ref))       | JuMP + Hierarchical | RSOC          | Clarabel      |
| Dispersion | [`LowOrderMoment`](@ref) ([`ECM`](@ref), [`ELM`](@ref))                      | JuMP + Hierarchical | PowerCone     | Clarabel      |
| Dispersion | [`HighOrderMoment`](@ref) ([`ECM`](@ref), [`ELM`](@ref))                     | Hierarchical        | -             | -             |

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
