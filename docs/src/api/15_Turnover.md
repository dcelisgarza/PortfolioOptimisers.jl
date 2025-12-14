# Turnover

The turnover is used to measure the absolute weight change between the current weights and benchmark weights. They can be used to compute fees, or as a constraint. It can also be used as a risk measure, but we will detail that use in [Risk Measures](./18_RiskMeasures/17_TurnoverRiskMeasure.md)

```@docs
TurnoverEstimator
Turnover
factory(tn::Turnover, w::VecNum)
factory(tn::TurnoverEstimator, w::VecNum)
turnover_constraints
turnover_view
TnE_Tn
VecTnE_Tn
VecTn
Tn_VecTn
TnE_Tn_VecTnE_Tn
```
