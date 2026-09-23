```@meta
Description = "Asset turnover, public API of PortfolioOptimisers.jl: TurnoverEstimator, Turnover, factory, turnover_constraints, port_opt_view, needs_previous_weights."
```

# Asset turnover

The turnover measures the absolute change of each weight from a set of reference weights, such as the weights before a rebalance. You can use it to compute fees or as a constraint. It is also a risk measure, which the [turnover risk measure](./16_RiskMeasures/15_TurnoverRiskMeasure.md) page describes.

```@docs
TurnoverEstimator
Turnover
factory(tn::Turnover, w::VecNum)
factory(tn::TurnoverEstimator, w::VecNum)
factory(tn::VecTnE_Tn, w::VecNum)
turnover_constraints
port_opt_view(tn::VecTnE_Tn, i, args...)
needs_previous_weights(tn::TnE_Tn)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
