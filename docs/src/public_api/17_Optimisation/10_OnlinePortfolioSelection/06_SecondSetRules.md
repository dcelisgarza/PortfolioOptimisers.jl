```@meta
Description = "Online selection rules: the second set, public API of PortfolioOptimisers.jl: AbstractConfidenceUpdate, VarianceUpdate, StandardDeviationUpdate, …"
```

# Online selection rules: the second set

The closed-form rules beyond the prototype and the weightings over an Expert Mixture's experts: the confidence weighted mean reversion with its two formulations, the anti-correlation wealth transfer, the expectation-maximisation update in its Soft-Bayes online form, the aggregating algorithm, the top-k selection, the weak aggregating algorithm, and the aggregation of exponentiated-gradient experts it constructs.

```@docs
PortfolioOptimisers.AbstractConfidenceUpdate
VarianceUpdate
StandardDeviationUpdate
PortfolioOptimisers.confidence_step
PortfolioOptimisers.confidence_gain
ConfidenceWeightedMeanReversion
AntiCorrelation
ExpectationMaximisation
AggregatingAlgorithm
TopK
WeakAggregatingAlgorithm
AggregatingExponentialGradient
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
