```@meta
Description = "Online selection rules: the second set, public API of PortfolioOptimisers.jl: AbstractConfidenceUpdate, VarianceUpdate, StandardDeviationUpdate, …"
```

# Online selection rules: the second set

This page has more rules with a closed-form update, and the weightings that an `ExpertMixture` can use over its experts. `ConfidenceWeightedMeanReversion` treats the weights as a Gaussian belief. It moves that belief as little as possible, so that the return of the period just observed is below `eps` with a given confidence. It has two formulations, `VarianceUpdate` and `StandardDeviationUpdate`. `AntiCorrelation` moves wealth from an asset to another whose recent growth lagged it and whose latest returns correlate with the earlier returns of the first asset. `ExpectationMaximisation`, the expectation-maximisation update in its online Soft-Bayes form, moves the allocation a share `eta` of the way to the wealth held at the end of the period. `AggregatingAlgorithm`, `TopK` and `WeakAggregatingAlgorithm` weight experts. `AggregatingExponentialGradient` builds an expert mixture of exponentiated gradient rules under the weak aggregating algorithm.

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
