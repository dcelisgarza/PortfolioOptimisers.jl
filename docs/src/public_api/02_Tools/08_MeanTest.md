```@meta
Description = "Mean test, public API of PortfolioOptimisers.jl: AbstractMeanTestEstimator, OverlapMeanTest, WildBootstrapMeanTest, mean_test, mean_test_rng, …"
```

# Mean test

## Tests of a zero mean

```@docs
AbstractMeanTestEstimator
OverlapMeanTest
WildBootstrapMeanTest
mean_test
mean_test_rng
```

## Long-run kernels

```@docs
AbstractLongRunKernel
BartlettKernel
QuadraticSpectralKernel
long_run_kernel_weight
```

## Span rules

```@docs
AbstractMeanTestSpan
FiniteEntrySpan
FullGridSpan
mean_test_span_length
```

## Bootstrap draws

```@docs
AbstractBootstrapDraws
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
