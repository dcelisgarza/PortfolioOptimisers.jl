```@meta
Description = "Mean tests, public API of PortfolioOptimisers.jl: AbstractMeanTestEstimator, OverlapMeanTest, WildBootstrapMeanTest, AbstractLongRunKernel, BartlettKernel, QuadraticSpectralKernel, …"
```

# Mean tests

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

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
