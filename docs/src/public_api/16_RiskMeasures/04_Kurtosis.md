```@meta
Description = "Kurtosis, public API of PortfolioOptimisers.jl: Kurtosis, factory, port_opt_view, supports_precomputed_returns."
```

# Kurtosis

```@docs
Kurtosis
factory(r::Kurtosis, pr::HighOrderPrior, args...; kwargs...)
factory(r::Kurtosis, pr::LowOrderPrior, args...; kwargs...)
port_opt_view(r::Kurtosis, i, args...)
supports_precomputed_returns(r::Kurtosis)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
