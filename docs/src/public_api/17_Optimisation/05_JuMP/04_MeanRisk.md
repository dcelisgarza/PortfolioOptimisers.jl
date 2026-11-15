```@meta
Description = "Mean Risk, public API of PortfolioOptimisers.jl: MeanRiskResult, MeanRisk, factory, Base.getproperty, port_opt_view, _optimise, needs_previous_weights."
```

# Mean Risk

```@docs
MeanRiskResult
MeanRisk
factory(res::MeanRiskResult, fb::Option{<:OptE_Opt_FbChain})
Base.getproperty(r::MeanRiskResult, sym::Symbol)
factory(mr::MeanRisk, w::AbstractVector)
port_opt_view(mr::MeanRisk, i, X::MatNum, args...)
_optimise(mr::MeanRisk, rd::ReturnsResult = ReturnsResult())
needs_previous_weights(opt::MeanRisk)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
