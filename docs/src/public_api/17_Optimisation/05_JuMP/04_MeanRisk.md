```@meta
Description = "Mean Risk, public API of PortfolioOptimisers.jl: MeanRiskResult, MeanRisk, factory, Base.getproperty, port_opt_view, optimise, needs_previous_weights."
```

# Mean Risk

```@docs
MeanRiskResult
MeanRisk
factory(res::MeanRiskResult, fb::Option{<:OptE_Opt_FbChain})
Base.getproperty(r::MeanRiskResult, sym::Symbol)
factory(mr::MeanRisk, w::AbstractVector)
port_opt_view(mr::MeanRisk, i, X::MatNum, args...)
optimise(mr::MeanRisk{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
needs_previous_weights(opt::MeanRisk)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
