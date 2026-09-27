```@meta
Description = "Investable universe, public API of PortfolioOptimisers.jl: optimise, _optimise, port_opt_view."
```

# Investable universe

```@docs
optimise(opt::OptimisationResult, args...; kwargs...)
optimise(opt::OptimisationEstimator, args...; kwargs...)
_optimise
optimise(td::TD_OptE_Opt, args...; kwargs...)
port_opt_view(opt::AbstractOptimisationEstimator, ::Any, args...)
port_opt_view(res::NonFiniteAllocationOptimisationResult, ::Colon, args...)
```
