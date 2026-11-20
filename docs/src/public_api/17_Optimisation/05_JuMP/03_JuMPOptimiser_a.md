```@meta
Description = "JuMP Optimiser (a), public API of PortfolioOptimisers.jl: ResolvedJuMPOptimiser, JuMPOptimisationResult, JuMPOptimiser, factory, port_opt_view, …"
```

# JuMP Optimiser (a)

```@docs
ResolvedJuMPOptimiser
JuMPOptimisationResult
JuMPOptimiser
factory(opt::JuMPOptimiser, w::AbstractVector)
port_opt_view(opt::JuMPOptimiser, i, X::MatNum, args...)
needs_previous_weights(opt::JuMPOptimiser)
obs_weights_view(opt::JuMPOptimiser, i)
```
