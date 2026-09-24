```@meta
Description = "JuMP Optimiser (a), public API of PortfolioOptimisers.jl: ProcessedJuMPOptimiserAttributes, JuMPOptimisationResult, JuMPOptimiser, factory, port_opt_view, …"
```

# JuMP Optimiser (a)

```@docs
ProcessedJuMPOptimiserAttributes
JuMPOptimisationResult
JuMPOptimiser
factory(opt::JuMPOptimiser, w::AbstractVector)
port_opt_view(opt::JuMPOptimiser, i, X::MatNum, args...)
needs_previous_weights(opt::JuMPOptimiser)
```
