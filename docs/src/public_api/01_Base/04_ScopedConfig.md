```@meta
Description = "Scoped config, public API of PortfolioOptimisers.jl: Base.getindex, set_resource_limits!, with_resource_limits, set_compact_show!, with_compact_show, …"
```

# Scoped config

A `set_*!` function replaces the global default of one setting of the package in one atomic step. A `with_*` function changes the setting for the duration of one call, and restores it after. The tasks that the call starts see the change too. The functions are public and not exported, so call them as `PortfolioOptimisers.set_resource_limits!`.

```@docs
Base.getindex(cfg::ScopedConfig)
PortfolioOptimisers.set_resource_limits!
PortfolioOptimisers.with_resource_limits
PortfolioOptimisers.set_compact_show!
PortfolioOptimisers.with_compact_show
PortfolioOptimisers.set_show_nothing_fields!
PortfolioOptimisers.with_show_nothing_fields
PortfolioOptimisers.set_string_distance!
PortfolioOptimisers.with_string_distance
PortfolioOptimisers.set_equation_limits!
PortfolioOptimisers.with_equation_limits
PortfolioOptimisers.set_log_limits!
PortfolioOptimisers.with_log_limits
```
