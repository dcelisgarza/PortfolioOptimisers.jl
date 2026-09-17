```@meta
Description = "Scoped configuration, private API of PortfolioOptimisers.jl: RESOURCE_LIMITS, COMPACT_SHOW, SHOW_NOTHING_FIELDS, STRING_DISTANCE, EQUATION_LIMITS, …"
```

# Scoped configuration: private API

Package-level configuration values (pretty-printing collapse, fuzzy-suggestion distance, equation-parser resource caps, the scenario-fill share) are held in thread-safe [`ScopedConfig`](@ref) holders: a `set_*!` setter swaps the global default atomically, a `with_*` helper overrides it for the dynamic extent of a call (task-scoped, automatically restored), and per-project defaults can be seeded at load time via Preferences.jl.

```@docs
RESOURCE_LIMITS
COMPACT_SHOW
SHOW_NOTHING_FIELDS
STRING_DISTANCE
EQUATION_LIMITS
ScopedConfig
ResourceLimits
ShowNothingFields
StringDistanceConfig
EquationLimits
set_default!
with_config
set_resource_limits!
with_resource_limits
set_compact_show!
with_compact_show
compact_show_budget
set_show_nothing_fields!
with_show_nothing_fields
set_string_distance!
with_string_distance
set_equation_limits!
with_equation_limits
```
