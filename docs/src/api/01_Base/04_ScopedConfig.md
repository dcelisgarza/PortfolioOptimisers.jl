# Scoped configuration

## Configuration

Package-level configuration values (pretty-printing collapse, fuzzy-suggestion distance, equation-parser resource caps, the scenario-fill share) are held in thread-safe [`ScopedConfig`](@ref) holders: a `set_*!` setter swaps the global default atomically, a `with_*` helper overrides it for the dynamic extent of a call (task-scoped, automatically restored), and per-project defaults can be seeded at load time via Preferences.jl.

```@docs
ScopedConfig
Base.getindex(cfg::ScopedConfig)
set_default!
with_config
RESOURCE_LIMITS
ResourceLimits
set_resource_limits!
with_resource_limits
```

## Pretty printing

```@docs
set_compact_show!
with_compact_show
COMPACT_SHOW
compact_show_budget
ShowNothingFields
SHOW_NOTHING_FIELDS
set_show_nothing_fields!
with_show_nothing_fields
```

## Logging

Functionality for logging messages.

```@docs
StringDistanceConfig
STRING_DISTANCE
set_string_distance!
with_string_distance
EquationLimits
EQUATION_LIMITS
set_equation_limits!
with_equation_limits
```
