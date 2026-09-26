```@meta
Description = "Scoped configuration, private API of PortfolioOptimisers.jl: RESOURCE_LIMITS, COMPACT_SHOW, SHOW_NOTHING_FIELDS, STRING_DISTANCE, EQUATION_LIMITS, …"
```

# Scoped configuration: private API

Five settings apply to the whole package. Each is in a [`ScopedConfig`](@ref), which many threads can read at once.

- [`COMPACT_SHOW`](@ref) sets how far the printout collapses a large nested type.
- [`SHOW_NOTHING_FIELDS`](@ref) sets whether the printout shows a field that holds `nothing`.
- [`STRING_DISTANCE`](@ref) sets how close a misspelled name must be for an error to suggest a correction.
- [`EQUATION_LIMITS`](@ref) sets the largest equation that the parser accepts.
- [`RESOURCE_LIMITS`](@ref) sets the largest draw counts, subset counts and grids that the sampling and sweep estimators accept.

A `set_*!` function replaces the global default in one atomic step. A `with_*` function changes the setting for the duration of one call, and restores it after. The tasks that the call starts see the change too. You can also set the defaults of a project at load time through Preferences.jl.

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
