```@meta
Description = "Scoped config, private API of PortfolioOptimisers.jl: RESOURCE_LIMITS, COMPACT_SHOW, SHOW_NOTHING_FIELDS, STRING_DISTANCE, EQUATION_LIMITS, ScopedConfig, …"
```

# Scoped config: private API

Six settings apply to the whole package. Each is in a [`ScopedConfig`](@ref), which many threads can read at once.

- [`COMPACT_SHOW`](@ref) sets how far the printout collapses a large nested type.
- [`SHOW_NOTHING_FIELDS`](@ref) sets whether the printout shows a field that holds `nothing`.
- [`STRING_DISTANCE`](@ref) sets how close a misspelled name must be for an error to suggest a correction.
- [`EQUATION_LIMITS`](@ref) sets the largest equation that the parser accepts.
- [`RESOURCE_LIMITS`](@ref) sets the largest draw counts, subset counts and grids that the sampling and sweep estimators accept.
- [`LOG_LIMITS`](@ref) sets how many times one log site repeats one message, for each log level.

The `set_*!` and `with_*` functions that change these settings are public, see [Scoped config](@ref "Scoped config"). You can also set the defaults of a project at load time through Preferences.jl.

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
compact_show_budget
LOG_LIMITS
LogLimits
log_limit
@log_id
```
