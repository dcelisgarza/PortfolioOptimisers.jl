"""
$(DocStringExtensions.TYPEDEF)

Thread-safe holder for a package-level configuration value. It holds a global default that persists, and a task-scoped override.

A read is `cfg[]`. Inside a `with_*` block it returns the innermost active scoped override, and in other code it returns the global default. The default is an `@atomic` field, and a `set_*!` call replaces the whole value in one atomic store. A concurrent reader thus never sees a configuration that is half written. The `FLoops.@floop` loops inside the meta-optimisers are such readers. The scoped override is a `Base.ScopedValues.ScopedValue`. A task spawned inside the scope gets the override, the override goes away when the scope exits, and a concurrent task outside the scope does not see it.

Each configuration of this kind holds an immutable struct or a bits value. A change to one setting builds a new value and stores it in place of the old value. No call mutates the value in place.

[`COMPACT_SHOW`](@ref), [`SHOW_NOTHING_FIELDS`](@ref), [`STRING_DISTANCE`](@ref), [`EQUATION_LIMITS`](@ref), [`RESOURCE_LIMITS`](@ref) and [`LOG_LIMITS`](@ref) are of this type. A `set_*!` function sets the global default of each, and a `with_*` function sets a scoped override. Preferences.jl sets a default for each project at load time, through [`apply_preferences!`](@ref).

# Related

  - [`set_compact_show!`](@ref) / [`with_compact_show`](@ref)
  - [`set_show_nothing_fields!`](@ref) / [`with_show_nothing_fields`](@ref)
  - [`set_string_distance!`](@ref) / [`with_string_distance`](@ref)
  - [`set_equation_limits!`](@ref) / [`with_equation_limits`](@ref)
  - [`set_resource_limits!`](@ref) / [`with_resource_limits`](@ref)
  - [`set_log_limits!`](@ref) / [`with_log_limits`](@ref)
  - [`apply_preferences!`](@ref)
"""
mutable struct ScopedConfig{T}
    @atomic default::T
    const scoped::ScopedValue{Union{Nothing, T}}
    function ScopedConfig{T}(x) where {T}
        return new{T}(convert(T, x), ScopedValue{Union{Nothing, T}}(nothing))
    end
end
ScopedConfig(x::T) where {T} = ScopedConfig{T}(x)
"""
    getindex(cfg::ScopedConfig)

Read the active value of a [`ScopedConfig`](@ref). Inside a `with_*` block it is the innermost task-scoped override. In other code it is the global default, which the function reads atomically.

# Algorithm

 1. Read `cfg.scoped[]`, the innermost task-scoped override. It is `nothing` outside every `with_*` block.
 2. When step 1 gives `nothing`, read `cfg.default` atomically.

# Returns

  - `x::T`: The active value of the configuration.

# Related

  - [`ScopedConfig`](@ref)
  - [`set_default!`](@ref)
  - [`with_config`](@ref)
"""
Base.getindex(cfg::ScopedConfig) = @something(cfg.scoped[], @atomic(cfg.default))
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace the global default of a [`ScopedConfig`](@ref) with `x` in one atomic store, and return `x`. An active scoped override does not change.

# Algorithm

 1. Convert `x` to `T`, the element type of `cfg`. A failed conversion raises before the function stores a value.
 2. Store the converted `x` into `cfg.default` in one atomic write. A concurrent reader sees the old value or the new value, and never a partial write.

# Arguments

  - `cfg`: The configuration holder whose global default the function replaces.
  - `x`: The new value. The function converts it to the element type `T` of `cfg`.

# Returns

  - `x::T`: The stored value.

# Related

  - [`ScopedConfig`](@ref)
  - [`with_config`](@ref)
"""
function set_default!(cfg::ScopedConfig{T}, x) where {T}
    x = convert(T, x)
    @atomic cfg.default = x
    return x
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run `f()` with the value of the [`ScopedConfig`](@ref) `cfg` set to `x` for the dynamic extent of the call. The previous value comes back when the call exits. The override is task-scoped, so it is thread-safe. A task spawned inside `f` gets the override, and a concurrent task outside `f` does not see it.

# Algorithm

 1. Convert `x` to `T`, the element type of `cfg`.
 2. Bind `cfg.scoped` to the converted `x`, and run `f()` inside that binding. The binding ends when `f` returns and when `f` raises.

# Arguments

  - `f`: The zero-argument function to run under the override.
  - `cfg`: The configuration holder to override.
  - `x`: The override value. The function converts it to the element type `T` of `cfg`.

# Returns

  - The value that `f()` returns.

# Related

  - [`ScopedConfig`](@ref)
  - [`set_default!`](@ref)
"""
function with_config(f, cfg::ScopedConfig{T}, x) where {T}
    return Base.ScopedValues.with(f, cfg.scoped => convert(T, x))
end
"""
Global control for the collapse of large nested structs in the output of [`@define_pretty_show`](@ref).

It holds one of these values:

  - `false`: The collapse is off, and a nested struct always prints in full.
  - `true`: The collapse is on, with a line budget that the size of the terminal sets.
  - `n::Int`: The collapse is on, with a fixed line budget of `n`.

It is a [`ScopedConfig`](@ref). [`set_compact_show!`](@ref) sets the global default, and [`with_compact_show`](@ref) sets an override for one scope. [`compact_show_budget`](@ref) reads it together with the per-call `:po_compact` IO property. The `"compact_show"` preference can set the default for a project at load time, through [`apply_preferences!`](@ref).
"""
const COMPACT_SHOW = ScopedConfig{Union{Bool, Int}}(true)
"""
    set_compact_show!(x::Bool)
    set_compact_show!(n::Integer)

Configure whether [`@define_pretty_show`](@ref) collapses large nested structs.

  - `set_compact_show!(false)`: Turn the collapse off, so that each struct prints in full.
  - `set_compact_show!(true)`: Turn the collapse on, with a budget that the size of the terminal sets.
  - `set_compact_show!(n)`: Turn the collapse on, with a fixed line budget `n`.

The collapse applies only to output with a limited height, where `get(io, :limit, false)` is `true`. The interactive REPL is such output. Output with no limit, such as `string`, `repr` and a write to a file, always prints in full. The documentation build turns the collapse off, so that the rendered docs show each field. A single call can override the global setting with the `:po_compact` IO property, which takes `false`, `true` or an `Int`.

The function sets the global default in one atomic store, as [`ScopedConfig`](@ref) states. For a temporary, task-scoped override, use [`with_compact_show`](@ref).

# Algorithm

 1. Widen an `Integer` argument to `Int`. The two methods then store one of the two members of the stored union, and never a third integer type. A `Bool` does not change.
 2. Store the value as the global default of [`COMPACT_SHOW`](@ref) through [`set_default!`](@ref).

# Arguments

  - `x::Bool`: `false` turns the collapse off. `true` turns it on, with the automatic budget.
  - `n::Integer`: The fixed line budget, which the function stores as an `Int`.

# Returns

  - `x::Union{Bool, Int}`: The stored setting.

# Related

  - [`@define_pretty_show`](@ref)
  - [`COMPACT_SHOW`](@ref)
  - [`compact_show_budget`](@ref)
  - [`with_compact_show`](@ref)
"""
set_compact_show!(x::Bool) = set_default!(COMPACT_SHOW, x)
set_compact_show!(n::Integer) = set_default!(COMPACT_SHOW, Int(n))
"""
    with_compact_show(f, x::Bool)
    with_compact_show(f, n::Integer)

Run `f()` with the collapse setting of [`COMPACT_SHOW`](@ref) set to `x` or `n` for the dynamic extent of the call. The previous setting comes back when the call exits. The override is task-scoped and thread-safe, as [`ScopedConfig`](@ref) states, and the global default does not change.

# Algorithm

 1. Widen an `Integer` argument to `Int`, as [`set_compact_show!`](@ref) does. A `Bool` does not change.
 2. Run `f()` with [`COMPACT_SHOW`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: The zero-argument function to run under the override.
  - `x::Bool`: `false` turns the collapse off. `true` turns it on, with the automatic budget.
  - `n::Integer`: The fixed line budget, which the function stores as an `Int`.

# Returns

  - The value that `f()` returns.

# Related

  - [`COMPACT_SHOW`](@ref)
  - [`set_compact_show!`](@ref)
  - [`compact_show_budget`](@ref)
"""
with_compact_show(f, x::Bool) = with_config(f, COMPACT_SHOW, x)
with_compact_show(f, n::Integer) = with_config(f, COMPACT_SHOW, Int(n))
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the line budget above which [`@define_pretty_show`](@ref) collapses a nested struct.

The per-call `:po_compact` IO property overrides the global [`COMPACT_SHOW`](@ref) setting. Both take `false` for no collapse, `true` for the automatic budget, or an `Int` for a fixed budget. The automatic budget is `max(8, displaysize(io)[1] - 4)`, so only a subtree that almost fills the terminal, or is taller than it, collapses.

# Algorithm

 1. Read `v`, the `:po_compact` property of `io`.
 2. When `io` sets no such property, the collapse applies only to output with a limited height. Return `nothing` when `:limit` is `false`. Otherwise read the global [`COMPACT_SHOW`](@ref) setting into `v`.
 3. Return `nothing` when `v` is `false`, because that value turns the collapse off.
 4. Return `Int(v)` when `v` is an integer that is not a `Bool`, because that value is the fixed budget.
 5. Otherwise `v` is `true`, so return the automatic budget `max(8, displaysize(io)[1] - 4)`.

# Arguments

  - `io`: The stream whose `:po_compact`, `:limit` and `:displaysize` properties the function reads.

# Returns

  - `nothing` when the collapse is off.
  - `budget::Int` in the other cases. It is the maximum number of lines that a nested struct can print before it collapses.

# Related

  - [`set_compact_show!`](@ref)
  - [`@define_pretty_show`](@ref)
"""
function compact_show_budget(io::IO)
    v = get(io, :po_compact, :__unset__)
    if v === :__unset__
        # No per-call override: only collapse height-limited output (the REPL),
        # leaving `string`/`repr`/file writes fully expanded.
        if !(get(io, :limit, false))
            return nothing
        end
        v = COMPACT_SHOW[]
    end
    if v === false
        return nothing
    end
    if v isa Integer && !(v isa Bool)
        return Int(v)
    end
    return max(8, displaysize(io)[1] - 4)
end
"""
$(DocStringExtensions.TYPEDEF)

Configuration for whether [`@define_pretty_show`](@ref) renders a field that holds `nothing`.

The [`ScopedConfig`](@ref) [`SHOW_NOTHING_FIELDS`](@ref) holds it, and [`pretty_show_fields`](@ref) reads it. [`set_show_nothing_fields!`](@ref) sets the global default, and [`with_show_nothing_fields`](@ref) sets an override for one scope. No call mutates a value. A change builds a new value with a copy of `by_type`, and stores it in place of the old value. A reader thus never sees a table that is half written.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShowNothingFields(default::Bool, by_type::Dict{Symbol, Bool}) -> ShowNothingFields
    ShowNothingFields(cfg::ShowNothingFields, name::Symbol,
                      x::Union{Nothing, Bool}) -> ShowNothingFields

The second form copies `cfg` and changes one per-name entry. A `Bool` `x` sets the entry for `name`, and `nothing` removes it.

# Related

  - [`SHOW_NOTHING_FIELDS`](@ref)
  - [`set_show_nothing_fields!`](@ref)
  - [`with_show_nothing_fields`](@ref)
  - [`pretty_show_fields`](@ref)
  - [`show_fields`](@ref)
  - [`ScopedConfig`](@ref)
"""
struct ShowNothingFields
    """
    Whether a field that holds `nothing` prints, for a type that no per-name entry names. The package default is `false`, which hides a `nothing` field.
    """
    default::Bool
    """
    The per-name overrides. The key is the name of a type without its module. A `true` entry prints every declared field of that type, and a `false` entry hides its `nothing` fields. An entry overrides `default` and the [`show_fields`](@ref) method of the type.
    """
    by_type::Dict{Symbol, Bool}
end
function ShowNothingFields(cfg::ShowNothingFields, name::Symbol, x::Union{Nothing, Bool})
    by_type = copy(cfg.by_type)
    if isnothing(x)
        delete!(by_type, name)
    else
        by_type[name] = x
    end
    return ShowNothingFields(cfg.default, by_type)
end
"""
    SHOW_NOTHING_FIELDS = ScopedConfig(ShowNothingFields(false, Dict{Symbol, Bool}()))

Global control for whether [`@define_pretty_show`](@ref) renders a field that holds `nothing`.

The package default hides such a field, because a reader at the REPL wants the fields that hold a value. The documentation build shows these fields with `set_show_nothing_fields!(true)`, next to its `set_compact_show!(false)` call, so that a rendered docstring shows the complete type. [`pretty_show_fields`](@ref) reads `SHOW_NOTHING_FIELDS[]`. The `"show_nothing_fields"` and `"show_nothing_fields_by_type"` preferences can set the default for a project at load time, through [`apply_preferences!`](@ref).

# Related

  - [`ShowNothingFields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
  - [`with_show_nothing_fields`](@ref)
  - [`pretty_show_fields`](@ref)
  - [`show_fields`](@ref)
  - [`COMPACT_SHOW`](@ref)
"""
const SHOW_NOTHING_FIELDS = ScopedConfig(ShowNothingFields(false, Dict{Symbol, Bool}()))
"""
    set_show_nothing_fields!(x::Bool)
    set_show_nothing_fields!(name::Symbol, x::Union{Nothing, Bool})

Configure whether [`@define_pretty_show`](@ref) renders a field that holds `nothing`.

  - `set_show_nothing_fields!(false)`: Hide a `nothing` field, for every type that no per-name entry names. This is the package default.
  - `set_show_nothing_fields!(true)`: Print a `nothing` field, for every type that no per-name entry names.
  - `set_show_nothing_fields!(:SimpleVariance, false)`: Hide the `nothing` fields of every type with the name `SimpleVariance`, whatever the global switch holds.
  - `set_show_nothing_fields!(:SimpleVariance, true)`: Print every declared field of every type with the name `SimpleVariance`, also a field that its own [`show_fields`](@ref) method hides.
  - `set_show_nothing_fields!(:SimpleVariance, nothing)`: Remove the per-name entry, so that the type again obeys the global switch and its own [`show_fields`](@ref) method.

The per-name form takes a name and not a type, because the preferences that load at start are TOML, and TOML holds no types. The name is the name of the type without its module, so two types with one name in two modules share one entry. The function accepts a name that matches no type, and the entry then has no effect. A type outside the package can also print through [`@define_pretty_show`](@ref), so the function cannot refuse an unknown name.

The function sets the global default in one atomic store, as [`ScopedConfig`](@ref) states. For a temporary, task-scoped override, use [`with_show_nothing_fields`](@ref).

# Algorithm

 1. Read the current global default of [`SHOW_NOTHING_FIELDS`](@ref) atomically. The function does not read a scoped override, so the call changes the global default alone.
 2. Build the new [`ShowNothingFields`](@ref). The one-argument form keeps the per-name table and replaces the global switch. The two-argument form keeps the global switch, and copies the table with one entry set or removed.
 3. Store it as the global default through [`set_default!`](@ref).

# Arguments

  - `x::Bool`: `false` hides a `nothing` field, and `true` prints it.
  - `name::Symbol`: The name of a type, without its module.
  - `x::Union{Nothing, Bool}`: In the per-name form, the entry to set, or `nothing` to remove the entry.

# Returns

  - `cfg::ShowNothingFields`: The new global default.

# Related

  - [`@define_pretty_show`](@ref)
  - [`SHOW_NOTHING_FIELDS`](@ref)
  - [`ShowNothingFields`](@ref)
  - [`with_show_nothing_fields`](@ref)
  - [`pretty_show_fields`](@ref)
  - [`set_compact_show!`](@ref)
"""
function set_show_nothing_fields!(x::Bool)
    cfg = @atomic SHOW_NOTHING_FIELDS.default
    return set_default!(SHOW_NOTHING_FIELDS, ShowNothingFields(x, cfg.by_type))
end
function set_show_nothing_fields!(name::Symbol, x::Union{Nothing, Bool})
    cfg = @atomic SHOW_NOTHING_FIELDS.default
    return set_default!(SHOW_NOTHING_FIELDS, ShowNothingFields(cfg, name, x))
end
"""
    with_show_nothing_fields(f, x::Bool)
    with_show_nothing_fields(f, name::Symbol, x::Union{Nothing, Bool})

Run `f()` with an override of the [`SHOW_NOTHING_FIELDS`](@ref) setting for the dynamic extent of the call. The previous setting comes back when the call exits. The override is task-scoped and thread-safe, as [`ScopedConfig`](@ref) states, and the global default does not change. The part of the setting that the call does not name comes from the active value, so a nested override adds to the override around it.

# Algorithm

 1. Read the active value of [`SHOW_NOTHING_FIELDS`](@ref). A nested override then starts from the override around it, and not from the global default.
 2. Build the override as [`set_show_nothing_fields!`](@ref) does. The one-argument form replaces the global switch and keeps the per-name table. The two-argument form keeps the global switch, and copies the table with one entry set or removed.
 3. Run `f()` with [`SHOW_NOTHING_FIELDS`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: The zero-argument function to run under the override.
  - `x::Bool`: `false` hides a `nothing` field, and `true` prints it.
  - `name::Symbol`: The name of a type, without its module.
  - `x::Union{Nothing, Bool}`: In the per-name form, the entry to set, or `nothing` to remove the entry.

# Returns

  - The value that `f()` returns.

# Related

  - [`SHOW_NOTHING_FIELDS`](@ref)
  - [`ShowNothingFields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
  - [`pretty_show_fields`](@ref)
  - [`with_config`](@ref)
  - [`with_compact_show`](@ref)
"""
function with_show_nothing_fields(f, x::Bool)
    cfg = SHOW_NOTHING_FIELDS[]
    return with_config(f, SHOW_NOTHING_FIELDS, ShowNothingFields(x, cfg.by_type))
end
function with_show_nothing_fields(f, name::Symbol, x::Union{Nothing, Bool})
    cfg = SHOW_NOTHING_FIELDS[]
    return with_config(f, SHOW_NOTHING_FIELDS, ShowNothingFields(cfg, name, x))
end
"""
$(DocStringExtensions.TYPEDEF)

Global configuration for the fuzzy "did you mean?" suggestions that [`did_you_mean`](@ref) adds to a "variable not in asset universe" message.

The value is immutable. The [`ScopedConfig`](@ref) [`STRING_DISTANCE`](@ref) holds it, and [`did_you_mean`](@ref) reads it. [`set_string_distance!`](@ref) sets the global default, and [`with_string_distance`](@ref) sets an override for one scope.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StringDistanceConfig(dist::StringDistances.StringDistance,
                         min_score::Real) -> StringDistanceConfig

## Validation

  - `min_score > 0`. `StringDistances.findnearest` never suggests a candidate with a score of exactly `0`. But a threshold at or below `0` admits every candidate with a nonzero similarity, so `0` and a negative value have the same effect. Both let a guessed name that is near a real asset name get that real name back, and the suggestion then leaks data.

# Related

  - [`STRING_DISTANCE`](@ref)
  - [`set_string_distance!`](@ref)
  - [`with_string_distance`](@ref)
  - [`did_you_mean`](@ref)
"""
struct StringDistanceConfig
    """
    The distance that scores a candidate name against the unknown name. The default is `StringDistances.Levenshtein()`.
    """
    dist::StringDistances.StringDistance
    """
    The minimum normalised similarity of a candidate that the function suggests. The default is `0.7`. A value near `1` keeps only an almost exact match, and a value above `1` turns all suggestions off. The inner loop of a meta-optimiser needs that value. A cluster or a subset that does not hold an asset name is correct, the name is not a typo, and a suggestion for it is wrong.
    """
    min_score::Float64
    function StringDistanceConfig(dist::StringDistances.StringDistance, min_score::Real)
        @argcheck(min_score > 0,
                  ArgumentError("min_score must be positive; got $(min_score). A value above 1 legitimately disables suggestions, but a zero or negative threshold admits every candidate with any nonzero similarity, making `did_you_mean` echo a real asset name for near-miss probes. A suggestion must never reveal which asset names exist to a caller who guesses."))
        return new(dist, min_score)
    end
end
"""
    STRING_DISTANCE = ScopedConfig(StringDistanceConfig(StringDistances.Levenshtein(), 0.7))

Default string distance configuration for the fuzzy "did you mean?" suggestions that [`did_you_mean`](@ref) adds to a "variable not in asset universe" message. [`did_you_mean`](@ref) reads `STRING_DISTANCE[]`. The `"suggestion_distance"` and `"suggestion_min_score"` preferences can set the defaults for a project at load time, through [`apply_preferences!`](@ref).

# Related

  - [`StringDistanceConfig`](@ref)
  - [`set_string_distance!`](@ref)
  - [`with_string_distance`](@ref)
  - [`did_you_mean`](@ref)
"""
const STRING_DISTANCE = ScopedConfig(StringDistanceConfig(StringDistances.Levenshtein(),
                                                          0.7))
"""
    set_string_distance!(; dist::StringDistances.StringDistance, min_score::Real)

Set the global default of the fuzzy-suggestion settings that [`did_you_mean`](@ref) reads. The store is atomic, as [`ScopedConfig`](@ref) states. A keyword that the caller omits keeps its current default. For a temporary, task-scoped override, use [`with_string_distance`](@ref).

# Algorithm

 1. Give each keyword that the caller omits the value of its field in the current global default, read atomically. The function does not read a scoped override, so the call changes the global default alone.
 2. Build a [`StringDistanceConfig`](@ref) from the two keywords. The validation below runs in that constructor.
 3. Store it as the global default of [`STRING_DISTANCE`](@ref) through [`set_default!`](@ref).

# Arguments

  - `dist::StringDistances.StringDistance`: The distance that ranks the candidate names, for example `StringDistances.Levenshtein()`, `StringDistances.DamerauLevenshtein()` or `StringDistances.JaroWinkler()`.
  - `min_score::Real`: The minimum normalised similarity in `(0, 1]` that gives a suggestion. A value above `1` turns all suggestions off.

# Validation

  - `min_score > 0`. [`StringDistanceConfig`](@ref) checks it. A threshold that is not positive admits every candidate with a nonzero similarity.

# Returns

  - `cfg::StringDistanceConfig`: The new global default.

# Related

  - [`did_you_mean`](@ref)
  - [`STRING_DISTANCE`](@ref)
  - [`with_string_distance`](@ref)
  - [`set_compact_show!`](@ref)
"""
function set_string_distance!(;
                              dist::StringDistances.StringDistance = (@atomic STRING_DISTANCE.default).dist,
                              min_score::Real = (@atomic STRING_DISTANCE.default).min_score)
    return set_default!(STRING_DISTANCE, StringDistanceConfig(dist, min_score))
end
"""
    with_string_distance(f; dist::StringDistances.StringDistance = STRING_DISTANCE[].dist,
                         min_score::Real = STRING_DISTANCE[].min_score)

Run `f()` with an override of the fuzzy-suggestion settings that [`did_you_mean`](@ref) reads, for the dynamic extent of the call. The previous settings come back when the call exits. The override is task-scoped and thread-safe, as [`ScopedConfig`](@ref) states, and the global default does not change. A keyword that the caller omits comes from the active value, so a nested override adds to the override around it.

Put this call around a meta-optimiser run, with `min_score` above `1`, to stop the suggestions in its inner loops. Other concurrent work keeps its suggestions.

# Algorithm

 1. Give each keyword that the caller omits the value of its field in the active value. A nested override then starts from the override around it, and not from the global default.
 2. Build a [`StringDistanceConfig`](@ref) from the two keywords. The validation below runs in that constructor.
 3. Run `f()` with [`STRING_DISTANCE`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: The zero-argument function to run under the override.
  - `dist::StringDistances.StringDistance`: The distance that ranks the candidate names.
  - `min_score::Real`: The minimum normalised similarity in `(0, 1]` that gives a suggestion. A value above `1` turns all suggestions off.

# Validation

  - `min_score > 0`. [`StringDistanceConfig`](@ref) checks it.

# Returns

  - The value that `f()` returns.

# Related

  - [`set_string_distance!`](@ref)
  - [`STRING_DISTANCE`](@ref)
  - [`StringDistanceConfig`](@ref)
  - [`with_config`](@ref)
  - [`did_you_mean`](@ref)
"""
function with_string_distance(f;
                              dist::StringDistances.StringDistance = STRING_DISTANCE[].dist,
                              min_score::Real = STRING_DISTANCE[].min_score)
    return with_config(f, STRING_DISTANCE, StringDistanceConfig(dist, min_score))
end
"""
$(DocStringExtensions.TYPEDEF)

Global resource caps for the parse of an equation. The caps stop a string whose expression tree is deep enough to use all of the stack.

A constraint string, a Black-Litterman view string and an entropy-pooling view string are untrusted input. They can come from a configuration file, a spreadsheet or a user interface. All of them go through [`parse_equation`](@ref), which calls `Meta.parse`. [`eval_numeric_functions`](@ref), `collect_terms!` and `has_invalid_plus` then walk the expression tree recursively. With no bound, a deeply nested string, such as tens of thousands of parentheses, gives a tree that uses all of the stack and stops the Julia process. These caps raise a `Meta.ParseError` long before that point.

The values are fixed defaults with a wide margin. A fixed value is the same on the build machine and on the deployment machine, and a value that precompilation measures is not. The value is immutable. The [`ScopedConfig`](@ref) [`EQUATION_LIMITS`](@ref) holds it. [`set_equation_limits!`](@ref) sets the global default, and [`with_equation_limits`](@ref) sets an override for one scope.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EquationLimits(max_length::Integer, max_depth::Integer) -> EquationLimits

## Validation

  - `max_length > 0 && max_depth > 0`.

# Related

  - [`EQUATION_LIMITS`](@ref)
  - [`set_equation_limits!`](@ref)
  - [`with_equation_limits`](@ref)
  - [`ResourceLimits`](@ref)
  - [`ScopedConfig`](@ref)
  - [`parse_equation`](@ref)
"""
struct EquationLimits
    """
    The maximum number of characters in an equation string that the library passes to `Meta.parse`. The default is `4096`. A real linear constraint is short, so the bound is far above each real constraint and far below the nesting depth that can use all of the stack. A nesting depth `d` needs at least `d` characters, so the length cap also bounds the depth of the tree of the string form.
    """
    max_length::Int
    """
    The maximum depth of an expression tree that the `Expr` form of [`parse_equation`](@ref) accepts. The default is `256`. That form gets a tree that is already built, and no length cap applies to it.
    """
    max_depth::Int
    function EquationLimits(max_length::Integer, max_depth::Integer)
        @argcheck(max_length > 0 && max_depth > 0,
                  ArgumentError("max_length and max_depth must be positive."))
        return new(Int(max_length), Int(max_depth))
    end
end
"""
    EQUATION_LIMITS = ScopedConfig(EquationLimits(4096, 256))

Default global resource caps for the parse of an equation. The caps stop a string whose expression tree is deep enough to use all of the stack. [`parse_equation`](@ref) and each other parser of an equation string read `EQUATION_LIMITS[]`. The `"equation_max_length"` and `"equation_max_depth"` preferences can set the defaults for a project at load time, through [`apply_preferences!`](@ref).

# Related

  - [`EquationLimits`](@ref)
  - [`set_equation_limits!`](@ref)
  - [`with_equation_limits`](@ref)
  - [`parse_equation`](@ref)
"""
const EQUATION_LIMITS = ScopedConfig(EquationLimits(4096, 256))
"""
    set_equation_limits!(; max_length::Integer, max_depth::Integer)

Set the global default of the equation-parser resource caps of [`EQUATION_LIMITS`](@ref).

Raise the caps for a large constraint set that a program wrote. Lower them to accept less untrusted input. A keyword that the caller omits keeps its current default. The store is atomic, as [`ScopedConfig`](@ref) states. For a temporary, task-scoped override, use[`with_equation_limits`](@ref).

# Algorithm

 1. Give each keyword that the caller omits the value of its field in the current global default, read atomically. The function does not read a scoped override, so the call changes the global default alone.
 2. Build an [`EquationLimits`](@ref) from the two keywords. The validation below runs in that constructor.
 3. Store it as the global default of [`EQUATION_LIMITS`](@ref) through [`set_default!`](@ref).

# Arguments

  - `max_length::Integer`: The maximum length of an equation string that the library passes to `Meta.parse`.
  - `max_depth::Integer`: The maximum depth of an expression tree that the `Expr` form of [`parse_equation`](@ref) accepts.

# Validation

  - `max_length > 0 && max_depth > 0`. [`EquationLimits`](@ref) checks it.

# Returns

  - `lims::EquationLimits`: The new global default.

# Related

  - [`EQUATION_LIMITS`](@ref)
  - [`with_equation_limits`](@ref)
  - [`parse_equation`](@ref)
  - [`set_string_distance!`](@ref)
"""
function set_equation_limits!(;
                              max_length::Integer = (@atomic EQUATION_LIMITS.default).max_length,
                              max_depth::Integer = (@atomic EQUATION_LIMITS.default).max_depth)
    return set_default!(EQUATION_LIMITS, EquationLimits(max_length, max_depth))
end
"""
    with_equation_limits(f; max_length::Integer = EQUATION_LIMITS[].max_length,
                         max_depth::Integer = EQUATION_LIMITS[].max_depth)

Run `f()` with an override of the equation-parser resource caps of [`EQUATION_LIMITS`](@ref) for the dynamic extent of the call. The previous caps come back when the call exits. The override is task-scoped and thread-safe, as [`ScopedConfig`](@ref) states, and the global default does not change. A keyword that the caller omits comes from the active value, so a nested override adds to the override around it.

Use it to lower the caps around one batch of untrusted constraint strings, or to raise them for one large constraint set that a program wrote. Other concurrent work keeps its caps.

# Algorithm

 1. Give each keyword that the caller omits the value of its field in the active value. A nested override then starts from the override around it, and not from the global default.
 2. Build an [`EquationLimits`](@ref) from the two keywords. The validation below runs in that constructor.
 3. Run `f()` with [`EQUATION_LIMITS`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: The zero-argument function to run under the override.
  - `max_length::Integer`: The maximum length of an equation string that the library passes to `Meta.parse`.
  - `max_depth::Integer`: The maximum depth of an expression tree that the `Expr` form of [`parse_equation`](@ref) accepts.

# Validation

  - `max_length > 0 && max_depth > 0`. [`EquationLimits`](@ref) checks it.

# Returns

  - The value that `f()` returns.

# Related

  - [`set_equation_limits!`](@ref)
  - [`EQUATION_LIMITS`](@ref)
  - [`EquationLimits`](@ref)
  - [`with_config`](@ref)
  - [`parse_equation`](@ref)
"""
function with_equation_limits(f; max_length::Integer = EQUATION_LIMITS[].max_length,
                              max_depth::Integer = EQUATION_LIMITS[].max_depth)
    return with_config(f, EQUATION_LIMITS, EquationLimits(max_length, max_depth))
end
"""
$(DocStringExtensions.TYPEDEF)

Global resource caps for the estimators that sample or sweep. The caps stop a configuration value that is large enough to use all of the memory, or to keep the process busy for days.

A draw count, a subset count, the length of a frontier sweep and a histogram bin count are untrusted configuration integers. They can come from a configuration file, a tuning grid or a user interface. Each one multiplies the size of an allocation, and a subset count or a frontier length multiplies the number of optimisations. The constructors of these estimators bound the values only from below, with `n_sim > 0`, `n_subsets >= 2`, `N > 0` and `bins > 0`. A value that is much too large, from an extra digit or a sweep with the wrong scale, passes these checks. The out-of-memory killer then stops the process, and the caller does not get a message. These caps raise a `DomainError` at the point where the library resolves the value.

Each allocation or loop that a value drives has its own cap, and the name of the cap follows the field that it checks. The caps are not shared, because a linear cap cannot bound a quadratic allocation. The `bins × bins` histogram has its own [`max_bins`](@ref ResourceLimits), and does not share the linear cap on draws.

The defaults are fixed values far above real use. A cap turns a kill by the out-of-memory killer into an error with a type. It does not judge the size that the caller chose. The value is immutable. The [`ScopedConfig`](@ref) [`RESOURCE_LIMITS`](@ref) holds it. [`set_resource_limits!`](@ref) sets the global default, and [`with_resource_limits`](@ref) sets an override for one scope. Use the keyword constructor `ResourceLimits(; …)`. The seven caps have one type and four of them have the value `100_000`, so a positional call can easily put a value in the wrong place.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ResourceLimits(max_n_sim::Integer, max_n_subsets::Integer, max_frontier::Integer,
                   max_bins::Integer, max_hop_count::Integer, max_search_grid::Integer,
                   max_ep_grid::Integer) -> ResourceLimits

    ResourceLimits(;
        max_n_sim::Integer = 1_000_000,
        max_n_subsets::Integer = 100_000,
        max_frontier::Integer = 100_000,
        max_bins::Integer = 10_000,
        max_hop_count::Integer = 100_000,
        max_search_grid::Integer = 100_000,
        max_ep_grid::Integer = 10_000
    ) -> ResourceLimits

Each keyword sets the field of the same name.

## Validation

  - `max_n_sim > 0 && max_n_subsets > 0 && max_frontier > 0 && max_bins > 0 && max_hop_count > 0 && max_search_grid > 0 && max_ep_grid > 0`.

# Related

  - [`RESOURCE_LIMITS`](@ref)
  - [`set_resource_limits!`](@ref)
  - [`with_resource_limits`](@ref)
  - [`assert_resource_cap`](@ref)
  - [`EquationLimits`](@ref)
"""
struct ResourceLimits
    """
    The maximum number of Monte-Carlo or bootstrap draws, `n_sim`, that [`NormalUncertaintySet`](@ref) and [`ARCHUncertaintySet`](@ref) accept. The default is `1_000_000`. Each draw stores an `N × N` covariance, so the array holds `N² · n_sim` elements. At 20 assets the default cap permits a request of 3.2 GB, and the default `n_sim` is `3_000`. This cap bounds memory.
    """
    max_n_sim::Int
    """
    The maximum number of resampled asset subsets, `n_subsets`, that [`SubsetResampling`](@ref) and [`MultipleRandomised`](@ref) accept. The default is `100_000`. This cap bounds compute much more than memory, because each subset runs a full inner optimisation. The cap is far above a real sweep, where the default is `2`. It is far below a value that keeps a session busy for days.
    """
    max_n_subsets::Int
    """
    The maximum number of points of an efficient-frontier sweep that the [`Frontier`](@ref) algorithm of [`MeanRisk`](@ref) and [`NearOptimalCentering`](@ref) accepts. The default is `100_000`. Each point runs a full inner `optimise_JuMP_model!` solve, so this cap bounds compute, and it has the same value as `max_n_subsets`. The default of [`Frontier`](@ref) is `N = 20`. The library checks this cap at two points. The constructor of [`Frontier`](@ref) caps the `N` of one bound. At Model Assembly, [`assert_frontier_sweep_cap`](@ref) caps the product over every swept return term and every swept risk measure. The sweep is an `Iterators.product`, so `k` bounds of `N` points cost `N^k` solves.
    """
    max_frontier::Int
    """
    The maximum number of histogram bins that [`MutualInfoCovariance`](@ref) and [`VariationInfoDistance`](@ref) accept. The default is `10_000`. The joint histogram is a `bins × bins` weights matrix for each pair of assets, so this cap bounds a quadratic allocation of memory. A histogram of `10_000²` cells takes about 800 MB. That size is below the limit of the out-of-memory killer, and far above the 50 bins, approximately, that a real binning gives.
    """
    max_bins::Int
    """
    The maximum hop count, `n`, that [`HopCount`](@ref) accepts. The default is `100_000`. Three readers sum `A^i` over `i in 0:n`, so the cost is linear in `n`, at `N³` floating point operations for each power. A large `n` keeps the session busy on compute, and not on memory. This cap bounds compute, and it has the same value as `max_n_subsets`. The default hop count is `n = 1`. The constructor of [`HopCount`](@ref) reads the cap, and [`resolve_separation`](@ref) also builds a [`HopCount`](@ref) from the value that a rule computes. One check thus applies to a stated value and to a computed value.
    """
    max_hop_count::Int
    """
    The maximum number of candidates in a search grid that [`GridSearchCrossValidation`](@ref) and [`RandomisedSearchCrossValidation`](@ref) accept. The default is `100_000`. Each candidate runs a full cross-validated fit, so this cap bounds compute, as `max_n_subsets` does. `collect` builds the grid from an `Iterators.product`, so `k` parameters of `N` values give `N^k` candidates. [`assert_search_grid_cap`](@ref) checks the cap on that product where the grid forms, because a check on each parameter cannot see the product.
    """
    max_search_grid::Int
    """
    The maximum number of grid points, `K`, that [`GridTailView`](@ref) accepts. The default is `10_000`. Each grid point is one binary variable of the mixed-integer program that an upper-bound or equality view builds. It is also one dense row over the `T` posterior probabilities. The memory is linear in `K`, at `K · T` coefficients, and the compute is a branch-and-bound tree over `K` binary variables. The cap is far below the compute caps that bound a plain solve, and far above the default `K = 11`.
    """
    max_ep_grid::Int
    function ResourceLimits(max_n_sim::Integer, max_n_subsets::Integer,
                            max_frontier::Integer, max_bins::Integer,
                            max_hop_count::Integer, max_search_grid::Integer,
                            max_ep_grid::Integer)
        @argcheck(max_n_sim > 0 &&
                  max_n_subsets > 0 &&
                  max_frontier > 0 &&
                  max_bins > 0 &&
                  max_hop_count > 0 &&
                  max_search_grid > 0 &&
                  max_ep_grid > 0,
                  ArgumentError("max_n_sim, max_n_subsets, max_frontier, max_bins, max_hop_count, max_search_grid and max_ep_grid must be positive."))
        return new(Int(max_n_sim), Int(max_n_subsets), Int(max_frontier), Int(max_bins),
                   Int(max_hop_count), Int(max_search_grid), Int(max_ep_grid))
    end
end
function ResourceLimits(; max_n_sim::Integer = 1_000_000, max_n_subsets::Integer = 100_000,
                        max_frontier::Integer = 100_000, max_bins::Integer = 10_000,
                        max_hop_count::Integer = 100_000,
                        max_search_grid::Integer = 100_000, max_ep_grid::Integer = 10_000)
    return ResourceLimits(max_n_sim, max_n_subsets, max_frontier, max_bins, max_hop_count,
                          max_search_grid, max_ep_grid)
end
"""
    RESOURCE_LIMITS = ScopedConfig(ResourceLimits())

Default global resource caps for the estimators that sample or sweep. The caps stop a configuration value that is large enough to use all of the memory, or to keep the process busy for days. Each estimator reads `RESOURCE_LIMITS[]`, and passes the cap to [`assert_resource_cap`](@ref). The `"max_n_sim"`, `"max_n_subsets"`, `"max_frontier"`, `"max_bins"`, `"max_hop_count"`, `"max_search_grid"` and `"max_ep_grid"` preferences can set the defaults for a project at load time, through [`apply_preferences!`](@ref).

# Related

  - [`ResourceLimits`](@ref)
  - [`set_resource_limits!`](@ref)
  - [`with_resource_limits`](@ref)
  - [`assert_resource_cap`](@ref)
"""
const RESOURCE_LIMITS = ScopedConfig(ResourceLimits())
"""
    set_resource_limits!(; max_n_sim::Integer, max_n_subsets::Integer,
                         max_frontier::Integer, max_bins::Integer,
                         max_hop_count::Integer, max_search_grid::Integer,
                         max_ep_grid::Integer)

Set the global default of the resource caps of [`RESOURCE_LIMITS`](@ref).

Raise the caps for a large run that a program configured, on a machine with enough memory for it. Lower them to accept less untrusted input. A keyword that the caller omits keeps its current default. The store is atomic, as [`ScopedConfig`](@ref) states. For a temporary, task-scoped override, use[`with_resource_limits`](@ref).

# Algorithm

 1. Give each keyword that the caller omits the value of its field in the current global default, read atomically. The function does not read a scoped override, so the call changes the global default alone.
 2. Build a [`ResourceLimits`](@ref) from the seven keywords. The validation below runs in that constructor.
 3. Store it as the global default of [`RESOURCE_LIMITS`](@ref) through [`set_default!`](@ref).

# Arguments

  - `max_n_sim::Integer`: The maximum `n_sim` that the uncertainty-set estimators accept.
  - `max_n_subsets::Integer`: The maximum `n_subsets` that the subset-resampling estimators accept.
  - `max_frontier::Integer`: The maximum `N` that one [`Frontier`](@ref) accepts, and the maximum number of sweep points over all swept bounds.
  - `max_bins::Integer`: The maximum `bins` that the mutual-information estimators accept.
  - `max_hop_count::Integer`: The maximum `n` that [`HopCount`](@ref) accepts, as a stated value or as the value that a rule resolves.
  - `max_search_grid::Integer`: The maximum number of candidates in the grid of a search cross-validation.
  - `max_ep_grid::Integer`: The maximum `K` that the grid formulations of the entropic and relativistic value-at-risk views accept.

# Validation

  - Every cap is positive. [`ResourceLimits`](@ref) checks it.

# Returns

  - `lims::ResourceLimits`: The new global default.

# Related

  - [`RESOURCE_LIMITS`](@ref)
  - [`with_resource_limits`](@ref)
  - [`set_equation_limits!`](@ref)
"""
function set_resource_limits!(;
                              max_n_sim::Integer = (@atomic RESOURCE_LIMITS.default).max_n_sim,
                              max_n_subsets::Integer = (@atomic RESOURCE_LIMITS.default).max_n_subsets,
                              max_frontier::Integer = (@atomic RESOURCE_LIMITS.default).max_frontier,
                              max_bins::Integer = (@atomic RESOURCE_LIMITS.default).max_bins,
                              max_hop_count::Integer = (@atomic RESOURCE_LIMITS.default).max_hop_count,
                              max_search_grid::Integer = (@atomic RESOURCE_LIMITS.default).max_search_grid,
                              max_ep_grid::Integer = (@atomic RESOURCE_LIMITS.default).max_ep_grid)
    return set_default!(RESOURCE_LIMITS,
                        ResourceLimits(; max_n_sim, max_n_subsets, max_frontier, max_bins,
                                       max_hop_count, max_search_grid, max_ep_grid))
end
"""
    with_resource_limits(f; max_n_sim::Integer = RESOURCE_LIMITS[].max_n_sim,
                         max_n_subsets::Integer = RESOURCE_LIMITS[].max_n_subsets,
                         max_frontier::Integer = RESOURCE_LIMITS[].max_frontier,
                         max_bins::Integer = RESOURCE_LIMITS[].max_bins,
                         max_hop_count::Integer = RESOURCE_LIMITS[].max_hop_count,
                         max_search_grid::Integer = RESOURCE_LIMITS[].max_search_grid,
                         max_ep_grid::Integer = RESOURCE_LIMITS[].max_ep_grid)

Run `f()` with an override of the resource caps of [`RESOURCE_LIMITS`](@ref) for the dynamic extent of the call. The previous caps come back when the call exits. The override is task-scoped and thread-safe, as [`ScopedConfig`](@ref) states, and the global default does not change. A keyword that the caller omits comes from the active value, so a nested override adds to the override around it.

Use it to raise a cap for one large run, while other concurrent work keeps its caps. The library reads a cap where it resolves the value. It reads the caps on `n_sim`, `N`, `bins` and `K` when the caller constructs the estimator. It reads the cap on `n_subsets` when the optimisation resolves its schedule, which can be a [`TimeDependent`](@ref). Put this call around the constructor call for the first four caps, and around the `optimise` call for `n_subsets`.

# Algorithm

 1. Give each keyword that the caller omits the value of its field in the active value. A nested override then starts from the override around it, and not from the global default.
 2. Build a [`ResourceLimits`](@ref) from the seven keywords. The validation below runs in that constructor.
 3. Run `f()` with [`RESOURCE_LIMITS`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: The zero-argument function to run under the override.
  - `max_n_sim::Integer`: The maximum `n_sim` that the uncertainty-set estimators accept.
  - `max_n_subsets::Integer`: The maximum `n_subsets` that the subset-resampling estimators accept.
  - `max_frontier::Integer`: The maximum `N` that one [`Frontier`](@ref) accepts, and the maximum number of sweep points over all swept bounds.
  - `max_bins::Integer`: The maximum `bins` that the mutual-information estimators accept.
  - `max_hop_count::Integer`: The maximum `n` that [`HopCount`](@ref) accepts, as a stated value or as the value that a rule resolves.
  - `max_search_grid::Integer`: The maximum number of candidates in the grid of a search cross-validation.
  - `max_ep_grid::Integer`: The maximum `K` that the grid formulations of the entropic and relativistic value-at-risk views accept.

# Validation

  - Every cap is positive. [`ResourceLimits`](@ref) checks it.

# Returns

  - The value that `f()` returns.

# Related

  - [`set_resource_limits!`](@ref)
  - [`RESOURCE_LIMITS`](@ref)
  - [`ResourceLimits`](@ref)
  - [`with_config`](@ref)
"""
function with_resource_limits(f; max_n_sim::Integer = RESOURCE_LIMITS[].max_n_sim,
                              max_n_subsets::Integer = RESOURCE_LIMITS[].max_n_subsets,
                              max_frontier::Integer = RESOURCE_LIMITS[].max_frontier,
                              max_bins::Integer = RESOURCE_LIMITS[].max_bins,
                              max_hop_count::Integer = RESOURCE_LIMITS[].max_hop_count,
                              max_search_grid::Integer = RESOURCE_LIMITS[].max_search_grid,
                              max_ep_grid::Integer = RESOURCE_LIMITS[].max_ep_grid)
    return with_config(f, RESOURCE_LIMITS,
                       ResourceLimits(; max_n_sim, max_n_subsets, max_frontier, max_bins,
                                      max_hop_count, max_search_grid, max_ep_grid))
end
"""
$(DocStringExtensions.TYPEDEF)

Global caps on how many times one log site of the library repeats one message, one cap per log level.

A cross-validation, a walk-forward, an online selection and a meta-optimiser call the same verb many times, and each call that meets the same condition logs the same message again. So every log site of the library passes `maxlog` and `_id` to its log macro. `maxlog` comes from [`log_limit`](@ref), which reads the cap of the level from this struct. `_id` comes from [`@log_id`](@ref), which keys the count on the site and on the stable parts of the configuration. A logger counts each `_id` separately, so a different configuration at the same site logs again.

A cap of `nothing` sets no cap. Each site then keeps the limit that it states, and most sites state none. An integer cap replaces the limit of every site of its level. A cap is at least `1`, so a cap never hides the first message of a site.

The value is immutable. The [`ScopedConfig`](@ref) [`LOG_LIMITS`](@ref) holds it. [`set_log_limits!`](@ref) sets the global default, and [`with_log_limits`](@ref) sets an override for one scope.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogLimits(max_warn::Union{Nothing, Integer}, max_info::Union{Nothing, Integer},
              max_debug::Union{Nothing, Integer}) -> LogLimits

    LogLimits(;
        max_warn::Union{Nothing, Integer} = nothing,
        max_info::Union{Nothing, Integer} = nothing,
        max_debug::Union{Nothing, Integer} = nothing
    ) -> LogLimits

Each keyword sets the field of the same name.

## Validation

  - Each cap is `nothing` or a positive integer that is not a `Bool`.

# Related

  - [`LOG_LIMITS`](@ref)
  - [`set_log_limits!`](@ref)
  - [`with_log_limits`](@ref)
  - [`log_limit`](@ref)
  - [`@log_id`](@ref)
"""
struct LogLimits
    """
    Maximum number of times one `@warn` site logs one `_id`. The default is `nothing`, and each site then keeps its own limit.
    """
    max_warn::Union{Nothing, Int}
    """
    Maximum number of times one `@info` site logs one `_id`. The default is `nothing`, and each site then keeps its own limit.
    """
    max_info::Union{Nothing, Int}
    """
    Maximum number of times one `@debug` site logs one `_id`. The default is `nothing`, and each site then keeps its own limit.
    """
    max_debug::Union{Nothing, Int}
    function LogLimits(max_warn::Union{Nothing, Integer}, max_info::Union{Nothing, Integer},
                       max_debug::Union{Nothing, Integer})
        @argcheck(all(x -> isnothing(x) || !(x isa Bool) && x > 0,
                      (max_warn, max_info, max_debug)),
                  ArgumentError("max_warn, max_info and max_debug must each be nothing or a positive integer."))
        return new(max_warn, max_info, max_debug)
    end
end
function LogLimits(; max_warn::Union{Nothing, Integer} = nothing,
                   max_info::Union{Nothing, Integer} = nothing,
                   max_debug::Union{Nothing, Integer} = nothing)
    return LogLimits(max_warn, max_info, max_debug)
end
"""
    LOG_LIMITS = ScopedConfig(LogLimits())

Default global caps on the repeats of one log site, one cap for each log level. A log site reads `LOG_LIMITS[]` through [`log_limit`](@ref). The `"max_warn"`, `"max_info"` and `"max_debug"` preferences can set the defaults for a project at load time, through [`apply_preferences!`](@ref).

# Related

  - [`LogLimits`](@ref)
  - [`set_log_limits!`](@ref)
  - [`with_log_limits`](@ref)
  - [`log_limit`](@ref)
"""
const LOG_LIMITS = ScopedConfig(LogLimits())
"""
    set_log_limits!(; max_warn::Union{Nothing, Integer}, max_info::Union{Nothing, Integer},
                    max_debug::Union{Nothing, Integer})

Set the global default of the caps of [`LOG_LIMITS`](@ref) on the repeats of one log site.

Set a cap to limit how many times each site of that level logs one `_id`. Set it to `nothing` to let each site keep its own limit. A keyword that the caller omits keeps its current default. The store is atomic, as [`ScopedConfig`](@ref) states. For a temporary, task-scoped override, use[`with_log_limits`](@ref).

# Algorithm

 1. Give each keyword that the caller omits the value of its field in the current global default, read atomically. The function does not read a scoped override, so the call changes the global default alone.
 2. Build a [`LogLimits`](@ref) from the three keywords. The validation below runs in that constructor.
 3. Store it as the global default of [`LOG_LIMITS`](@ref) through [`set_default!`](@ref).

# Arguments

  - `max_warn`: The maximum number of repeats of one `_id` at one `@warn` site, or `nothing`.
  - `max_info`: The maximum number of repeats of one `_id` at one `@info` site, or `nothing`.
  - `max_debug`: The maximum number of repeats of one `_id` at one `@debug` site, or `nothing`.

# Validation

  - Each cap is `nothing` or a positive integer. [`LogLimits`](@ref) checks it.

# Returns

  - `lims::LogLimits`: The new global default.

# Related

  - [`LOG_LIMITS`](@ref)
  - [`with_log_limits`](@ref)
  - [`set_resource_limits!`](@ref)
"""
function set_log_limits!(;
                         max_warn::Union{Nothing, Integer} = (@atomic LOG_LIMITS.default).max_warn,
                         max_info::Union{Nothing, Integer} = (@atomic LOG_LIMITS.default).max_info,
                         max_debug::Union{Nothing, Integer} = (@atomic LOG_LIMITS.default).max_debug)
    return set_default!(LOG_LIMITS, LogLimits(; max_warn, max_info, max_debug))
end
"""
    with_log_limits(f; max_warn::Union{Nothing, Integer} = LOG_LIMITS[].max_warn,
                    max_info::Union{Nothing, Integer} = LOG_LIMITS[].max_info,
                    max_debug::Union{Nothing, Integer} = LOG_LIMITS[].max_debug)

Run `f()` with an override of the caps of [`LOG_LIMITS`](@ref) on the repeats of one log site, for the dynamic extent of the call. The previous caps come back when the call exits. The override is task-scoped and thread-safe, as [`ScopedConfig`](@ref) states, and the global default does not change. A keyword that the caller omits comes from the active value, so a nested override adds to the override around it.

Use it to reduce the log of one long cross-validation, while other concurrent work keeps its log. A logger counts the repeats of an `_id` for as long as the logger exists. A site that reached its limit inside the block thus logs nothing after the block, under the same logger.

# Algorithm

 1. Give each keyword that the caller omits the value of its field in the active value. A nested override then starts from the override around it, and not from the global default.
 2. Build a [`LogLimits`](@ref) from the three keywords. The validation below runs in that constructor.
 3. Run `f()` with [`LOG_LIMITS`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: The zero-argument function to run under the override.
  - `max_warn`: The maximum number of repeats of one `_id` at one `@warn` site, or `nothing`.
  - `max_info`: The maximum number of repeats of one `_id` at one `@info` site, or `nothing`.
  - `max_debug`: The maximum number of repeats of one `_id` at one `@debug` site, or `nothing`.

# Validation

  - Each cap is `nothing` or a positive integer. [`LogLimits`](@ref) checks it.

# Returns

  - The value that `f()` returns.

# Related

  - [`set_log_limits!`](@ref)
  - [`LOG_LIMITS`](@ref)
  - [`LogLimits`](@ref)
  - [`with_config`](@ref)
"""
function with_log_limits(f; max_warn::Union{Nothing, Integer} = LOG_LIMITS[].max_warn,
                         max_info::Union{Nothing, Integer} = LOG_LIMITS[].max_info,
                         max_debug::Union{Nothing, Integer} = LOG_LIMITS[].max_debug)
    return with_config(f, LOG_LIMITS, LogLimits(; max_warn, max_info, max_debug))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Give the `maxlog` of one log site. It is the cap of the level in [`LOG_LIMITS`](@ref) when that cap is set, and the limit that the site states in the other case.

Every log site of the library passes `maxlog = log_limit(level)`, or `maxlog = log_limit(level, n)` when the site limits itself to `n` repeats. A `maxlog` of `nothing` sets no limit, because a logger limits a record only when its `maxlog` is an integer.

# Algorithm

 1. Read the active value of [`LOG_LIMITS`](@ref), and take the cap of `level` from it.
 2. Return the cap when it is an integer, and `site` when it is `nothing`.

# Arguments

  - `level`: The level of the log macro, `:warn`, `:info` or `:debug`.
  - `site`: The limit that the site states, or `nothing` for no limit of its own.

# Validation

  - `level` is `:warn`, `:info` or `:debug`. Each other symbol raises an `ArgumentError`.

# Returns

  - `maxlog::Union{Nothing, Integer}`: The value to pass as the `maxlog` of the log macro.

# Related

  - [`LOG_LIMITS`](@ref)
  - [`LogLimits`](@ref)
  - [`@log_id`](@ref)
"""
function log_limit(level::Symbol, site::Union{Nothing, Integer} = nothing)
    lims = LOG_LIMITS[]
    cap = if level === :warn
        lims.max_warn
    elseif level === :info
        lims.max_info
    elseif level === :debug
        lims.max_debug
    else
        throw(ArgumentError("level must be :warn, :info or :debug; got $(repr(level))."))
    end
    return isnothing(cap) ? site : cap
end
"""
    @log_id(parts...)

Give the `_id` of one log site. It is a hash of the site and of the stable parts of its message.

A logger counts the `maxlog` of each `_id` separately. The default `_id` of a log macro is one id for each source line. A limit on a line that logs different messages hides every later message, and a limit inside a shared helper such as [`strict_diagnostic`](@ref) hides the messages of every caller. An `_id` made from the full message never limits a message that holds a value that changes on each call. `@log_id` therefore keys the count on the module, the file and the line of its call, and on `parts`.

`parts` are the stable parts of the configuration that the message reports. They are the types of the estimators, and the keys and names that the caller wrote. The values that the data gives, such as an asset list, a count, an index, a float or a timestamp, stay in the message text and out of `parts`. The same condition at the same site under the same configuration counts as one message, and a different configuration logs again.

# Algorithm

 1. At expansion, build a `Symbol` from the module, the file and the line of the call.
 2. At run time, return the hash of the tuple of that symbol and the values of `parts`.

# Arguments

  - `parts`: The stable parts of the message, each evaluated in the scope of the call.

# Returns

  - `id::UInt`: The value to pass as the `_id` of the log macro, or as the `id` of [`strict_diagnostic`](@ref).

# Related

  - [`log_limit`](@ref)
  - [`LOG_LIMITS`](@ref)
  - [`strict_diagnostic`](@ref)
"""
macro log_id(parts...)
    site = Symbol(__module__, ".", something(__source__.file, :none), ":", __source__.line)
    return :(hash(($(QuoteNode(site)), $(map(esc, parts)...))))
end
