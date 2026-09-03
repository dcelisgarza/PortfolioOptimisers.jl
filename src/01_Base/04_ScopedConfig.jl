"""
$(DocStringExtensions.TYPEDEF)

Thread-safe holder for a package-level configuration value, combining a persistent global default with a task-scoped override.

Reads go through `cfg[]`, which returns the innermost active scoped override when inside a `with_*` block, otherwise the global default. The default is an `@atomic` field swapped as a whole — a `set_*!` call is a single atomic store, so concurrent readers (e.g. the `FLoops.@floop` loops inside meta-optimisers) can never observe a torn or partially-updated configuration. The scoped override is a `Base.ScopedValues.ScopedValue`: it is inherited by tasks spawned inside the scope, restored automatically when the scope exits, and invisible to unrelated concurrent tasks.

Configs held this way store *immutable* structs (or bits values); changing any knob builds a new value and swaps it in, never mutates in place.

Used by [`COMPACT_SHOW`](@ref), [`SHOW_NOTHING_FIELDS`](@ref), [`STRING_DISTANCE`](@ref), and [`EQUATION_LIMITS`](@ref); their global defaults are set via the `set_*!` setters, scoped overrides via the `with_*` helpers, and load-time per-project defaults via Preferences.jl (see [`apply_preferences!`](@ref)).

# Related

  - [`set_compact_show!`](@ref) / [`with_compact_show`](@ref)
  - [`set_show_nothing_fields!`](@ref) / [`with_show_nothing_fields`](@ref)
  - [`set_string_distance!`](@ref) / [`with_string_distance`](@ref)
  - [`set_equation_limits!`](@ref) / [`with_equation_limits`](@ref)
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

Read the active value of a [`ScopedConfig`](@ref): the innermost task-scoped override when inside a `with_*` block, otherwise the global default (read atomically).

# Algorithm

 1. Read `cfg.scoped[]`, the innermost task-scoped override, giving `nothing` outside every `with_*` block.
 2. When step 1 gives `nothing`, read `cfg.default` atomically instead.

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

Atomically replace the global default of a [`ScopedConfig`](@ref) with `x` and return it. Does not affect any active scoped override.

# Algorithm

 1. Convert `x` to `T`, the element type of `cfg`, so that a failed conversion raises before anything is stored.
 2. Store the converted `x` into `cfg.default` in one atomic write, so that a concurrent reader sees either the old value or the new one and never a partial write.

# Arguments

  - `cfg`: Configuration holder whose global default is replaced.
  - `x`: New value, converted to the element type `T` of `cfg`.

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

Run `f()` with the [`ScopedConfig`](@ref) `cfg` overridden to `x` for the dynamic extent of the call, restoring the previous value on exit. Thread-safe: the override is task-scoped (inherited by tasks spawned inside `f`, invisible to concurrent tasks outside it).

# Algorithm

 1. Convert `x` to `T`, the element type of `cfg`.
 2. Bind `cfg.scoped` to the converted `x` and run `f()` inside that binding, so that the binding is restored when `f` returns and when `f` raises.

# Arguments

  - `f`: Zero-argument function to run under the override.
  - `cfg`: Configuration holder to override.
  - `x`: Override value, converted to the element type `T` of `cfg`.

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
Global control for collapsing large nested structs in [`@define_pretty_show`](@ref) output.

Holds one of:

  - `false`: collapsing disabled; nested structs always expand fully.
  - `true`: collapsing enabled with an automatic, terminal-size-derived line budget.
  - `n::Int`: collapsing enabled with a fixed line budget of `n`.

Held in a [`ScopedConfig`](@ref): set the global default via [`set_compact_show!`](@ref), override per scope via [`with_compact_show`](@ref), and read (together with the per-call `:po_compact` IO property) by [`compact_show_budget`](@ref). The default may be seeded per project at load time via the `"compact_show"` preference (see [`apply_preferences!`](@ref)).
"""
const COMPACT_SHOW = ScopedConfig{Union{Bool, Int}}(true)
"""
    set_compact_show!(x::Bool)
    set_compact_show!(n::Integer)

Configure whether [`@define_pretty_show`](@ref) collapses large nested structs.

  - `set_compact_show!(false)`: disable collapsing (always expand fully).
  - `set_compact_show!(true)`: enable collapsing with an automatic, terminal-size-derived budget.
  - `set_compact_show!(n)`: enable collapsing with a fixed line budget `n`.

Collapsing only ever applies to height-limited output (`get(io, :limit, false)`), i.e. the interactive REPL. Non-limited output (`string`, `repr`, file writes) always expands fully. The documentation build disables this so rendered docs keep full detail. Individual calls can override the global setting with the `:po_compact` IO property (`false`, `true`, or an `Int`).

Sets the global default (atomically; see [`ScopedConfig`](@ref)). For a temporary, task-scoped override use [`with_compact_show`](@ref).

# Algorithm

 1. Widen an `Integer` argument to `Int`, so that the two methods store one of the two members of the stored union and never a third integer type. A `Bool` is stored unchanged.
 2. Store the value as the global default of [`COMPACT_SHOW`](@ref) through [`set_default!`](@ref).

# Arguments

  - `x::Bool`: `false` disables collapsing, `true` enables it with the automatic budget.
  - `n::Integer`: Fixed line budget, stored as an `Int`.

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

Run `f()` with the [`COMPACT_SHOW`](@ref) collapsing setting overridden to `x`/`n` for the dynamic extent of the call, restoring the previous setting on exit. Task-scoped and thread-safe (see [`ScopedConfig`](@ref)); the global default is untouched.

# Algorithm

 1. Widen an `Integer` argument to `Int`, as [`set_compact_show!`](@ref) does. A `Bool` is passed unchanged.
 2. Run `f()` with [`COMPACT_SHOW`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: Zero-argument function to run under the override.
  - `x::Bool`: `false` disables collapsing, `true` enables it with the automatic budget.
  - `n::Integer`: Fixed line budget, stored as an `Int`.

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

Resolve the line budget that triggers collapsing a nested struct rendered by [`@define_pretty_show`](@ref).

The per-call `:po_compact` IO property takes precedence over the global [`COMPACT_SHOW`](@ref) setting; both accept `false` (disabled), `true` (automatic budget), or an `Int` (fixed budget). The automatic budget is `max(8, displaysize(io)[1] - 4)`, so only subtrees that nearly fill or exceed the terminal collapse.

# Algorithm

 1. Read the `:po_compact` property of `io`, giving `v`.
 2. When `io` sets no such property, collapsing applies only to height-limited output: return `nothing` when `:limit` is `false`, and otherwise read the global [`COMPACT_SHOW`](@ref) setting into `v`.
 3. Return `nothing` when `v` is `false`, because that value disables collapsing.
 4. Return `Int(v)` when `v` is an integer that is not a `Bool`, because that value is the fixed budget.
 5. Otherwise `v` is `true`, so return the automatic budget `max(8, displaysize(io)[1] - 4)`.

# Arguments

  - `io`: Stream whose `:po_compact`, `:limit` and `:displaysize` properties the resolution reads.

# Returns

  - `nothing` when collapsing is disabled.
  - `budget::Int` (the maximum number of rendered lines a nested struct may occupy before collapsing) otherwise.

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

Held in the [`SHOW_NOTHING_FIELDS`](@ref) [`ScopedConfig`](@ref), and read by [`pretty_show_fields`](@ref). Set the global default via [`set_show_nothing_fields!`](@ref), override per scope via [`with_show_nothing_fields`](@ref). The value is treated as immutable: a change builds a new value with a copied `by_type` and swaps it in, so a reader never observes a half-written table.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShowNothingFields(default::Bool, by_type::Dict{Symbol, Bool}) -> ShowNothingFields
    ShowNothingFields(cfg::ShowNothingFields, name::Symbol,
                      x::Union{Nothing, Bool}) -> ShowNothingFields

The second form copies `cfg` with one per-name entry changed: `x` sets the entry for `name`, and `nothing` removes it.

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
    Whether a field that holds `nothing` prints for a type that no per-name entry names. The shipped default is `false`, so a `nothing` field is hidden.
    """
    default::Bool
    """
    Per-name overrides, keyed on the bare name of a type. A `true` entry prints every declared field of that type, and a `false` entry hides its `nothing` fields, whatever `default` holds and whatever the type's own [`show_fields`](@ref) method returns.
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

The shipped default hides such a field, because a reader at the REPL wants the fields that carry a value. The documentation build turns them on with `set_show_nothing_fields!(true)`, beside its `set_compact_show!(false)` call, so a rendered docstring shows the complete type. Read as `SHOW_NOTHING_FIELDS[]` by [`pretty_show_fields`](@ref); the default may be seeded per project at load time via the `"show_nothing_fields"` and `"show_nothing_fields_by_type"` preferences (see [`apply_preferences!`](@ref)).

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

  - `set_show_nothing_fields!(false)`: hide a `nothing` field, for every type that no per-name entry names. This is the shipped default.
  - `set_show_nothing_fields!(true)`: render a `nothing` field, for every type that no per-name entry names.
  - `set_show_nothing_fields!(:SimpleVariance, false)`: hide the `nothing` fields of every type named `SimpleVariance`, whatever the global switch says.
  - `set_show_nothing_fields!(:SimpleVariance, true)`: render every declared field of every type named `SimpleVariance`, including one that its own [`show_fields`](@ref) method hides.
  - `set_show_nothing_fields!(:SimpleVariance, nothing)`: remove the per-name entry, so the type follows the global switch and its own [`show_fields`](@ref) method again.

The per-name form takes a name and not a type, because the load-time preference channel is TOML, which carries no types. The name is the bare name of the type, so two types of one name in two modules share one entry. A name that matches no type is accepted and does nothing, because a type outside the package can render through [`@define_pretty_show`](@ref) too.

Sets the global default (atomically; see [`ScopedConfig`](@ref)). For a temporary, task-scoped override use [`with_show_nothing_fields`](@ref).

# Algorithm

 1. Read the current global default of [`SHOW_NOTHING_FIELDS`](@ref) atomically. A scoped override is not read, so the call configures the global default alone.
 2. Build the new [`ShowNothingFields`](@ref): the one-argument form keeps the per-name table and replaces the global switch, and the two-argument form keeps the global switch and copies the table with one entry set or removed.
 3. Store it as the global default through [`set_default!`](@ref).

# Arguments

  - `x::Bool`: `false` hides a `nothing` field, `true` renders it.
  - `name::Symbol`: The bare name of a type.
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

Run `f()` with the [`SHOW_NOTHING_FIELDS`](@ref) setting overridden for the dynamic extent of the call, restoring the previous setting on exit. Task-scoped and thread-safe (see [`ScopedConfig`](@ref)); the global default is untouched. The part of the setting the call does not name inherits from the currently active value, so nested overrides compose.

# Algorithm

 1. Read the currently active value of [`SHOW_NOTHING_FIELDS`](@ref), so that a nested override inherits from the enclosing one instead of from the global default.
 2. Build the override as [`set_show_nothing_fields!`](@ref) does: the one-argument form replaces the global switch and keeps the per-name table, and the two-argument form keeps the global switch and copies the table with one entry set or removed.
 3. Run `f()` with [`SHOW_NOTHING_FIELDS`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: Zero-argument function to run under the override.
  - `x::Bool`: `false` hides a `nothing` field, `true` renders it.
  - `name::Symbol`: The bare name of a type.
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

Global configuration for the fuzzy "did you mean?" suggestions appended to "variable not in asset universe" messages by [`did_you_mean`](@ref).

Immutable; held in the [`STRING_DISTANCE`](@ref) [`ScopedConfig`](@ref). Set the global default via [`set_string_distance!`](@ref), override per scope via [`with_string_distance`](@ref). Read by [`did_you_mean`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StringDistanceConfig(dist::StringDistances.StringDistance,
                         min_score::Real) -> StringDistanceConfig

## Validation

  - `min_score > 0`. `StringDistances.findnearest` never suggests a candidate scoring exactly `0`, but any threshold at or below `0` admits every candidate with *some* nonzero similarity, so `0` and a negative value behave identically. Both defeat the info-leak-safe boundary by naming a real asset for a near-miss probe.

# Related

  - [`STRING_DISTANCE`](@ref)
  - [`set_string_distance!`](@ref)
  - [`with_string_distance`](@ref)
  - [`did_you_mean`](@ref)
"""
struct StringDistanceConfig
    """
    Distance used to score a candidate name against the offending one. The default is `StringDistances.Levenshtein()`.
    """
    dist::StringDistances.StringDistance
    """
    Minimum normalised similarity a candidate must reach before it is suggested. The default is `0.7`. A value toward `1` keeps only a near-exact match, and a value above `1` disables suggestions entirely, which is what a meta-optimiser inner loop wants: an asset name legitimately absent from a cluster or a subset is not a typo and must draw no suggestion.
    """
    min_score::Float64
    function StringDistanceConfig(dist::StringDistances.StringDistance, min_score::Real)
        @argcheck(min_score > 0,
                  ArgumentError("min_score must be positive; got $(min_score). A value above 1 legitimately disables suggestions, but a zero or negative threshold admits every candidate with any nonzero similarity, making `did_you_mean` echo a real asset name for near-miss probes and defeating the info-leak-safe boundary (ADR 0026)."))
        return new(dist, Float64(min_score))
    end
end
"""
    STRING_DISTANCE = ScopedConfig(StringDistanceConfig(StringDistances.Levenshtein(), 0.7))

Default string distance configuration for fuzzy "did you mean?" suggestions appended to "variable not in asset universe" messages by [`did_you_mean`](@ref). Read as `STRING_DISTANCE[]`; the defaults may be seeded per project at load time via the `"suggestion_distance"` / `"suggestion_min_score"` preferences (see [`apply_preferences!`](@ref)).

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

Configure the global default fuzzy-suggestion settings read by [`did_you_mean`](@ref). The store is atomic (see [`ScopedConfig`](@ref)); unspecified keywords keep their current default. For a temporary, task-scoped override use [`with_string_distance`](@ref).

# Algorithm

 1. Default each keyword the caller omits to the field of the current global default, read atomically. A scoped override is not read, so the call configures the global default alone.
 2. Build a [`StringDistanceConfig`](@ref) from the two keywords, which is where the validation below runs.
 3. Store it as the global default of [`STRING_DISTANCE`](@ref) through [`set_default!`](@ref).

# Arguments

  - `dist::StringDistances.StringDistance`: Distance used to rank candidate names, for example `StringDistances.Levenshtein()`, `StringDistances.DamerauLevenshtein()` or `StringDistances.JaroWinkler()`.
  - `min_score::Real`: Minimum normalised similarity in `(0, 1]` that emits a suggestion. A value above `1` disables suggestions.

# Validation

  - `min_score > 0`, enforced by [`StringDistanceConfig`](@ref). A non-positive threshold admits every candidate with any nonzero similarity.

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
    return set_default!(STRING_DISTANCE, StringDistanceConfig(dist, Float64(min_score)))
end
"""
    with_string_distance(f; dist::StringDistances.StringDistance = STRING_DISTANCE[].dist,
                         min_score::Real = STRING_DISTANCE[].min_score)

Run `f()` with the fuzzy-suggestion settings read by [`did_you_mean`](@ref) overridden for the dynamic extent of the call, restoring the previous settings on exit. Task-scoped and thread-safe (see [`ScopedConfig`](@ref)); the global default is untouched. Unspecified keywords inherit from the currently active value, so nested overrides compose.

Useful around a meta-optimiser run to silence suggestions (`min_score` above `1`) in its inner loops without affecting other concurrent work.

# Algorithm

 1. Default each keyword the caller omits to the field of the currently active value, so that a nested override inherits from the enclosing one instead of from the global default.
 2. Build a [`StringDistanceConfig`](@ref) from the two keywords, which is where the validation below runs.
 3. Run `f()` with [`STRING_DISTANCE`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: Zero-argument function to run under the override.
  - `dist::StringDistances.StringDistance`: Distance used to rank candidate names.
  - `min_score::Real`: Minimum normalised similarity in `(0, 1]` that emits a suggestion. A value above `1` disables suggestions.

# Validation

  - `min_score > 0`, enforced by [`StringDistanceConfig`](@ref).

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
    return with_config(f, STRING_DISTANCE, StringDistanceConfig(dist, Float64(min_score)))
end
"""
$(DocStringExtensions.TYPEDEF)

Global resource caps for equation parsing, guarding the string→AST trust boundary against a stack-exhaustion denial of service.

Constraint, Black-Litterman view and entropy-pooling view strings are untrusted input (config files, spreadsheets, UI). They funnel through [`parse_equation`](@ref), which calls `Meta.parse` and then walks the resulting expression tree recursively ([`eval_numeric_functions`](@ref), `collect_terms!`, `has_invalid_plus`). Without a bound, a deeply nested string (e.g. tens of thousands of parentheses) produces an AST deep enough to exhaust the stack and take down the host process. These caps fail closed with a typed `Meta.ParseError` well before that point.

The values are conservative static defaults (portable across build and deployment machines, unlike a value auto-detected during precompilation). Immutable; held in the [`EQUATION_LIMITS`](@ref) [`ScopedConfig`](@ref). Set the global default via [`set_equation_limits!`](@ref), override per scope via [`with_equation_limits`](@ref). See `docs/adr/0027-cap-equation-parser-recursion.md`.

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
    Maximum number of characters in an equation string handed to `Meta.parse`. The default is `4096`. A legitimate linear constraint is short, so the bound sits far above any real constraint and far below the nesting depth that threatens the stack. A nesting depth `d` needs at least `d` characters, so the length cap also bounds the AST depth of the *string* form.
    """
    max_length::Int
    """
    Maximum expression-tree depth accepted by the `Expr` form of [`parse_equation`](@ref). The default is `256`. That form receives a pre-built AST, which no length cap covers.
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

Default global resource caps for equation parsing, guarding the string→AST trust boundary against a stack-exhaustion denial of service. Read as `EQUATION_LIMITS[]`; the defaults may be seeded per project at load time via the `"equation_max_length"` / `"equation_max_depth"` preferences (see [`apply_preferences!`](@ref)).

# Related

  - [`EquationLimits`](@ref)
  - [`set_equation_limits!`](@ref)
  - [`with_equation_limits`](@ref)
  - [`parse_equation`](@ref)
"""
const EQUATION_LIMITS = ScopedConfig(EquationLimits(4096, 256))
"""
    set_equation_limits!(; max_length::Integer, max_depth::Integer)

Configure the global default equation-parser resource caps read at the string→AST trust boundary (see [`EQUATION_LIMITS`](@ref)).

Raise them for a genuinely large machine-generated constraint set, or lower them to tighten the boundary. Unspecified keywords keep their current default. The store is atomic (see [`ScopedConfig`](@ref)); for a temporary, task-scoped override use [`with_equation_limits`](@ref).

# Algorithm

 1. Default each keyword the caller omits to the field of the current global default, read atomically. A scoped override is not read, so the call configures the global default alone.
 2. Build an [`EquationLimits`](@ref) from the two keywords, which is where the validation below runs.
 3. Store it as the global default of [`EQUATION_LIMITS`](@ref) through [`set_default!`](@ref).

# Arguments

  - `max_length::Integer`: Maximum equation-string length passed to `Meta.parse`.
  - `max_depth::Integer`: Maximum expression-tree depth accepted by the `Expr` form of [`parse_equation`](@ref).

# Validation

  - `max_length > 0 && max_depth > 0`, enforced by [`EquationLimits`](@ref).

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

Run `f()` with the equation-parser resource caps (see [`EQUATION_LIMITS`](@ref)) overridden for the dynamic extent of the call, restoring the previous caps on exit. Task-scoped and thread-safe (see [`ScopedConfig`](@ref)); the global default is untouched. Unspecified keywords inherit from the currently active value, so nested overrides compose.

Useful to tighten the boundary around one batch of untrusted constraint strings, or to raise it for a single machine-generated constraint set, without affecting other concurrent work.

# Algorithm

 1. Default each keyword the caller omits to the field of the currently active value, so that a nested override inherits from the enclosing one instead of from the global default.
 2. Build an [`EquationLimits`](@ref) from the two keywords, which is where the validation below runs.
 3. Run `f()` with [`EQUATION_LIMITS`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: Zero-argument function to run under the override.
  - `max_length::Integer`: Maximum equation-string length passed to `Meta.parse`.
  - `max_depth::Integer`: Maximum expression-tree depth accepted by the `Expr` form of [`parse_equation`](@ref).

# Validation

  - `max_length > 0 && max_depth > 0`, enforced by [`EquationLimits`](@ref).

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

Global resource caps for the sampling- and sweep-based estimators, guarding the config→allocation trust boundary against memory and compute exhaustion.

Draw counts, subset counts, frontier-sweep lengths and histogram bin counts are untrusted configuration integers (config files, tuning grids, UI): each directly multiplies an allocation, and in the subset and frontier cases a whole optimisation. Their own constructors only bound them from *below* (`n_sim > 0`, `n_subsets >= 2`, `N > 0`, `bins > 0`), so an absurd value — a stray extra digit, a mis-scaled sweep — is accepted and the process is killed by the OOM killer rather than told what went wrong. These caps fail closed with a typed `DomainError` at the point the value is resolved.

There is **one cap per sink**, each named to mirror the field it guards. Reuse across sinks is deliberately avoided: a *linear* cap cannot bound a *quadratic* sink, which is why the `bins × bins` histogram gets its own [`max_bins`](@ref ResourceLimits) rather than sharing the linear draw cap.

The values are conservative static defaults, deliberately far above legitimate use: they exist to convert an OOM kill into a typed error, not to second-guess a sizing choice. Immutable; held in the [`RESOURCE_LIMITS`](@ref) [`ScopedConfig`](@ref). Set the global default via [`set_resource_limits!`](@ref), override per scope via [`with_resource_limits`](@ref). Prefer the keyword constructor `ResourceLimits(; …)` — the seven caps are same-typed and four share the value `100_000`, so positional construction is error-prone.

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

Keywords correspond to the struct's fields.

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
    Maximum Monte-Carlo or bootstrap draws (`n_sim`) accepted by [`NormalUncertaintySet`](@ref) and [`ARCHUncertaintySet`](@ref). The default is `1_000_000`. Each draw stores an `N × N` covariance, so the backing array is `N² · n_sim` elements: at 20 assets the default cap already permits a 3.2 GB request, while the shipped `n_sim` is `3_000`. *Memory*-bound.
    """
    max_n_sim::Int
    """
    Maximum resampled asset subsets (`n_subsets`) accepted by [`SubsetResampling`](@ref) and [`MultipleRandomised`](@ref). The default is `100_000`. This cap bounds *compute* far more than memory, because every subset runs a full inner optimisation, so it sits far above any realistic sweep (the shipped default is `2`) yet well below a value that would wedge a session for days.
    """
    max_n_subsets::Int
    """
    Maximum efficient-frontier sweep points accepted by the [`Frontier`](@ref) algorithm of [`MeanRisk`](@ref) and [`NearOptimalCentering`](@ref). The default is `100_000`. Like `max_n_subsets` it is *compute*-bound, because every point runs a full inner `optimise_JuMP_model!` solve, so it mirrors that ceiling; the shipped [`Frontier`](@ref) default is `N = 20`. It is enforced **twice**: [`Frontier`](@ref)'s constructor caps the `N` of one bound, and [`assert_frontier_sweep_cap`](@ref) caps the **product** across every swept return term and every swept risk measure at Model Assembly, because the sweep is an `Iterators.product` and `k` bounds of `N` points cost `N^k` solves.
    """
    max_frontier::Int
    """
    Maximum histogram bins accepted by [`MutualInfoCovariance`](@ref) and [`VariationInfoDistance`](@ref). The default is `10_000`. The joint histogram is a `bins × bins` weights matrix built per asset pair, so this bounds a *quadratic* memory allocation: `10_000²` cells is about 800 MB per histogram, below OOM yet far above the roughly 50-bin range that legitimate binning produces.
    """
    max_bins::Int
    """
    Maximum hop count (`n`) accepted by [`HopCount`](@ref). The default is `100_000`. Three readers sum `A^i` over `i in 0:n`, so the sink is *linear* in `n` at `N³` flops a power, and a large `n` wedges the session on compute rather than on memory. Like `max_n_subsets` it is compute-bound and mirrors that ceiling; the shipped default is `n = 1`. The cap is read in [`HopCount`](@ref)'s constructor, which is also where [`resolve_separation`](@ref) sends a rule's answer, so one check covers the stated value and the computed one alike.
    """
    max_hop_count::Int
    """
    Maximum search-grid candidates accepted by [`GridSearchCrossValidation`](@ref) and [`RandomisedSearchCrossValidation`](@ref). The default is `100_000`. Every candidate runs a full cross-validated fit, so this is *compute*-bound like `max_n_subsets`. The grid is an `Iterators.product` materialised by `collect`, so `k` parameters of `N` values cost `N^k` candidates: the cap is asserted on the **product** by [`assert_search_grid_cap`](@ref) where the grid is formed, because a per-parameter check can never see it.
    """
    max_search_grid::Int
    """
    Maximum grid points (`K`) accepted by [`GridEntropicValueatRiskView`](@ref) and [`GridRelativisticValueatRiskView`](@ref). The default is `10_000`. Every grid point is one binary variable of the mixed-integer program an upper-bound or equality view builds, and one dense row over the `T` posterior probabilities, so the sink is a *mixed-integer* one: linear in `K` in memory at `K · T` coefficients, and a branch-and-bound tree over `K` binaries in compute. It therefore sits far below the compute ceilings that bound a plain solve, and far above the shipped `K = 11`.
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

Default global resource caps for the sampling- and sweep-based estimators, guarding the config→allocation trust boundary against memory and compute exhaustion. Read as `RESOURCE_LIMITS[]`; the defaults may be seeded per project at load time via the `"max_n_sim"` / `"max_n_subsets"` / `"max_frontier"` / `"max_bins"` / `"max_hop_count"` / `"max_search_grid"` / `"max_ep_grid"` preferences (see [`apply_preferences!`](@ref)).

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

Configure the global default resource caps read at the config→allocation trust boundary (see [`RESOURCE_LIMITS`](@ref)).

Raise them for a genuinely large machine-authored run on a machine sized for it, or lower them to tighten the boundary. Unspecified keywords keep their current default. The store is atomic (see [`ScopedConfig`](@ref)); for a temporary, task-scoped override use [`with_resource_limits`](@ref).

# Algorithm

 1. Default each keyword the caller omits to the field of the current global default, read atomically. A scoped override is not read, so the call configures the global default alone.
 2. Build a [`ResourceLimits`](@ref) from the seven keywords, which is where the validation below runs.
 3. Store it as the global default of [`RESOURCE_LIMITS`](@ref) through [`set_default!`](@ref).

# Arguments

  - `max_n_sim::Integer`: Maximum `n_sim` accepted by the uncertainty-set estimators.
  - `max_n_subsets::Integer`: Maximum `n_subsets` accepted by the subset-resampling estimators.
  - `max_frontier::Integer`: Maximum `N` accepted by one [`Frontier`](@ref), and maximum total sweep points across every swept bound.
  - `max_bins::Integer`: Maximum `bins` accepted by the mutual-information estimators.
  - `max_hop_count::Integer`: Maximum `n` accepted by [`HopCount`](@ref), stated or resolved from a rule.
  - `max_search_grid::Integer`: Maximum total candidates in a search cross-validation grid.
  - `max_ep_grid::Integer`: Maximum `K` accepted by the grid formulations of the entropic and relativistic value-at-risk views.

# Validation

  - Every cap is positive, enforced by [`ResourceLimits`](@ref).

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

Run `f()` with the resource caps (see [`RESOURCE_LIMITS`](@ref)) overridden for the dynamic extent of the call, restoring the previous caps on exit. Task-scoped and thread-safe (see [`ScopedConfig`](@ref)); the global default is untouched. Unspecified keywords inherit from the currently active value, so nested overrides compose.

Useful to raise the ceiling for one deliberately large run without loosening the boundary for other concurrent work. Note the cap is read where the value is *resolved*: `n_sim`, `N`, `bins` and `K` at estimator construction, `n_subsets` when the optimisation resolves its (possibly [`TimeDependent`](@ref)) schedule — so wrap the constructor call in the former cases and the `optimise` call in the latter.

# Algorithm

 1. Default each keyword the caller omits to the field of the currently active value, so that a nested override inherits from the enclosing one instead of from the global default.
 2. Build a [`ResourceLimits`](@ref) from the seven keywords, which is where the validation below runs.
 3. Run `f()` with [`RESOURCE_LIMITS`](@ref) bound to that value through [`with_config`](@ref).

# Arguments

  - `f`: Zero-argument function to run under the override.
  - `max_n_sim::Integer`: Maximum `n_sim` accepted by the uncertainty-set estimators.
  - `max_n_subsets::Integer`: Maximum `n_subsets` accepted by the subset-resampling estimators.
  - `max_frontier::Integer`: Maximum `N` accepted by one [`Frontier`](@ref), and maximum total sweep points across every swept bound.
  - `max_bins::Integer`: Maximum `bins` accepted by the mutual-information estimators.
  - `max_hop_count::Integer`: Maximum `n` accepted by [`HopCount`](@ref), stated or resolved from a rule.
  - `max_search_grid::Integer`: Maximum total candidates in a search cross-validation grid.
  - `max_ep_grid::Integer`: Maximum `K` accepted by the grid formulations of the entropic and relativistic value-at-risk views.

# Validation

  - Every cap is positive, enforced by [`ResourceLimits`](@ref).

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
