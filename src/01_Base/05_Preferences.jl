"""
    PREFERENCE_DISTANCES

Enumerated allowlist mapping the names accepted by the `"suggestion_distance"` preference to their `StringDistances.StringDistance` objects. Membership and dispatch are one `Dict` — the same single-source-of-truth discipline as the equation parser's function allowlist (`docs/adr/0025-enumerated-parser-allowlist.md`): an unknown name fails closed at load with a typed error carrying a [`did_you_mean`](@ref) suggestion.

Supported names: `"levenshtein"`, `"damerau_levenshtein"`, `"jaro"`, `"jaro_winkler"`, `"ratcliff_obershelp"`.

# Related

  - [`apply_preferences!`](@ref)
  - [`set_string_distance!`](@ref)
"""
const PREFERENCE_DISTANCES = Dict{String, StringDistances.StringDistance}("levenshtein" =>
                                                                              StringDistances.Levenshtein(),
                                                                          "damerau_levenshtein" =>
                                                                              StringDistances.DamerauLevenshtein(),
                                                                          "jaro" =>
                                                                              StringDistances.Jaro(),
                                                                          "jaro_winkler" =>
                                                                              StringDistances.JaroWinkler(),
                                                                          "ratcliff_obershelp" =>
                                                                              StringDistances.RatcliffObershelp())
"""
    PREFERENCE_KEYS

The Preferences.jl keys read at package load to seed the global config defaults (see [`apply_preferences!`](@ref)):

  - `"equation_max_length"` / `"equation_max_depth"`: positive integers for [`EQUATION_LIMITS`](@ref).
  - `"max_n_sim"` / `"max_n_subsets"` / `"max_frontier"` / `"max_bins"` / `"max_hop_count"` / `"max_search_grid"`: positive integers for [`RESOURCE_LIMITS`](@ref).
  - `"suggestion_min_score"`: real number for the [`STRING_DISTANCE`](@ref) threshold.
  - `"suggestion_distance"`: a [`PREFERENCE_DISTANCES`](@ref) name for the [`STRING_DISTANCE`](@ref) metric.
  - `"compact_show"`: boolean or integer for [`COMPACT_SHOW`](@ref).

Preferences.jl offers no way to enumerate the keys a project has set, so a misspelled *key* cannot be detected and is silently ignored (the shipped default applies) — misspelled or invalid *values* under these keys fail closed at load.

A valid value is applied, but a value that *widens* a guard is announced with a warning (see [`relaxed_preferences_msg`](@ref)): a preference file is data, it travels with a cloned project, and it applies before any user code runs.

# Related

  - [`apply_preferences!`](@ref)
  - [`relaxed_preferences_msg`](@ref)
"""
const PREFERENCE_KEYS = ("equation_max_length", "equation_max_depth", "max_n_sim",
                         "max_n_subsets", "max_frontier", "max_bins", "max_hop_count",
                         "max_search_grid", "suggestion_min_score", "suggestion_distance",
                         "compact_show")
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the warning text for the load-time preferences that widened a guard (see [`apply_preferences!`](@ref)). One line per key: the key, the default it replaced, and the value the project asked for.

A preference file is data. It ships with a cloned project or a template, it is often untracked, and [`__init__`](@ref PortfolioOptimisers.__init__) applies it at `using PortfolioOptimisers`, before any user code runs. A value that *tightens* a guard needs no announcement, so the warning names the widened guards alone: the [`RESOURCE_LIMITS`](@ref) and [`EQUATION_LIMITS`](@ref) caps a file raised, and a [`STRING_DISTANCE`](@ref) suggestion threshold it lowered (a lower threshold admits more candidates, which is the info-leak direction of `docs/adr/0026-lenient-constraint-names-with-suggestions.md`).

Never interpolates the whole preference dictionary, so a key the message does not name stays out of the log — the same info-leak-safe message discipline as [`unknown_variable_msg`](@ref).

# Algorithm

 1. Open `msg` with the number of widened guards and the sentence that says when a preference applies.
 2. Append one line per triple, naming the key, the default it replaced, and the value the project asked for.
 3. Close `msg` with the two repairs: delete the key, or widen the guard for one scope with a `with_*` helper.

# Arguments

  - `relaxations`: One `(key, default, value)` triple per widened guard, in [`PREFERENCE_KEYS`](@ref) order.

# Returns

  - `msg::String`: Multi-line warning text, one line per triple.

# Related

  - [`apply_preferences!`](@ref)
  - [`PREFERENCE_KEYS`](@ref)
  - [`unknown_variable_msg`](@ref)
"""
function relaxed_preferences_msg(relaxations::AbstractVector)
    msg = "$(length(relaxations)) load-time preference(s) of the active project widened a PortfolioOptimisers guard. Preferences apply at `using PortfolioOptimisers`, before any user code runs, so these values hold for the whole session:"
    for (key, default, val) in relaxations
        msg *= "\n  $(key): $(repr(default)) → $(repr(val))"
    end
    return msg *
           "\nThe values come from the `[PortfolioOptimisers]` section of the active project's `LocalPreferences.toml`. Delete a key there to restore the default, or widen the guard for one call only with `with_equation_limits`, `with_resource_limits` or `with_string_distance`."
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply load-time preference values to the global config defaults ([`EQUATION_LIMITS`](@ref), [`RESOURCE_LIMITS`](@ref), [`STRING_DISTANCE`](@ref), [`COMPACT_SHOW`](@ref)). Called by the package `__init__` with the [`PREFERENCE_KEYS`](@ref) values read via `Preferences.load_preference`; `nothing` values (unset preferences) are skipped and keep the shipped default.

Fails closed on an *invalid* value: it throws a typed `ArgumentError` naming the key and value, so the package refuses to load rather than silently running with a value the project got wrong. Values are applied through the `set_*!` setters, so they receive the same validation as runtime calls.

A *valid* value is applied whatever its size — the caps exist to turn an OOM kill into a typed error, not to second-guess a sizing choice, and a project on a large machine may legitimately raise one. A value that widens a guard is announced with a `@warn` built by [`relaxed_preferences_msg`](@ref), because the channel needs no code: a `LocalPreferences.toml` is data, it travels with a cloned project, and it applies before any user code runs. Widening means a raised [`RESOURCE_LIMITS`](@ref) or [`EQUATION_LIMITS`](@ref) cap, or a lowered [`STRING_DISTANCE`](@ref) suggestion threshold. A value that tightens a guard, or that equals the default it replaces, is silent. The comparison is against the default *in effect when the preference is applied*, which at load is the shipped default. See the amendment of `docs/adr/0041-one-resource-cap-per-sink.md`.

To persist a configuration, put the keys in the active project's `LocalPreferences.toml`, e.g.:

```toml
[PortfolioOptimisers]
equation_max_length = 512
equation_max_depth = 64
max_n_sim = 50_000
max_n_subsets = 1_000
max_frontier = 1_000
max_bins = 500
max_hop_count = 100
max_search_grid = 10_000
suggestion_min_score = 0.8
suggestion_distance = "damerau_levenshtein"
compact_show = 4
```

# Algorithm

 1. Start `relaxations` empty. It collects one `(key, default, value)` triple per guard the preferences widen.
 2. Read the two equation keys. When either is set, check that every set value is a positive integer, read the current [`EQUATION_LIMITS`](@ref) default, record a triple for each value above its default, and apply both through [`set_equation_limits!`](@ref). A key the project left unset keeps the value it already has.
 3. Read the six resource keys and repeat step 2 against [`RESOURCE_LIMITS`](@ref) and [`set_resource_limits!`](@ref).
 4. Read `"suggestion_min_score"`. When it is set, check that it is a real number, record a triple when it is below the current threshold, and apply it through [`set_string_distance!`](@ref). A lower threshold widens the guard, which is the opposite direction from a cap.
 5. Read `"suggestion_distance"`. When it is set, check that it is a string, and look it up in [`PREFERENCE_DISTANCES`](@ref). An unknown name raises, and the message carries a [`did_you_mean`](@ref) suggestion. Apply the resolved distance through [`set_string_distance!`](@ref).
 6. Read `"compact_show"`. When it is set, check that it is a boolean or an integer, and apply it through [`set_compact_show!`](@ref). This key guards nothing, so it records no triple.
 7. When `relaxations` is not empty, emit the text of [`relaxed_preferences_msg`](@ref) as a warning.

# Arguments

  - `prefs`: One entry per key of [`PREFERENCE_KEYS`](@ref). A `nothing` value means the project set no preference for that key, and the shipped default stands.

# Validation

  - Each of the eight cap keys is a positive integer that is not a `Bool`.
  - `"suggestion_min_score"` is a real number that is not a `Bool`.
  - `"suggestion_distance"` is a string, and it names an entry of [`PREFERENCE_DISTANCES`](@ref).
  - `"compact_show"` is a `Bool` or an `Integer`.
  - A breach of any rule above raises an `ArgumentError` that names the key and the value, so the package refuses to load.

# Returns

  - `nothing`.

# Related

  - [`PREFERENCE_KEYS`](@ref)
  - [`PREFERENCE_DISTANCES`](@ref)
  - [`set_equation_limits!`](@ref)
  - [`set_resource_limits!`](@ref)
  - [`set_string_distance!`](@ref)
  - [`set_compact_show!`](@ref)
  - [`relaxed_preferences_msg`](@ref)
"""
function apply_preferences!(prefs::AbstractDict{<:AbstractString, <:Any})
    relaxations = Vector{Tuple{String, Any, Any}}()
    ml = get(prefs, "equation_max_length", nothing)
    md = get(prefs, "equation_max_depth", nothing)
    if !(isnothing(ml) && isnothing(md))
        for (key, val) in ("equation_max_length" => ml, "equation_max_depth" => md)
            @argcheck(isnothing(val) || val isa Integer && !(val isa Bool) && val > 0,
                      ArgumentError("preference `$(key) = $(repr(val))` must be a positive integer."))
        end
        lim = @atomic EQUATION_LIMITS.default
        for (key, val, default) in (("equation_max_length", ml, lim.max_length),
                                    ("equation_max_depth", md, lim.max_depth))
            if !isnothing(val) && val > default
                push!(relaxations, (key, default, val))
            end
        end
        set_equation_limits!(; max_length = something(ml, lim.max_length),
                             max_depth = something(md, lim.max_depth))
    end
    xs = get(prefs, "max_n_sim", nothing)
    xb = get(prefs, "max_n_subsets", nothing)
    xf = get(prefs, "max_frontier", nothing)
    xn = get(prefs, "max_bins", nothing)
    xh = get(prefs, "max_hop_count", nothing)
    xg = get(prefs, "max_search_grid", nothing)
    if !all(isnothing, (xs, xb, xf, xn, xh, xg))
        for (key, val) in ("max_n_sim" => xs, "max_n_subsets" => xb, "max_frontier" => xf,
                           "max_bins" => xn, "max_hop_count" => xh, "max_search_grid" => xg)
            @argcheck(isnothing(val) || val isa Integer && !(val isa Bool) && val > 0,
                      ArgumentError("preference `$(key) = $(repr(val))` must be a positive integer."))
        end
        rlim = @atomic RESOURCE_LIMITS.default
        for (key, val, default) in
            (("max_n_sim", xs, rlim.max_n_sim), ("max_n_subsets", xb, rlim.max_n_subsets),
             ("max_frontier", xf, rlim.max_frontier), ("max_bins", xn, rlim.max_bins),
             ("max_hop_count", xh, rlim.max_hop_count),
             ("max_search_grid", xg, rlim.max_search_grid))
            if !isnothing(val) && val > default
                push!(relaxations, (key, default, val))
            end
        end
        set_resource_limits!(; max_n_sim = something(xs, rlim.max_n_sim),
                             max_n_subsets = something(xb, rlim.max_n_subsets),
                             max_frontier = something(xf, rlim.max_frontier),
                             max_bins = something(xn, rlim.max_bins),
                             max_hop_count = something(xh, rlim.max_hop_count),
                             max_search_grid = something(xg, rlim.max_search_grid))
    end
    ms = get(prefs, "suggestion_min_score", nothing)
    if !isnothing(ms)
        @argcheck(ms isa Real && !(ms isa Bool),
                  ArgumentError("preference `suggestion_min_score = $(repr(ms))` must be a real number."))
        msd = (@atomic STRING_DISTANCE.default).min_score
        if ms < msd
            push!(relaxations, ("suggestion_min_score", msd, ms))
        end
        set_string_distance!(; min_score = ms)
    end
    dn = get(prefs, "suggestion_distance", nothing)
    if !isnothing(dn)
        @argcheck(dn isa AbstractString,
                  ArgumentError("preference `suggestion_distance = $(repr(dn))` must be a string."))
        dist = get(PREFERENCE_DISTANCES, dn, nothing)
        if isnothing(dist)
            throw(ArgumentError("preference `suggestion_distance = $(repr(dn))` is not one of the $(length(PREFERENCE_DISTANCES)) supported distance names ($(join(sort!(collect(keys(PREFERENCE_DISTANCES))), ", ")))" *
                                did_you_mean(dn, collect(keys(PREFERENCE_DISTANCES)))))
        end
        set_string_distance!(; dist = dist)
    end
    cs = get(prefs, "compact_show", nothing)
    if !isnothing(cs)
        @argcheck(cs isa Bool || cs isa Integer,
                  ArgumentError("preference `compact_show = $(repr(cs))` must be a boolean or an integer."))
        set_compact_show!(cs)
    end
    if !isempty(relaxations)
        @warn relaxed_preferences_msg(relaxations)
    end
    return nothing
end
"""
    __init__()

Package load hook: reads the [`PREFERENCE_KEYS`](@ref) preferences of the active project via `Preferences.load_preference` and applies them to the global config defaults through [`apply_preferences!`](@ref). An invalid preference value fails closed — the package refuses to load — rather than running with a value the project got wrong.

This is the one channel that reaches the guards without running code: a `LocalPreferences.toml` is data, it travels with a cloned project or a template, and it is read here, before any user code. A valid value is therefore applied but not silent — a value that widens a guard is announced with a warning (see [`relaxed_preferences_msg`](@ref)).

# Algorithm

 1. Read every key of [`PREFERENCE_KEYS`](@ref) with `Preferences.load_preference`, giving `nothing` for a key the active project did not set.
 2. Pass the resulting dictionary to [`apply_preferences!`](@ref), which validates each value and applies it.

# Returns

  - `nothing`.

# Related

  - [`apply_preferences!`](@ref)
  - [`PREFERENCE_KEYS`](@ref)
  - [`relaxed_preferences_msg`](@ref)
"""
function __init__()
    return apply_preferences!(Dict{String, Any}(key =>
                                                    Preferences.load_preference(@__MODULE__,
                                                                                key,
                                                                                nothing)
                                                for key in PREFERENCE_KEYS))
end
