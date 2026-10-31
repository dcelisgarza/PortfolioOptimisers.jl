"""
    PREFERENCE_DISTANCES

Allowlist that maps each name that the `"suggestion_distance"` preference accepts to its `StringDistances.StringDistance` object. One `Dict` holds the accepted names and the objects that they select, as the function allowlist of the equation parser does. An unknown name stops the load with an `ArgumentError`, and the message has a [`did_you_mean`](@ref) suggestion.

The accepted names are `"levenshtein"`, `"damerau_levenshtein"`, `"jaro"`, `"jaro_winkler"` and `"ratcliff_obershelp"`.

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

The Preferences.jl keys that the package reads at load time to set the global configuration defaults, through [`apply_preferences!`](@ref).

  - `"equation_max_length"` and `"equation_max_depth"`: Positive integers for [`EQUATION_LIMITS`](@ref).
  - `"max_n_sim"`, `"max_n_subsets"`, `"max_frontier"`, `"max_bins"`, `"max_hop_count"`, `"max_search_grid"` and `"max_ep_grid"`: Positive integers for [`RESOURCE_LIMITS`](@ref).
  - `"suggestion_min_score"`: A real number for the threshold of [`STRING_DISTANCE`](@ref).
  - `"suggestion_distance"`: A name from [`PREFERENCE_DISTANCES`](@ref) for the distance of [`STRING_DISTANCE`](@ref).
  - `"compact_show"`: A boolean or an integer for [`COMPACT_SHOW`](@ref).
  - `"show_nothing_fields"`: A boolean for the global switch of [`SHOW_NOTHING_FIELDS`](@ref).
  - `"show_nothing_fields_by_type"`: A table of booleans, one for each type name, for the per-name entries of [`SHOW_NOTHING_FIELDS`](@ref).
  - `"max_warn"`, `"max_info"` and `"max_debug"`: Positive integers for [`LOG_LIMITS`](@ref).

Preferences.jl cannot list the keys that a project set. The package thus cannot find a misspelled key, and it ignores that key with no message, so the package default applies. A misspelled or invalid value under one of these keys stops the load. The package also cannot refuse a name in the `"show_nothing_fields_by_type"` table that matches no type. A type outside the package can print through [`@define_pretty_show`](@ref), so the package accepts an unknown name, and the entry has no effect.

The package applies each valid value, and it warns about a value that widens a guard, through [`relaxed_preferences_msg`](@ref). A preference file is data. It comes with a cloned project, and it applies before any user code runs.

# Related

  - [`apply_preferences!`](@ref)
  - [`relaxed_preferences_msg`](@ref)
"""
const PREFERENCE_KEYS = ("equation_max_length", "equation_max_depth", "max_n_sim",
                         "max_n_subsets", "max_frontier", "max_bins", "max_hop_count",
                         "max_search_grid", "max_ep_grid", "suggestion_min_score",
                         "suggestion_distance", "compact_show", "show_nothing_fields",
                         "show_nothing_fields_by_type", "max_warn", "max_info", "max_debug")
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the warning text for the load-time preferences that widened a guard, for [`apply_preferences!`](@ref). The text has one line for each key, with the key, the default that it replaced, and the value that the project set.

A preference file is data. It comes with a cloned project or a template, git often does not track it, and [`__init__`](@ref PortfolioOptimisers.__init__) applies it at `using PortfolioOptimisers`, before any user code runs. A value that tightens a guard needs no warning, so the warning names only the widened guards. These are a cap of [`RESOURCE_LIMITS`](@ref) or [`EQUATION_LIMITS`](@ref) that the file raised, and a suggestion threshold of [`STRING_DISTANCE`](@ref) that it lowered. A lower threshold admits more candidates, so it can leak more asset names.

The function never interpolates the whole preference dictionary, so a key that the message does not name stays out of the log. [`unknown_variable_msg`](@ref) follows the same rule to leak no data.

# Algorithm

 1. Start `msg` with the number of widened guards and the sentence that says when a preference applies.
 2. Add one line for each triple. The line names the key, the default that it replaced, and the value that the project set.
 3. End `msg` with the two repairs. The user can delete the key, or widen the guard for one scope with a `with_*` function.

# Arguments

  - `relaxations`: One `(key, default, value)` triple for each widened guard, in the order of [`PREFERENCE_KEYS`](@ref).

# Returns

  - `msg::String`: The warning text, with one line for each triple.

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

Apply the two load-time preferences of [`SHOW_NOTHING_FIELDS`](@ref), the global switch and the per-name table. [`apply_preferences!`](@ref) calls this function, and its docstring states the rules for all preferences. The function skips a `nothing` value, which is a preference that the project did not set, and the package default stays.

An invalid value raises an `ArgumentError` that names the key and the value, as in [`apply_preferences!`](@ref). The function does not refuse a name in the table that matches no type. [`PREFERENCE_KEYS`](@ref) gives the reason.

# Algorithm

 1. Read `"show_nothing_fields"`. When it is set, check that it is a `Bool`, and apply it through the one-argument form of [`set_show_nothing_fields!`](@ref).
 2. Read `"show_nothing_fields_by_type"`. When it is set, check that it is a table and that every value is a `Bool`. Convert each key to a `Symbol`, and apply each entry through the per-name form of [`set_show_nothing_fields!`](@ref).

# Arguments

  - `prefs`: The dictionary that [`apply_preferences!`](@ref) gets. The function reads only the two keys above.

# Validation

  - `"show_nothing_fields"` is a `Bool`.
  - `"show_nothing_fields_by_type"` is a dictionary, and every value in it is a `Bool`.
  - A breach of either rule raises an `ArgumentError` that names the key and the value.

# Returns

  - `nothing`.

# Related

  - [`apply_preferences!`](@ref)
  - [`PREFERENCE_KEYS`](@ref)
  - [`set_show_nothing_fields!`](@ref)
  - [`SHOW_NOTHING_FIELDS`](@ref)
"""
function apply_show_preferences!(prefs::AbstractDict{<:AbstractString, <:Any})
    sn = get(prefs, "show_nothing_fields", nothing)
    if !isnothing(sn)
        @argcheck(sn isa Bool,
                  ArgumentError("preference `show_nothing_fields = $(repr(sn))` must be a boolean."))
        set_show_nothing_fields!(sn)
    end
    sb = get(prefs, "show_nothing_fields_by_type", nothing)
    if !isnothing(sb)
        @argcheck(sb isa AbstractDict,
                  ArgumentError("preference `show_nothing_fields_by_type = $(repr(sb))` must be a table of booleans, one per type name."))
        for (name, val) in sb
            @argcheck(val isa Bool,
                      ArgumentError("preference `show_nothing_fields_by_type.$(name) = $(repr(val))` must be a boolean."))
            set_show_nothing_fields!(Symbol(name), val)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply the three load-time preferences of [`LOG_LIMITS`](@ref), `"max_warn"`, `"max_info"` and `"max_debug"`. [`apply_preferences!`](@ref) calls this function, and its docstring states the rules for all preferences. A `nothing` value is a preference that the project did not set, and the package default stays.

An invalid value raises an `ArgumentError` that names the key and the value, as in [`apply_preferences!`](@ref). A valid value gives no message. A cap limits only the repeats of a message and never hides the first one. A cap thus widens no guard, and records no triple for [`relaxed_preferences_msg`](@ref). TOML has no `nothing`, so a project gets the package default back when it deletes the key.

# Algorithm

 1. Read the three keys. When none is set, return.
 2. Check that every set value is a positive integer that is not a `Bool`.
 3. Apply the three values through [`set_log_limits!`](@ref). A key that the project did not set keeps the value that it already has.

# Arguments

  - `prefs`: The dictionary that [`apply_preferences!`](@ref) gets. The function reads only the three keys above.

# Validation

  - Each of `"max_warn"`, `"max_info"` and `"max_debug"` is a positive integer that is not a `Bool`.
  - A breach raises an `ArgumentError` that names the key and the value.

# Returns

  - `nothing`.

# Related

  - [`apply_preferences!`](@ref)
  - [`PREFERENCE_KEYS`](@ref)
  - [`set_log_limits!`](@ref)
  - [`LOG_LIMITS`](@ref)
"""
function apply_log_preferences!(prefs::AbstractDict{<:AbstractString, <:Any})
    vals = map(key -> get(prefs, key, nothing), ("max_warn", "max_info", "max_debug"))
    if all(isnothing, vals)
        return nothing
    end
    for (key, val) in zip(("max_warn", "max_info", "max_debug"), vals)
        @argcheck(isnothing(val) || val isa Integer && !(val isa Bool) && val > 0,
                  ArgumentError("preference `$(key) = $(repr(val))` must be a positive integer."))
    end
    lim = @atomic LOG_LIMITS.default
    set_log_limits!(; max_warn = something(vals[1], Some(lim.max_warn)),
                    max_info = something(vals[2], Some(lim.max_info)),
                    max_debug = something(vals[3], Some(lim.max_debug)))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply the load-time preference values to the global configuration defaults. These are [`EQUATION_LIMITS`](@ref), [`RESOURCE_LIMITS`](@ref), [`STRING_DISTANCE`](@ref), [`COMPACT_SHOW`](@ref), [`SHOW_NOTHING_FIELDS`](@ref) and [`LOG_LIMITS`](@ref). The package `__init__` calls this function with the values of [`PREFERENCE_KEYS`](@ref) that `Preferences.load_preference` reads. The function skips a `nothing` value, which is a preference that the project did not set, and the package default stays.

An invalid value raises an `ArgumentError` that names the key and the value. The package then does not load, and it does not run with a wrong value. The function applies each value through its `set_*!` function, so a preference gets the same validation as a call at run time.

The function applies a valid value of any size. A cap turns a kill by the out-of-memory killer into an error with a type. It does not judge a size, and a project on a large machine can correctly raise a cap. A value that widens a guard gives a `@warn` with the text of [`relaxed_preferences_msg`](@ref), because a preference needs no code to change a guard. A `LocalPreferences.toml` is data, it comes with a cloned project, and it applies before any user code runs. A widened guard is a raised cap of [`RESOURCE_LIMITS`](@ref) or [`EQUATION_LIMITS`](@ref), or a lowered suggestion threshold of [`STRING_DISTANCE`](@ref). A value that tightens a guard, or that is equal to the default that it replaces, gives no message. The function compares each value with the default that is active when it applies the preference. At load, that is the package default.

To keep a configuration, put the keys in the `LocalPreferences.toml` of the active project, as in this example:

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
max_ep_grid = 500
suggestion_min_score = 0.8
suggestion_distance = "damerau_levenshtein"
compact_show = 4
show_nothing_fields = true
max_warn = 1
max_info = 1
max_debug = 10

[PortfolioOptimisers.show_nothing_fields_by_type]
SimpleVariance = false
```

# Algorithm

 1. Start `relaxations` as an empty vector. It collects one `(key, default, value)` triple for each guard that the preferences widen.
 2. Read the two equation keys. When one of them is set, check that every set value is a positive integer, and read the current default of [`EQUATION_LIMITS`](@ref). Record a triple for each value above its default, and apply both values through [`set_equation_limits!`](@ref). A key that the project did not set keeps the value that it already has.
 3. Read the seven resource keys, and do step 2 again with [`RESOURCE_LIMITS`](@ref) and [`set_resource_limits!`](@ref).
 4. Read `"suggestion_min_score"`. When it is set, check that it is a real number. Record a triple when it is below the current threshold, and apply it through [`set_string_distance!`](@ref). A lower threshold widens the guard, which is the opposite direction to a cap.
 5. Read `"suggestion_distance"`. When it is set, check that it is a string, and find it in [`PREFERENCE_DISTANCES`](@ref). An unknown name raises an error, and the message has a [`did_you_mean`](@ref) suggestion. Apply the distance that the name selects through [`set_string_distance!`](@ref).
 6. Read `"compact_show"`. When it is set, check that it is a boolean or an integer, and apply it through [`set_compact_show!`](@ref). This key is not a guard, so it records no triple.
 7. Pass `prefs` to [`apply_show_preferences!`](@ref), which reads `"show_nothing_fields"` and `"show_nothing_fields_by_type"`. Neither key is a guard, so neither records a triple.
 8. Pass `prefs` to [`apply_log_preferences!`](@ref), which reads `"max_warn"`, `"max_info"` and `"max_debug"`. A cap never hides the first message of a site, so none of the three records a triple.
 9. When `relaxations` is not empty, log the text of [`relaxed_preferences_msg`](@ref) as a warning, under the cap that step 8 set.

# Arguments

  - `prefs`: One entry for each key of [`PREFERENCE_KEYS`](@ref). A `nothing` value means that the project set no preference for that key, and the package default stays.

# Validation

  - Each of the nine cap keys is a positive integer that is not a `Bool`.
  - `"suggestion_min_score"` is a real number that is not a `Bool`.
  - `"suggestion_distance"` is a string, and it names an entry of [`PREFERENCE_DISTANCES`](@ref).
  - `"compact_show"` is a `Bool` or an `Integer`.
  - `"show_nothing_fields"` is a `Bool`, and `"show_nothing_fields_by_type"` is a table of `Bool` values. [`apply_show_preferences!`](@ref) checks both.
  - Each of `"max_warn"`, `"max_info"` and `"max_debug"` is a positive integer that is not a `Bool`. [`apply_log_preferences!`](@ref) checks them.
  - A breach of a rule above raises an `ArgumentError` that names the key and the value, and the package does not load.

# Returns

  - `nothing`.

# Related

  - [`PREFERENCE_KEYS`](@ref)
  - [`PREFERENCE_DISTANCES`](@ref)
  - [`set_equation_limits!`](@ref)
  - [`set_resource_limits!`](@ref)
  - [`set_string_distance!`](@ref)
  - [`set_compact_show!`](@ref)
  - [`apply_show_preferences!`](@ref)
  - [`apply_log_preferences!`](@ref)
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
    xe = get(prefs, "max_ep_grid", nothing)
    if !all(isnothing, (xs, xb, xf, xn, xh, xg, xe))
        for (key, val) in ("max_n_sim" => xs, "max_n_subsets" => xb, "max_frontier" => xf,
                           "max_bins" => xn, "max_hop_count" => xh, "max_search_grid" => xg,
                           "max_ep_grid" => xe)
            @argcheck(isnothing(val) || val isa Integer && !(val isa Bool) && val > 0,
                      ArgumentError("preference `$(key) = $(repr(val))` must be a positive integer."))
        end
        rlim = @atomic RESOURCE_LIMITS.default
        for (key, val, default) in
            (("max_n_sim", xs, rlim.max_n_sim), ("max_n_subsets", xb, rlim.max_n_subsets),
             ("max_frontier", xf, rlim.max_frontier), ("max_bins", xn, rlim.max_bins),
             ("max_hop_count", xh, rlim.max_hop_count),
             ("max_search_grid", xg, rlim.max_search_grid),
             ("max_ep_grid", xe, rlim.max_ep_grid))
            if !isnothing(val) && val > default
                push!(relaxations, (key, default, val))
            end
        end
        set_resource_limits!(; max_n_sim = something(xs, rlim.max_n_sim),
                             max_n_subsets = something(xb, rlim.max_n_subsets),
                             max_frontier = something(xf, rlim.max_frontier),
                             max_bins = something(xn, rlim.max_bins),
                             max_hop_count = something(xh, rlim.max_hop_count),
                             max_search_grid = something(xg, rlim.max_search_grid),
                             max_ep_grid = something(xe, rlim.max_ep_grid))
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
    apply_show_preferences!(prefs)
    apply_log_preferences!(prefs)
    if !isempty(relaxations)
        @warn(relaxed_preferences_msg(relaxations), maxlog = log_limit(:warn),
              _id = @log_id(first.(relaxations)))
    end
    return nothing
end
"""
    __init__()

Load hook of the package. It reads the preferences of [`PREFERENCE_KEYS`](@ref) for the active project with `Preferences.load_preference`, and applies them to the global configuration defaults through [`apply_preferences!`](@ref). An invalid preference value stops the load, and the package does not run with a wrong value.

A preference is the one way to change a guard with no code. A `LocalPreferences.toml` is data, it comes with a cloned project or a template, and this hook reads it before any user code runs. The package thus applies a valid value, and it warns about a value that widens a guard, through [`relaxed_preferences_msg`](@ref).

# Algorithm

 1. Read every key of [`PREFERENCE_KEYS`](@ref) with `Preferences.load_preference`. The value is `nothing` for a key that the active project did not set.
 2. Pass the dictionary to [`apply_preferences!`](@ref), which checks each value and applies it.

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
