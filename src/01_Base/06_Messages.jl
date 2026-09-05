"""
    did_you_mean(name::AbstractString, candidates) -> String

Return a "did you mean" suffix naming the closest match to `name` among `candidates`, or an empty string when no candidate reaches the global [`STRING_DISTANCE`](@ref) `min_score` threshold (or `candidates` is empty). The suffix reads `" (did you mean X?)"`, with the match in place of X.

Do not wrap the suffix in a code span that also carries escaped backticks. `JuliaFormatter` mis-pairs the backticks and deletes the spaces around the neighbouring code spans, which breaks the rendering.

Used to enrich "variable not in asset universe" messages (see [`unknown_variable_msg`](@ref)) with a typo suggestion. The distance and threshold are read from the active [`STRING_DISTANCE`](@ref) config — global default via [`set_string_distance!`](@ref), task-scoped override via [`with_string_distance`](@ref); the threshold gating means a name legitimately absent from a meta-optimiser cluster/subset (no close neighbour) draws no suggestion.

# Algorithm

 1. Return an empty string when `candidates` is empty, because there is nothing to search.
 2. Read the active [`STRING_DISTANCE`](@ref) configuration into `sd`.
 3. Search `candidates` for the entry nearest to `name` under `sd.dist`, keeping only a match whose normalised similarity reaches `sd.min_score`, giving `match`.
 4. Return an empty string when step 3 finds no match. Otherwise return the suffix that names `match`.

# Arguments

  - `name::AbstractString`: The offending name the caller wrote.
  - `candidates`: Collection of valid names to search.

# Returns

  - `msg::String`: The suffix `" (did you mean X?)"`, or an empty string when no candidate reaches the threshold.

# Related

  - [`STRING_DISTANCE`](@ref)
  - [`set_string_distance!`](@ref)
  - [`unknown_variable_msg`](@ref)
"""
function did_you_mean(name::AbstractString, candidates)
    if isempty(candidates)
        return ""
    end
    sd = STRING_DISTANCE[]
    match, _ = StringDistances.findnearest(name, candidates, sd.dist;
                                           min_score = sd.min_score)
    return isnothing(match) ? "" : " (did you mean `$(match)`?)"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Suggest the nearest `candidates` entry to a mistyped **declaration key**: a macro block key, a dictionary key, a struct field name, or a keyword of a generated constructor.

Wraps [`did_you_mean`](@ref) in a looser scoped configuration than the global default: Damerau-Levenshtein (so a transposed pair costs one edit, not two) at `min_score = 0.5`. The strict global default exists to keep near-miss probes from echoing real *asset names* back to the caller (ADR 0026); that boundary does not apply here, because the candidates are compile-time constants — block keys, dictionary keys and field names — with nothing to leak. At the default `0.7` under plain Levenshtein, short keys never match: `nuon` scores 0.5 against `noun`, so the suggestion would be dead code.

# Arguments

  - `key`: The mistyped declaration key, converted to a `String`.
  - `candidates`: Collection of valid keys, each converted to a `String`.

# Returns

  - `msg::String`: The suffix that [`did_you_mean`](@ref) returns under the looser configuration.

# Related

  - [`did_you_mean`](@ref)
  - [`with_string_distance`](@ref)
  - [`windowed_estimator_suggest`](@ref)
  - [`check_propagatable_contracts`](@ref)
"""
function suggest_declared_key(key, candidates)
    return with_string_distance(; dist = StringDistances.DamerauLevenshtein(),
                                min_score = 0.5) do
        return did_you_mean(string(key), string.(collect(candidates)))
    end
end
"""
    unknown_variable_msg(v, nx, key; candidates = nx, axis = "asset") -> String

Build the warning/error text for a constraint or view variable `v` that is absent from the universe `nx` (stored under `key`). Names the variable and the universe *size* only — never the full universe — and appends a [`did_you_mean`](@ref) suggestion when a close match exists.

`candidates` is the pool searched for the typo suggestion (default: the universe `nx`). Callers whose valid namespace is broader than the raw universe — e.g. [`name_to_val!`](@ref), where a key may name a *group* rather than an asset — pass a wider pool (asset names plus group/set keys) so the suggestion can name a mistyped group. The reported universe *size* is always `length(nx)` regardless of `candidates`.

`axis` names the universe the variable was looked up in. It defaults to `"asset"` because that is the axis every constraint resolved against before [`ExposureConstraintEstimator`](@ref); a re-based constraint resolves its names against the *factor* universe and passes `"factor"`, so the message names the axis the user actually wrote in.

Shared by [`get_linear_constraints`](@ref), Black-Litterman view generation, entropy-pooling view generation, and [`name_to_val!`](@ref) so the message (and its info-leak-safe shape) lives in exactly one place.

# Arguments

  - `v`: The variable name that is absent from the universe.
  - `nx`: The universe the lookup failed against. Only its length reaches the message.
  - `key`: The key the universe is stored under.
  - `candidates = nx`: Pool searched for the typo suggestion.
  - `axis::AbstractString = "asset"`: Name of the universe the variable was looked up in.

# Returns

  - `msg::String`: The diagnostic text, with a [`did_you_mean`](@ref) suffix when a close match exists.

# Related

  - [`did_you_mean`](@ref)
  - [`empty_row_msg`](@ref)
  - [`empty_projected_row_msg`](@ref)
"""
function unknown_variable_msg(v, nx, key; candidates = nx, axis::AbstractString = "asset")
    return "variable `$(v)` not in $(axis) universe ($(length(nx)) $(axis)s under key `$(key)`); term dropped" *
           did_you_mean(string(v), candidates)
end
"""
    empty_row_msg(eqn, nx, key; noun::AbstractString = "constraint",
                  axis::AbstractString = "asset") -> String

Build the warning/error text for a parsed equation `eqn` whose every term missed the universe `nx` (stored under `key`), leaving an all-zero row that is dropped. Names the equation and the universe *size* only — never the full universe or the parsed struct. `noun` is `"constraint"` for linear constraints or `"view"` for Black-Litterman views; `axis` names the universe, as in [`unknown_variable_msg`](@ref).

Shared by [`get_linear_constraints`](@ref) and Black-Litterman view generation.

# Arguments

  - `eqn`: The parsed equation whose every term missed the universe.
  - `nx`: The universe the terms missed. Only its length reaches the message.
  - `key`: The key the universe is stored under.
  - `noun::AbstractString = "constraint"`: `"constraint"` for a linear constraint, `"view"` for a Black-Litterman view.
  - `axis::AbstractString = "asset"`: Name of the universe, as in [`unknown_variable_msg`](@ref).

# Returns

  - `msg::String`: The diagnostic text.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`empty_projected_row_msg`](@ref)
"""
function empty_row_msg(eqn, nx, key; noun::AbstractString = "constraint",
                       axis::AbstractString = "asset")
    return "$(noun) `$(eqn)` matched no $(axis)s in the universe ($(length(nx)) $(axis)s under key `$(key)`); row dropped"
end
"""
    empty_projected_row_msg(eqn, nf, key, n; noun::AbstractString = "constraint") -> String

Build the warning/error text for a re-based equation `eqn` whose terms *did* resolve against the factor universe `nf` (stored under `key`), but whose projection through the loadings is an all-zero row over `n` assets.

This diagnosis exists only under a re-basis, and it is a different failure from [`empty_row_msg`](@ref): there the names missed the universe, here they hit it and the basis annihilated them. Reporting the first for the second would send a user hunting for a typo that is not there — the real cause is a factor no asset loads on.

# Arguments

  - `eqn`: The re-based equation whose projection is an all-zero row.
  - `nf`: The factor universe the terms resolved against. Only its length reaches the message.
  - `key`: The key the factor universe is stored under.
  - `n`: Number of assets the row was projected over.
  - `noun::AbstractString = "constraint"`: `"constraint"` for a linear constraint, `"view"` for a view.

# Returns

  - `msg::String`: The diagnostic text.

# Related

  - [`empty_row_msg`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
"""
function empty_projected_row_msg(eqn, nf, key, n; noun::AbstractString = "constraint")
    return "$(noun) `$(eqn)` resolved against the factor universe ($(length(nf)) factors under key `$(key)`) but projected to an all-zero row over $(n) assets: every matched factor has zero loadings; row dropped"
end
"""
    zero_centrality_msg(alg, n) -> String

Build the warning/error text for a centrality constraint whose centrality vector carries no information, so the row it would build is dropped.

The vector is either empty or all zero. Either way the row reads `0' w <= B`, which every set of weights satisfies, so it constrains nothing. This is a fact about the graph and never a mistyped name: two assets under [`BetweennessCentrality`](@ref) give the zero vector, because no vertex lies on a shortest path between two others. Reporting it through [`unknown_variable_msg`](@ref) or one of its siblings would send a reader hunting for a typo that is not there, which is why this message is its own.

Names the centrality algorithm and the *length* of the vector only — never its entries — the same info-leak-safe discipline as [`empty_row_msg`](@ref).

# Arguments

  - `alg`: The centrality algorithm that produced the vector. Only its type name reaches the message.
  - `n`: Length of the centrality vector. Zero means the vector is empty.

# Returns

  - `msg::String`: The diagnostic text.

# Related

  - [`strict_diagnostic`](@ref)
  - [`empty_row_msg`](@ref)
  - [`empty_projected_row_msg`](@ref)
"""
function zero_centrality_msg(alg, n)
    cause = if iszero(n)
        "its centrality vector is empty"
    else
        "all $(n) entries of its centrality vector are zero"
    end
    return "centrality constraint under `$(nameof(typeof(alg)))` contributes no row because $(cause): the row it would build holds for every set of weights; row dropped"
end
"""
    gross_budget_bounds_msg(lb, ub) -> String

Build the error text for a gross budget (`gbgt`) whose weight bounds `lb` and `ub` admit no short position. With no negative bound the gross exposure equals the net exposure, so the net budget (`bgt`) already owns the constraint and `gbgt` has nothing left to express.

Names the *size* of the bounds and the failed predicate only — never the bound values — the same info-leak-safe discipline as [`unknown_variable_msg`](@ref) and its siblings. Scalar or absent bounds have no size, so the message names the bounds without a count.

# Algorithm

 1. Take `n`, the greater of the two bound lengths, counting a bound that is not a vector as zero.
 2. Build `scope`, which names the bounds alone when `n` is zero and names them with the asset count otherwise.
 3. Build the message from the fixed explanation and `scope`.

# Arguments

  - `lb`: Lower weight bound. Only its length reaches the message.
  - `ub`: Upper weight bound. Only its length reaches the message.

# Returns

  - `msg::String`: The error text.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`assert_gross_budget_admissible`](@ref)
  - [`w_neg_flag`](@ref)
"""
function gross_budget_bounds_msg(lb, ub)
    n = max(isa(lb, AbstractVector) ? length(lb) : 0,
            isa(ub, AbstractVector) ? length(ub) : 0)
    scope = iszero(n) ? "weight bounds" : "weight bounds over $(n) assets"
    return "gross budget (gbgt) requires weight bounds that admit short positions: with non-negative bounds no short weights exist, so the gross exposure equals the net exposure and the net budget (bgt) already constrains it. Got $(scope) with no negative element in lb or ub."
end
"""
    strict_diagnostic(msg::AbstractString, strict::Bool) -> Nothing

Report a **term that cannot contribute a row**: throw an `ArgumentError` under `strict`, warn otherwise, and in both cases the offending term is dropped.

`strict` governs what is droppable: a name that resolves against nothing, and a row whose coefficients carry no information, as with a zero centrality vector. Nothing else is refused, and a malformed *entry* throws unconditionally, because there is no reading of it to fall back to. Every such diagnostic in the library routes through here, so the strictness policy is one edit.

# Algorithm

 1. Throw an `ArgumentError` carrying `msg` when `strict` is `true`, which ends the call.
 2. Otherwise emit `msg` as a warning and return.

# Arguments

  - `msg`: The diagnostic text, built by [`unknown_variable_msg`](@ref) or one of its siblings.
  - `strict`: If `true`, throws an `ArgumentError`; if `false`, issues a warning.

# Validation

  - The call raises an `ArgumentError` carrying `msg` when `strict` is `true`.

# Returns

  - `nothing`.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`missing_group_assets_msg`](@ref)
  - [`empty_row_msg`](@ref)
  - [`zero_centrality_msg`](@ref)
"""
function strict_diagnostic(msg::AbstractString, strict::Bool)::Nothing
    if strict
        throw(ArgumentError(msg))
    end
    @warn(msg)
    return nothing
end
"""
    missing_group_assets_msg(group, missing_assets, nx, key) -> String

Build the warning/error text for a `group` that resolves in the asset sets but whose members
`missing_assets` are absent from the asset universe `nx` (stored under `key`). Names the group, the
offending member names (which are caller input, not internal state), and the universe *size* only —
never the full universe or the input value dictionary — and appends a [`did_you_mean`](@ref)
suggestion for the first missing member.

Shared by [`name_to_val!`](@ref) so the info-leak-safe message shape lives in exactly one place,
alongside [`unknown_variable_msg`](@ref) and [`empty_row_msg`](@ref).

# Arguments

  - `group`: The group name that resolved in the asset sets.
  - `missing_assets`: The member names absent from the asset universe.
  - `nx`: The asset universe. Only its length reaches the message.
  - `key`: The key the asset universe is stored under.

# Returns

  - `msg::String`: The diagnostic text, with a [`did_you_mean`](@ref) suffix for the first missing member.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`empty_row_msg`](@ref)
  - [`did_you_mean`](@ref)
"""
function missing_group_assets_msg(group, missing_assets, nx, key)
    return "group `$(group)`: $(length(missing_assets)) member(s) not in asset universe " *
           "($(length(nx)) assets under key `$(key)`): $(missing_assets); dropped" *
           did_you_mean(string(first(missing_assets)), nx)
end
"""
    misaligned_axis_msg(declared, names, axis, key, sym) -> String

Build the error text for a universe declared under `key` that disagrees with the axis `sym` of the data it will be used against — `declared` against `names`.

Position is the only link between a name and a column, so a disagreement is not a naming inconvenience: every constraint row, bound and group would be attached to the wrong column and the optimisation would succeed with the wrong answer. The message therefore names what to fix, not just what is wrong.

Two disagreements are reported differently because they have different causes. Different lengths mean the two describe different universes — usually a stale `sets` against freshly sliced data. Equal lengths mean they describe the same universe in a different order, and the first differing position is the whole diagnosis. Names the sizes and the *first* differing pair only — never either universe in full, the same info-leak-safe discipline as [`unknown_variable_msg`](@ref).

# Algorithm

 1. Build `detail` through the branch that the two lengths select.
 2. When the lengths differ, `detail` names both counts and nothing else, because the two describe different universes.
 3. When the lengths agree, find `i`, the first position at which the two disagree, and let `detail` name the shared count, `i`, and the pair at `i`.
 4. Build the message from `detail`, the axis, the key, and the repair to make.

# Arguments

  - `declared`: The universe declared under `key`.
  - `names`: The axis of the data the universe is used against.
  - `axis`: Name of the axis, for example `"asset"`.
  - `key`: The key the declared universe is stored under.
  - `sym`: Field of the returns data that carries the correct axis, named in the repair.

# Returns

  - `msg::String`: The error text.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`missing_group_assets_msg`](@ref)
"""
function misaligned_axis_msg(declared, names, axis, key, sym)
    detail = if length(declared) != length(names)
        "$(length(declared)) $(axis)s are declared, but the data has $(length(names))"
    else
        i = findfirst(declared .!= names)
        "both have $(length(names)) $(axis)s but the order differs, first at position $(i): `$(declared[i])` vs `$(names[i])`"
    end
    return "the $(axis) universe declared under key `$(key)` does not describe the returns data: $(detail). Position is the only link between a name and a column, so this attaches constraints, bounds and groups to the wrong $(axis) rather than failing. Set `sets.dict[\"$(key)\"]` to `rd.$(sym)`, or slice the sets to match the data."
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Render the first line of an error for a log message, truncated to `max_line_length` characters (a trailing `…` marks the cut). Exceptions render via `showerror`, so the line carries the exception type and message; anything else renders via `repr`.

# Algorithm

 1. Render `err` into `s`, through `showerror` when `err` is an `Exception` and through `repr` otherwise.
 2. Take `line`, the text of `s` up to its first newline.
 3. Return `line` unchanged when it fits `max_line_length`, and otherwise return its first `max_line_length` characters followed by `…`.

# Arguments

  - `err`: The error to render.
  - `max_line_length::Integer`: Maximum number of characters the returned line may hold before the cut.

# Returns

  - `line::String`: The rendered first line, truncated when needed.

# Related

  - [`failed_solve_msg`](@ref)
"""
function first_error_line(err, max_line_length::Integer)
    s = err isa Exception ? sprint(showerror, err) : repr(err)
    line = String(first(split(s, '\n')))
    return length(line) <= max_line_length ? line : first(line, max_line_length) * "…"
end
"""
    failed_solve_msg(trials::AbstractDict; max_line_length::Integer = 200) -> String

Build the warning text for a JuMP model that no configured solver could solve satisfactorily (see `JuMPResult`). One line per failed stage of each solver trial: the solver name, the stage that failed (`set_optimizer`, `optimize!`, or `assert_is_solved_and_feasible`), and the first line of the error truncated to `max_line_length` characters — so a JuMP termination status stays visible.

Never interpolates the whole trials dictionary, the solver settings, or full exception payloads into the log; the raw data remains available on the returned `JuMPResult.trials`. This is the same info-leak-safe message discipline as [`unknown_variable_msg`](@ref) and its siblings. Solver names and stages are sorted so the message is deterministic.

# Algorithm

 1. Open `msg` with the trial count, so the reader sees how many solvers were tried before the detail.
 2. For each solver name, in sorted order, read its entry into `trial`.
 3. Read the stages of `trial`, giving `stages`. An entry that is not a dictionary is wrapped as the single stage `:trial`, so a solver that failed before any stage was recorded still reports.
 4. For each stage, in sorted order, skip `:settings`, because the solver settings are caller input and never reach a log.
 5. Append one line per remaining stage: the solver name, the stage, and the first line of its error from [`first_error_line`](@ref).

# Arguments

  - `trials::AbstractDict`: One entry per solver trial, keyed by solver name.
  - `max_line_length::Integer = 200`: Maximum number of characters of each error line.

# Returns

  - `msg::String`: The warning text, one line per failed stage.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`empty_row_msg`](@ref)
  - [`missing_group_assets_msg`](@ref)
  - [`first_error_line`](@ref)
"""
function failed_solve_msg(trials::AbstractDict; max_line_length::Integer = 200)
    msg = "Model could not be solved satisfactorily ($(length(trials)) solver trial(s))."
    for name in sort!(collect(keys(trials)); by = string)
        trial = trials[name]
        stages = trial isa AbstractDict ? trial : Dict{Symbol, Any}(:trial => trial)
        for stage in sort!(collect(keys(stages)); by = string)
            if stage === :settings
                continue
            end
            msg *= "\n  $(name): $(stage) → $(first_error_line(stages[stage], max_line_length))"
        end
    end
    return msg
end
