"""
    did_you_mean(name::AbstractString, candidates;
                 sd::StringDistanceConfig = STRING_DISTANCE[]) -> String

Return a "did you mean" suffix that names the closest match to `name` in `candidates`. Return an empty string when `candidates` is empty, or when no candidate reaches the `min_score` threshold of `sd`. The suffix is `" (did you mean X?)"`, with the match in the place of X.

Do not put the suffix in a code span that also holds escaped backticks. `JuliaFormatter` pairs the backticks wrongly and deletes the spaces around the code spans next to it, and the docstring then renders wrongly.

[`unknown_variable_msg`](@ref) adds this suffix to a "variable not in asset universe" message, to suggest the correct name for a typo. By default, the function reads the distance and the threshold from the active [`STRING_DISTANCE`](@ref) configuration. A caller with settings of its own passes them in `sd`, as [`suggest_declared_key`](@ref) does. The constructor of [`StringDistanceConfig`](@ref) checks `min_score`, so `sd` cannot hold a threshold that echoes every name. [`set_string_distance!`](@ref) sets the global default, and [`with_string_distance`](@ref) sets a task-scoped override. A cluster or a subset of a meta-optimiser can correctly omit a name, and that name has no close neighbour, so the threshold gives it no suggestion.

# Algorithm

 1. Return an empty string when `candidates` is empty, because the function has nothing to search.
 2. Find `match`, the entry of `candidates` nearest to `name` under `sd.dist`. Keep it only when its normalised similarity reaches `sd.min_score`.
 3. Return an empty string when step 2 finds no match. Otherwise return the suffix that names `match`.

# Arguments

  - `name::AbstractString`: The unknown name that the caller wrote.
  - `candidates`: The collection of valid names to search.
  - `sd::StringDistanceConfig`: The distance and the threshold. The default is the active [`STRING_DISTANCE`](@ref) configuration.

# Returns

  - `msg::String`: The suffix `" (did you mean X?)"`, or an empty string when no candidate reaches the threshold.

# Related

  - [`STRING_DISTANCE`](@ref)
  - [`set_string_distance!`](@ref)
  - [`unknown_variable_msg`](@ref)
"""
function did_you_mean(name::AbstractString, candidates;
                      sd::StringDistanceConfig = STRING_DISTANCE[])
    if isempty(candidates)
        return ""
    end
    match, _ = StringDistances.findnearest(name, candidates, sd.dist;
                                           min_score = sd.min_score)
    return isnothing(match) ? "" : " (did you mean `$(match)`?)"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Suggest the entry of `candidates` nearest to a mistyped declaration key. A declaration key is a macro block key, a dictionary key, a struct field name, or a keyword of a generated constructor.

The function passes [`did_you_mean`](@ref) a configuration that admits more matches than the global default. It overrides any scoped configuration of the caller. It uses Damerau-Levenshtein, where a transposed pair costs one edit and not two, at `min_score = 0.5`. The global default is strict so that a guessed name near a real asset name does not get the real name back. That risk does not apply here. The candidates are constants of the code, such as block keys, dictionary keys and field names, and they leak no data. At the default `0.7` under plain Levenshtein, a short key never matches. `nuon` scores 0.5 against `noun`, so the suggestion never shows.

# Arguments

  - `key`: The mistyped declaration key. The function converts it to a `String`.
  - `candidates`: The collection of valid keys. The function converts each key to a `String`.

# Returns

  - `msg::String`: The suffix that [`did_you_mean`](@ref) returns under this configuration.

# Related

  - [`did_you_mean`](@ref)
  - [`StringDistanceConfig`](@ref)
  - [`windowed_estimator_suggest`](@ref)
  - [`check_propagatable_contracts`](@ref)
"""
function suggest_declared_key(key, candidates)
    return did_you_mean(string(key), string.(collect(candidates));
                        sd = StringDistanceConfig(StringDistances.DamerauLevenshtein(),
                                                  0.5))
end
"""
    unknown_variable_msg(v, nx, key; candidates = nx, axis::AbstractString = "asset",
                         consequence::AbstractString = "term dropped") -> String

Build the warning or error text for a constraint or view variable `v` that is not in the universe `nx`, which the asset sets hold under `key`. The message names the variable and the size of the universe, and never the full universe. It ends with a [`did_you_mean`](@ref) suggestion when a close match exists.

`candidates` is the set of names that the function searches for the typo suggestion. The default is the universe `nx`. Some callers accept more names than the universe holds. [`name_to_val!`](@ref) is one of them, because a key can name a group and not an asset. Such a caller passes the asset names together with the group and set keys, so that the suggestion can name a mistyped group. The message always gives `length(nx)` as the size of the universe, whatever `candidates` holds.

`axis` names the universe that the function searched for the variable. The default is `"asset"`, because each constraint resolved against the assets before [`ExposureConstraintEstimator`](@ref) existed. A re-based constraint resolves its names against the factor universe and passes `"factor"`, so the message names the axis that the user wrote in.

`consequence` states what the failure removed, because the unit that the library drops depends on the shape. A value with the name as its key loses only that entry. [`name_to_val!`](@ref) is that case, and it uses the default `"term dropped"`. A row is one statement over many names with one right-hand side, so the library drops the whole row. A fit of `a + c == 0.05` as `a == 0.05` states a constraint that the caller did not write. [`get_linear_constraints`](@ref) passes `"row dropped"`. The text "term dropped" in that case sent a reader to look for a row that the library still fitted.

[`get_linear_constraints`](@ref), the generation of Black-Litterman views, the generation of entropy-pooling views and [`name_to_val!`](@ref) all call this function. The text of the message, and the rule that it leaks no data, are thus in one place.

# Arguments

  - `v`: The variable name that is not in the universe.
  - `nx`: The universe that the function searched. Only its length goes into the message.
  - `key`: The key of the universe in the asset sets.
  - `candidates = nx`: The names that the function searches for the typo suggestion.
  - `axis::AbstractString = "asset"`: The name of the universe that the function searched.
  - `consequence::AbstractString = "term dropped"`: What the failure removed. It is `"row dropped"` where the row is the unit.

# Returns

  - `msg::String`: The diagnostic text, with a [`did_you_mean`](@ref) suffix when a close match exists.

# Related

  - [`did_you_mean`](@ref)
  - [`empty_row_msg`](@ref)
  - [`empty_projected_row_msg`](@ref)
"""
function unknown_variable_msg(v, nx, key; candidates = nx, axis::AbstractString = "asset",
                              consequence::AbstractString = "term dropped")
    return "variable `$(v)` not in $(axis) universe ($(length(nx)) $(axis)s under key `$(key)`); $(consequence)" *
           did_you_mean(string(v), candidates)
end
"""
    empty_row_msg(eqn, nx, key; noun::AbstractString = "constraint",
                  axis::AbstractString = "asset") -> String

Build the warning or error text for a parsed equation `eqn` whose names all resolved against the universe `nx`, which the asset sets hold under `key`, but whose row is all zero. The library drops that row. The message names the equation and the size of the universe, and never the full universe or the parsed struct. `noun` is `"constraint"` for a linear constraint or `"view"` for a Black-Litterman or an entropy pooling view. `axis` names the universe, as in [`unknown_variable_msg`](@ref).

A name that is not in the universe never gets to this message. The row is the unit, so a name that does not resolve removes the whole row at that name, and [`unknown_variable_msg`](@ref) reports it with `consequence = "row dropped"`. A row that gets to this message summed to zero for a different reason. Its coefficients cancel, as in `A - A == 0.0`, which parses to the single term `0.0*A`. Or the row holds no variable, as in `1 == 0.004`. A typo message for either case sends the user to look for a typo that does not exist. [`empty_projected_row_msg`](@ref) and [`zero_centrality_msg`](@ref) follow the same rule.

[`get_linear_constraints`](@ref), the generation of Black-Litterman views, and the covariance and correlation views of entropy pooling ([`ep_cov_views!`](@ref), [`ep_rho_views!`](@ref)) call this function.

# Arguments

  - `eqn`: The parsed equation that resolved and still summed to zero.
  - `nx`: The universe that the terms resolved against. Only its length goes into the message.
  - `key`: The key of the universe in the asset sets.
  - `noun::AbstractString = "constraint"`: `"constraint"` for a linear constraint, and `"view"` for a Black-Litterman or an entropy pooling view.
  - `axis::AbstractString = "asset"`: The name of the universe, as in [`unknown_variable_msg`](@ref).

# Returns

  - `msg::String`: The diagnostic text.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`empty_projected_row_msg`](@ref)
"""
function empty_row_msg(eqn, nx, key; noun::AbstractString = "constraint",
                       axis::AbstractString = "asset")
    return "$(noun) `$(eqn)` resolved every name against the universe ($(length(nx)) $(axis)s under key `$(key)`) and still summed to zero, so it constrains nothing; row dropped"
end
"""
    empty_projected_row_msg(eqn, nf, key, n; noun::AbstractString = "constraint") -> String

Build the warning or error text for a re-based equation `eqn` whose terms resolved against the factor universe `nf`, which the sets hold under `key`. The projection of the row through the loadings is all zero over `n` assets.

This failure occurs only under a re-basis, and it is different from the failure of [`empty_row_msg`](@ref). In that failure, the coefficients of the row sum to zero. In this failure, they do not, and the projection through the basis makes them zero. The cause is a factor that no asset loads on. A message about the arithmetic of the equation sends the user to check coefficients that are correct.

# Arguments

  - `eqn`: The re-based equation whose projection is an all-zero row.
  - `nf`: The factor universe that the terms resolved against. Only its length goes into the message.
  - `key`: The key of the factor universe in the sets.
  - `n`: The number of assets of the projected row.
  - `noun::AbstractString = "constraint"`: `"constraint"` for a linear constraint, and `"view"` for a view.

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

Build the warning or error text for a centrality constraint whose centrality vector holds no information. The library drops the row that the constraint builds from it.

The vector is empty or all zero. In both cases the row is `0' w ⋄ B`, which does not depend on the weights, so it constrains nothing that the weights can change. The cause is a property of the graph, and never a mistyped name. Two assets under [`BetweennessCentrality`](@ref) give the zero vector, because no vertex is on a shortest path between two other vertices. A message from [`unknown_variable_msg`](@ref) or from a function of its family sends the reader to look for a typo that does not exist. For that reason, this case has its own message.

The message names the centrality algorithm and the length of the vector, and never its entries. [`empty_row_msg`](@ref) follows the same rule to leak no data.

# Arguments

  - `alg`: The centrality algorithm that made the vector. Only its type name goes into the message.
  - `n`: The length of the centrality vector. Zero means that the vector is empty.

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
    return "centrality constraint under `$(nameof(typeof(alg)))` contributes no row because $(cause): the row it would build does not depend on the weights; row dropped"
end
"""
    gross_budget_bounds_msg(lb, ub) -> String

Build the error text for a gross budget `gbgt` whose weight bounds `lb` and `ub` admit no short position. With no negative bound, the gross exposure is equal to the net exposure. The net budget `bgt` then already sets the constraint, and `gbgt` adds nothing.

The message names the size of the bounds and the failed condition, and never the values of the bounds. [`unknown_variable_msg`](@ref) and the other functions of its family follow the same rule to leak no data. A scalar bound or an absent bound has no size, so the message then names the bounds with no count.

# Algorithm

 1. Take `n`, the larger of the two bound lengths. A bound that is not a vector counts as zero.
 2. Build `scope`. It names the bounds alone when `n` is zero, and it names them with the asset count in the other case.
 3. Build the message from the fixed explanation and `scope`.

# Arguments

  - `lb`: The lower weight bound. Only its length goes into the message.
  - `ub`: The upper weight bound. Only its length goes into the message.

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
    strict_diagnostic(msg::AbstractString, strict::Bool; id,
                      exception::Type{<:Exception} = ArgumentError) -> Nothing

Report a term that cannot give a row. Under `strict`, throw an `exception`. In the other case, log a warning. In both cases the library drops the term.

A builder of linear constraint rows passes `exception = IsZeroError` when it drops a row whose coefficients are all zero. Such a row does not depend on the weights, so it is true for all weights or for none. The builder drops it at construction time in both cases.

`strict` applies to three kinds of term that the library can drop. The first is a name that resolves against nothing. The second is a row whose coefficients hold no information, as with a zero centrality vector. The third is a holding that a factor attribution cannot attribute, in an asset that the prior did not estimate or at an observation with no return. The library refuses nothing else. A malformed entry always throws, because the library has no other meaning of it to use. Every such diagnostic of the library goes through this function, so one edit changes the strictness for all of them.

The warning passes `maxlog = log_limit(:warn)` and `_id = id`. The caller gives `id` through [`@log_id`](@ref), because only the caller knows its site and the stable parts of its message. The default `_id` is the line of this function, and every caller shares that line.

# Algorithm

 1. Throw an `exception` that holds `msg` when `strict` is `true`. The call ends there.
 2. Otherwise log `msg` as a warning, under the cap of [`log_limit`](@ref) and the `_id` of the caller, and return.

# Arguments

  - `msg`: The diagnostic text, from [`unknown_variable_msg`](@ref) or from a function of its family.
  - `strict`: If `true`, the function throws an `exception`. If `false`, it logs a warning.
  - `id`: The `_id` of the warning, which [`@log_id`](@ref) makes at the site of the caller.
  - `exception`: The exception type that `strict` throws. It is [`IsZeroError`](@ref) for a row whose coefficients are all zero, and `ArgumentError` in the other cases.

# Validation

  - The call raises an `exception` that holds `msg` when `strict` is `true`.

# Returns

  - `nothing`.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`missing_group_assets_msg`](@ref)
  - [`empty_row_msg`](@ref)
  - [`zero_centrality_msg`](@ref)
  - [`IsZeroError`](@ref)
  - [`@log_id`](@ref)
"""
function strict_diagnostic(msg::AbstractString, strict::Bool; id,
                           exception::Type{<:Exception} = ArgumentError)::Nothing
    if strict
        throw(exception(msg))
    end
    @warn(msg, maxlog = log_limit(:warn), _id = id)
    return nothing
end
"""
    missing_group_assets_msg(group, missing_assets, nx, key) -> String

Build the warning or error text for a `group` that resolves in the asset sets, but whose members
`missing_assets` are not in the asset universe `nx`, which the sets hold under `key`. The message
names the group, the missing member names, and the size of the universe. The member names are input
from the caller, so they leak no data. The message never names the full universe or the dictionary
of input values. It ends with a [`did_you_mean`](@ref) suggestion for the first missing member.

[`name_to_val!`](@ref) calls this function. The text of the message, and the rule that it leaks no
data, are thus in one place, next to [`unknown_variable_msg`](@ref) and [`empty_row_msg`](@ref).

# Arguments

  - `group`: The group name that resolved in the asset sets.
  - `missing_assets`: The member names that are not in the asset universe.
  - `nx`: The asset universe. Only its length goes into the message.
  - `key`: The key of the asset universe in the asset sets.

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

Build the error text for a universe, declared under `key`, that does not agree with the axis `sym` of the data that the library uses it with. The function compares `declared` with `names`.

The position is the only link between a name and a column. A disagreement attaches every constraint row, bound and group to the wrong column, and the optimisation succeeds with the wrong answer. The message therefore names the repair, and not only the fault.

The message reports two kinds of disagreement differently, because they have different causes. Different lengths mean that the two describe different universes. The usual cause is old `sets` with new sliced data. Equal lengths mean that the two describe the same universe in a different order, and the first position that differs is the full diagnosis. The message names the sizes and the first pair that differs, and never one of the universes in full. [`unknown_variable_msg`](@ref) follows the same rule to leak no data.

# Algorithm

 1. Build `detail` through the branch that the two lengths select.
 2. When the lengths differ, `detail` names both counts and nothing more, because the two describe different universes.
 3. When the lengths are equal, find `i`, the first position at which the two differ. `detail` then names the common count, `i`, and the pair at `i`.
 4. Build the message from `detail`, the axis, the key, and the repair to make.

# Arguments

  - `declared`: The universe declared under `key`.
  - `names`: The axis of the data that the library uses the universe with.
  - `axis`: The name of the axis, for example `"asset"`.
  - `key`: The key of the declared universe in the sets.
  - `sym`: The field of the returns data that holds the correct axis. The repair names it.

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

Render the first line of an error for a log message, cut to `max_line_length` characters. A `…` at the end marks the cut. An exception renders through `showerror`, so the line holds the exception type and message. Each other value renders through `repr`.

# Algorithm

 1. Render `err` into `s`, through `showerror` when `err` is an `Exception` and through `repr` in the other case.
 2. Take `line`, the text of `s` up to its first newline.
 3. Return `line` unchanged when it has at most `max_line_length` characters. Otherwise return its first `max_line_length` characters and `…`.

# Arguments

  - `err`: The error to render.
  - `max_line_length::Integer`: The maximum number of characters of the returned line before the cut.

# Returns

  - `line::String`: The rendered first line, cut when it is too long.

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

Build the warning text for a JuMP model that no configured solver solved to a satisfactory status. `JuMPResult` holds the trials. The text has one line for each failed stage of each solver trial. The line gives the solver name, the stage that failed, and the first line of the error, cut to `max_line_length` characters. The stage is `set_optimizer`, `optimize!` or `assert_is_solved_and_feasible`. The cut keeps the JuMP termination status in the line.

The function never interpolates the whole trials dictionary, the solver settings, or the full exception data into the log. The returned `JuMPResult.trials` still holds the raw data. [`unknown_variable_msg`](@ref) and the other functions of its family follow the same rule to leak no data. The function sorts the solver names and the stages, so the message is the same on each run.

# Algorithm

 1. Start `msg` with the trial count, so that the reader sees how many solvers ran before the detail.
 2. For each solver name, in sorted order, read its entry into `trial`.
 3. Read `stages`, the stages of `trial`. An entry that is not a dictionary becomes the single stage `:trial`, so a solver that failed before the library recorded a stage still gives a line.
 4. For each stage, in sorted order, skip `:settings`. The solver settings are input from the caller, and they never go into a log.
 5. Add one line for each other stage. The line gives the solver name, the stage, and the first line of its error from [`first_error_line`](@ref).

# Arguments

  - `trials::AbstractDict`: One entry for each solver trial, with the solver name as its key.
  - `max_line_length::Integer = 200`: The maximum number of characters of each error line.

# Returns

  - `msg::String`: The warning text, with one line for each failed stage.

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
