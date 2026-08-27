"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all constraint result types.

All concrete and/or abstract types representing the result of constraint generation or evaluation should be subtypes of `AbstractConstraintResult`.

# Interfaces

In order to implement a new constraint result which will work seamlessly with the library, subtype `AbstractConstraintResult` and hold the assembled constraint in its fields. Both methods below carry a library-wide fallback, so a result the fallback already describes implements neither, and a result the fallback describes wrongly implements one or both.

## Routing target

  - `PortfolioOptimisers.implicit_constraint_target(res::AbstractConstraintResult) -> Option{Symbol}`: The one [routing target](@ref PIPELINE_ROUTING_TARGETS) a result of this type can land in, or `nothing` when the type names none.

The fallback answers `nothing`, and [`add_constraint_result`](@ref) then pairs the value with the target its step declared, in a [`TargetedConstraint`](@ref). Implement the method only when the type names exactly one field of a [`JuMPOptimiser`](@ref), which is what lets the value be injected bare.

### Arguments

  - `res`: The constraint result.

### Returns

  - `target::Option{Symbol}`: One of [`PIPELINE_ROUTING_TARGETS`](@ref), or `nothing`.

## Asset view

  - `PortfolioOptimisers.port_opt_view(res::AbstractConstraintResult, i, args...) -> AbstractConstraintResult`: An asset-sliced copy of the result.

The generic fallback slices an array field along the asset axis, and `@propagatable`'s `@vprop` tag derives the method for a struct whose fields are all asset-parallel. Write the method by hand when the fallback is wrong for the type. [`LinearConstraint`](@ref) is the case: its method reads neither the index nor the tail that follows it, and the reason the identity is the right answer there is stated on the method itself.

### Arguments

  - `res`: The constraint result.
  - `i`: The asset index the view keeps.

### Returns

  - `res::AbstractConstraintResult`: The constraint over the selected assets.

# Related

  - [`AbstractConstraintEstimator`](@ref)
  - [`AbstractResult`](@ref)
  - [`constraint_results`](@ref)
  - [`implicit_constraint_target`](@ref)
  - [`port_opt_view`](@ref)
"""
abstract type AbstractConstraintResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all constraint estimator types.

All concrete and/or abstract types implementing constraint generation or estimation algorithms should be subtypes of `AbstractConstraintEstimator`.

# Interfaces

In order to implement a new constraint family which will work seamlessly with the library, subtype `AbstractConstraintEstimator` with all necessary parameters as part of the struct, and implement the following methods.

## Constraint generation

  - `<family>_constraints(ce::AbstractConstraintEstimator, sets::UniverseSets)`: Assemble the family's constraint over the declared universe.

The family owns the verb, and the library holds one verb per family rather than one shared name: [`linear_constraints`](@ref), [`weight_bounds_constraints`](@ref), [`threshold_constraints`](@ref), [`risk_budget_constraints`](@ref), [`asset_sets_matrix`](@ref), [`phylogeny_constraints`](@ref) and [`centrality_constraints`](@ref). A family that refits its structure from data takes a [`ReturnsResult`](@ref) in place of the sets. This is the method a caller reaches directly, and it is the one the pipeline step below delegates to.

### Arguments

  - `ce`: The constraint estimator.
  - `sets`: The declared universe the names resolve against.

### Returns

  - `res`: The assembled constraint. Every family but the asset sets matrix returns an [`AbstractConstraintResult`](@ref); [`asset_sets_matrix`](@ref) returns a plain membership matrix, and the [routing target](@ref PIPELINE_ROUTING_TARGETS) carries it instead of the type.

## Pipeline step

Implement both methods, or neither. [`pipe_constraint_targets`](@ref) falls back to an empty tuple, which declares that the family computes no value for the `constraints` slot and is therefore not a [`Pipeline`](@ref) step, and [`resolve_constraint_target`](@ref) raises on that tuple before the value is ever reached. [`JuMPConstraintEstimator`](@ref) is the family that takes the fallback: it is configuration for the model rather than a computation over data.

  - `PortfolioOptimisers.pipe_constraint_targets(ce::AbstractConstraintEstimator) -> Tuple`: The [routing targets](@ref PIPELINE_ROUTING_TARGETS) the family's result can land in. One target places the value with no annotation, several make the [`PipelineStep`](@ref) wrapper name one.
  - `PortfolioOptimisers.constraint_step_value(ce::AbstractConstraintEstimator, ctx::PipelineContext)`: The value the step contributes, computed by calling the family's own verb above.

[`pipe_reads`](@ref) and [`pipe_writes`](@ref) need no method. The family answers `(:returns,)` and `:constraints` for every subtype, and a family that reads a computed slot as well states so on its own type.

### Arguments

  - `ce`: The constraint estimator.
  - `ctx`: The pipeline context.

### Returns

  - `res`: The computed value, `nothing`, or a vector of either.

# Related

  - [`AbstractConstraintResult`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`constraint_step_value`](@ref)
  - [`pipe_constraint_targets`](@ref)
  - [`run_constraint_step`](@ref)
"""
abstract type AbstractConstraintEstimator <: AbstractEstimator end
"""
    const ComparisonOperator = Union{typeof(==), typeof(<=), typeof(>=)}

Union type representing supported comparison operators for constraint generation.

This type is used to specify which comparison operators are valid for defining constraints. It includes equality and both directions of inequality.

The group exists to be a **type bound** rather than a check. A field annotated with it refuses a fourth operator where the value is stored, so `CentralityConstraint(; comp = <)` raises a `TypeError` from the keyword constructor, and no constraint generator carries a branch for an operator it can never receive.

# Related

  - [`comparison_sign_ineq_flag`](@ref)
  - [`CentralityConstraint`](@ref)
"""
const ComparisonOperator = Union{typeof(==), typeof(<=), typeof(>=)}
"""
    comparison_sign_ineq_flag(op::ComparisonOperator)
    comparison_sign_ineq_flag(op::AbstractString)

Return the multiplicative sign and inequality flag for a given comparison operator.

This is the one table mapping a comparison operator to the pair every constraint generator needs: the sign that files a `>=` row as a `<=` row, and the flag that sorts the row into the inequality block or the equality block. A parsed constraint carries its operator as a `String`, and a constraint estimator carries it as a function, so the table dispatches on both spellings and neither caller writes its own copy.

# Arguments

  - `op::ComparisonOperator`: The comparison operator, as a function.
  - `op::AbstractString`: The comparison operator, as the string a [`ParsingResult`](@ref) carries.

# Validation

  - A string `op` is one of `"=="`, `"<="` or `">="`. Anything else raises an `ArgumentError` naming the three.
  - A function `op` outside [`ComparisonOperator`](@ref) matches no method and raises a `MethodError`. A field bounded by that alias refuses such an operator earlier, at construction.

# Returns

  - `sign::Int`: The multiplicative sign for the constraint.
  - `is_inequality::Bool`: `true` if the operator is an inequality, `false` for equality.

# Examples

```jldoctest
julia> PortfolioOptimisers.comparison_sign_ineq_flag(==)
(1, false)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(<=)
(1, true)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(>=)
(-1, true)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(\">=\")
(-1, true)
```

# Related

  - [`ComparisonOperator`](@ref)
  - [`ParsingResult`](@ref)
"""
function comparison_sign_ineq_flag(::typeof(==))::Tuple{Int, Bool}
    return 1, false
end
function comparison_sign_ineq_flag(::typeof(<=))::Tuple{Int, Bool}
    return 1, true
end
function comparison_sign_ineq_flag(::typeof(>=))::Tuple{Int, Bool}
    return -1, true
end
function comparison_sign_ineq_flag(op::AbstractString)::Tuple{Int, Bool}
    return if op == "=="
        1, false
    elseif op == "<="
        1, true
    elseif op == ">="
        -1, true
    else
        throw(ArgumentError("`op` must be one of \"==\", \"<=\", \">=\". Got\nop => $op"))
    end
end
"""
    resolve_axis_name(name, nx::AbstractVector, sdict::AbstractDict) -> Option{Vector}

Resolve one name against an axis: an axis entry names itself, a `sdict` key expands to its members, and an unknown name gives `nothing`.

This is the precedence every name-taking constraint generator uses — asset first, then group — written once. The caller diagnoses the `nothing`, because the suggestion pool differs by caller. The member vector is a **copy**: `sdict` is the caller's [`UniverseSets`](@ref) dictionary, which is configuration reused across folds and optimisers, so de-duplication must never edit it.

# Algorithm

 1. When `name` is on `nx`, return `[name]`. The axis is read before the dictionary, so an asset and a group of one name resolve to the asset, and a group of that name is unreachable.
 2. Otherwise read `members`, the entry `sdict` holds for `name`, or `nothing` when it holds none.
 3. Return `unique(members)`, so a group naming one asset twice contributes one member.

# Arguments

  - `name`: The name to resolve.
  - `nx`: The axis, usually the asset universe.
  - `sdict`: Dictionary mapping group names to vectors of member names.

# Returns

  - `[name]` if `name` is on the axis.
  - `unique(members)` if `name` is a key of `sdict`.
  - `nothing` if `name` is neither.

# Examples

An asset name wins over a group of the same name, and a repeated member collapses.

```jldoctest
julia> nx = [\"A\", \"B\", \"C\"];

julia> sdict = Dict(\"A\" => [\"B\", \"C\"], \"G\" => [\"A\", \"A\", \"B\"]);

julia> PortfolioOptimisers.resolve_axis_name(\"A\", nx, sdict)
1-element Vector{String}:
 \"A\"

julia> PortfolioOptimisers.resolve_axis_name(\"G\", nx, sdict)
2-element Vector{String}:
 \"A\"
 \"B\"

julia> isnothing(PortfolioOptimisers.resolve_axis_name(\"Q\", nx, sdict))
true
```

# Related

  - [`axis_name_indices`](@ref)
  - [`name_to_val!`](@ref)
  - [`estimator_to_val`](@ref)
  - [`UniverseSets`](@ref)
"""
function resolve_axis_name(name, nx::AbstractVector, sdict::AbstractDict)
    if name in nx
        return [name]
    end
    members = get(sdict, name, nothing)
    return isnothing(members) ? nothing : unique(members)
end
"""
    axis_name_indices(members, nx::AbstractVector, on_missing) -> Vector

Map resolved member names to axis indices, drop the members that miss the axis, and report them once through `on_missing`.

`on_missing` takes the vector of missing members and decides the policy, so a caller can throw, warn, or stay silent without a second copy of the mapping. `on_missing` is not called when every member is on the axis.

# Algorithm

 1. Map each member to `idx`, its first index on `nx`, or to `nothing` when it is on no index. A repeated axis name is therefore reachable only at its first index.
 2. Collect `missing_members`, the members whose index is `nothing`.
 3. Drop the `nothing`s from `idx`.
 4. When `missing_members` is not empty, call `on_missing` on it. The call comes **after** step 3, so a callback that throws throws with the surviving indices already discarded, and a callback that warns leaves the caller holding them.

# Arguments

  - `members`: Member names, as [`resolve_axis_name`](@ref) returns them.
  - `nx`: The axis, usually the asset universe.
  - `on_missing`: Callable applied to the vector of members that miss the axis.

# Returns

  - `idx`: Axis indices of the members that are on the axis, in the order of `members`. The element type is `Int` when no member missed the axis, and `Union{Nothing, Int}` when one did, because step 3 removes the `nothing`s from the vector and not from its type.

# Examples

```jldoctest
julia> nx = [\"A\", \"B\", \"C\"];

julia> PortfolioOptimisers.axis_name_indices([\"C\", \"A\"], nx, m -> nothing)
2-element Vector{Int64}:
 3
 1

julia> missed = String[];

julia> PortfolioOptimisers.axis_name_indices([\"A\", \"Z\", \"C\"], nx, m -> append!(missed, m))
2-element Vector{Union{Nothing, Int64}}:
 1
 3

julia> missed
1-element Vector{String}:
 \"Z\"
```

# Related

  - [`resolve_axis_name`](@ref)
  - [`missing_group_assets_msg`](@ref)
"""
function axis_name_indices(members, nx::AbstractVector, on_missing)
    idx = [findfirst(x -> isequal(x, m), nx) for m in members]
    missing_members = members[isnothing.(idx)]
    filter!(!isnothing, idx)
    if !isempty(missing_members)
        on_missing(missing_members)
    end
    return idx
end
